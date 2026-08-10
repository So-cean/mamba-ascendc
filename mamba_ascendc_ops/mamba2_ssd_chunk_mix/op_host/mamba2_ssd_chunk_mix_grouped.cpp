// Copyright (c) 2026, mamba-ascendc authors.
// Ascend 950PR H/G=4 ChunkMix with direct group-owned state output.

#include "mamba2_ssd_chunk_mix_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"

namespace {
constexpr int64_t kTile = 64;
constexpr int64_t kHeadsPerGroup = 4;

size_t WorkspaceBytesPerCore(bool isAscend950)
{
    // 910B key 11 packs four weighted-X heads as [T,4P], retains two W
    // ping-pong slots, and writes the wide state GEMM directly to the
    // group-owned output.  Only CB needs FP32 workspace.  The 950 key keeps
    // its existing ABI because its MatmulImpl path has a different address
    // map.
    const size_t halfTiles = 6 * kTile * kTile;
    const size_t floatTiles = (isAscend950 ? 5 : 1) * kTile * kTile;
    return halfTiles * sizeof(uint16_t) + floatTiles * sizeof(float);
}

bool BuildMatmulTiling(
    const platform_ascendc::PlatformAscendC &platform,
    int64_t m, int64_t n, int64_t k,
    optiling::TCubeTiling &tiling)
{
    matmul_tiling::MultiCoreMatmulTiling cube(platform);
    return cube.SetAType(matmul_tiling::TPosition::GM,
                         matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT16) == 0 &&
           cube.SetBType(matmul_tiling::TPosition::GM,
                         matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT16) == 0 &&
           cube.SetCType(matmul_tiling::TPosition::GM,
                         matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT) == 0 &&
           cube.SetBiasType(matmul_tiling::TPosition::GM,
                            matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT) == 0 &&
           cube.SetShape(m, n, k) == 0 &&
           cube.SetOrgShape(m, n, k) == 0 &&
           cube.SetDim(1) == 0 &&
           cube.SetSingleShape(m, n, k) == 0 &&
           cube.SetFixSplit(m, n, k) == 0 &&
           cube.EnableBias(false) == 0 &&
           cube.SetBufferSpace(-1, -1, -1) == 0 &&
           cube.GetTiling(tiling) != -1;
}

bool BuildBatchDiagTiling(
    const platform_ascendc::PlatformAscendC &platform,
    optiling::TCubeTiling &tiling)
{
    matmul_tiling::BatchMatmulTiling cube(platform);
    return cube.SetAType(matmul_tiling::TPosition::GM,
                         matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT16) == 0 &&
           cube.SetBType(matmul_tiling::TPosition::GM,
                         matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT16) == 0 &&
           cube.SetCType(matmul_tiling::TPosition::GM,
                         matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT) == 0 &&
           cube.SetBiasType(matmul_tiling::TPosition::GM,
                            matmul_tiling::CubeFormat::ND,
                            matmul_tiling::DataType::DT_FLOAT) == 0 &&
           cube.SetShape(kTile, kTile, kTile) == 0 &&
           cube.SetOrgShape(kTile, kTile, kTile) == 0 &&
           cube.SetBatchNum(kHeadsPerGroup) == 0 &&
           cube.EnableBias(false) == 0 &&
           cube.SetBufferSpace(-1, -1, -1) == 0 &&
           cube.GetTiling(tiling) != -1;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto x = context->GetInputTensor(0)->GetOriginShape();
    const auto da = context->GetInputTensor(1)->GetOriginShape();
    const auto b = context->GetInputTensor(2)->GetOriginShape();
    const auto c = context->GetInputTensor(3)->GetOriginShape();
    const bool groupedX = x.GetDimNum() == 6;
    if ((!groupedX && x.GetDimNum() != 5) || da.GetDimNum() != 4 ||
        b.GetDimNum() != 5 || c.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = x.GetDim(0);
    const int64_t chunks = groupedX ? x.GetDim(1) : x.GetDim(2);
    const int64_t groups = b.GetDim(2);
    const int64_t heads = groupedX
        ? groups * x.GetDim(4)
        : x.GetDim(1);
    const bool xCompatible = groupedX
        ? x.GetDim(1) == chunks && x.GetDim(2) == groups &&
              x.GetDim(3) == kTile &&
              x.GetDim(4) == kHeadsPerGroup && x.GetDim(5) == kTile
        : x.GetDim(2) == chunks && x.GetDim(3) == kTile &&
              x.GetDim(4) == kTile;
    const bool compatible =
        batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
        heads == groups * kHeadsPerGroup && xCompatible &&
        da.GetDim(0) == batch &&
        da.GetDim(1) == heads && da.GetDim(2) == chunks &&
        da.GetDim(3) == kTile && b.GetDim(0) == batch &&
        b.GetDim(1) == chunks && b.GetDim(3) == kTile &&
        b.GetDim(4) == kTile && c.GetDim(0) == batch &&
        c.GetDim(1) == chunks && c.GetDim(2) == groups &&
        c.GetDim(3) == kTile && c.GetDim(4) == kTile;
    if (!compatible) return ge::GRAPH_FAILED;

    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const bool isAscend950 = mamba2_compat::IsAscend950(platform);
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    if (aic == 0 || (isAscend950 ? aiv < aic : aiv < 2 * aic)) {
        return ge::GRAPH_FAILED;
    }
    // The 910B candidate deliberately changes only the state producer and
    // keeps the accepted head-major preprocess.  Grouped-X has a measured
    // 1.8% preprocess regression on 910B and is not selected here.
    if (!isAscend950 && groupedX) return ge::GRAPH_FAILED;
    const uint64_t headTasks64 =
        static_cast<uint64_t>(batch) * heads * chunks;
    const uint64_t groupTasks64 =
        static_cast<uint64_t>(batch) * groups * chunks;
    if (headTasks64 > UINT32_MAX || groupTasks64 == 0 ||
        groupTasks64 > UINT32_MAX || groupTasks64 < aic) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t headTasks = static_cast<uint32_t>(headTasks64);
    const uint32_t groupTasks = static_cast<uint32_t>(groupTasks64);
    const uint32_t used = groupTasks < aic ? groupTasks : aic;
    const uint32_t workspace = static_cast<uint32_t>(
        WorkspaceBytesPerCore(isAscend950));

    Mamba2SsdChunkMixTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_heads(static_cast<uint32_t>(heads));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_chunkSize(kTile);
    tiling.set_headDim(kTile);
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_stateDim(kTile);
    tiling.set_taskCount(groupTasks);
    tiling.set_headTaskCount(headTasks);
    tiling.set_usedCoreNum(used);
    tiling.set_workspaceBytesPerCore(workspace);
    tiling.set_taskMode(1);
    tiling.set_headsPerGroup(kHeadsPerGroup);
    tiling.set_stateNpLayout(1);
    tiling.set_inputGroupedX(groupedX ? 1U : 0U);
    if (isAscend950 &&
        (!BuildMatmulTiling(platform, kTile, kTile, kTile,
                            tiling.cubeTilingData) ||
         !BuildMatmulTiling(platform, kTile, 4 * kTile, kTile,
                            tiling.groupedStateCubeTilingData) ||
         (!groupedX &&
          !BuildBatchDiagTiling(platform, tiling.batchDiagCubeTilingData)))) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(used);
    // Key 11 is the 910B-native 1AIC:2AIV dynamic-MMAD path.  It avoids the
    // CANN 8.2 64x256 MatmulImpl/Fixpipe fault observed with key 9 while
    // still reducing four state GEMMs to one 64x256 operation.
    context->SetTilingKey(!isAscend950 ? 11 : (groupedX ? 9 : 10));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
        static_cast<size_t>(used) * workspace;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *x = context->GetInputShape(0);
    const auto *b = context->GetInputShape(2);
    auto *y = context->GetOutputShape(0);
    auto *state = context->GetOutputShape(1);
    if (x == nullptr || b == nullptr || y == nullptr || state == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *y = *x;
    if (x->GetDimNum() == 6) {
        *state = *x;
    } else {
        state->SetDimNum(6);
        state->SetDim(0, x->GetDim(0));
        state->SetDim(1, x->GetDim(2));
        state->SetDim(2, b->GetDim(2));
        state->SetDim(3, kTile);
        state->SetDim(4, kHeadsPerGroup);
        state->SetDim(5, kTile);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Mamba2SsdChunkMixGrouped : public OpDef {
public:
    explicit Mamba2SsdChunkMixGrouped(const char *name) : OpDef(name)
    {
        this->Input("x_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d_a_cumsum").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y_diag").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("chunk_states_grouped").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};
OP_ADD(Mamba2SsdChunkMixGrouped);
}  // namespace ops

namespace optiling {
REGISTER_TILING_DATA_CLASS(Mamba2SsdChunkMixGrouped,
                           Mamba2SsdChunkMixTilingData)
}  // namespace optiling
