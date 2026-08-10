// Copyright (c) 2026, mamba-ascendc authors.

#include "mamba2_ssd_state_projection_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kTile = 64;

bool BuildCubeTiling(const platform_ascendc::PlatformAscendC &platform,
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
           cube.SetShape(kTile, kTile, kTile) == 0 &&
           cube.SetOrgShape(kTile, kTile, kTile) == 0 &&
           cube.SetDim(1) == 0 &&
           cube.SetSingleShape(kTile, kTile, kTile) == 0 &&
           cube.SetFixSplit(kTile, kTile, kTile) == 0 &&
           cube.EnableBias(false) == 0 &&
           cube.SetBufferSpace(-1, -1, -1) == 0 &&
           cube.GetTiling(tiling) != -1;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *stateShape = context->GetInputShape(0);
    const auto *cShape = context->GetInputShape(1);
    const auto *stateDesc = context->GetInputDesc(0);
    const auto *cDesc = context->GetInputDesc(1);
    if (stateShape == nullptr || cShape == nullptr || stateDesc == nullptr ||
        cDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto state = stateShape->GetOriginShape();
    const auto c = cShape->GetOriginShape();
    if (state.GetDimNum() != 5 || c.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = state.GetDim(0);
    const int64_t heads = state.GetDim(1);
    const int64_t chunks = state.GetDim(2);
    const int64_t groups = c.GetDim(2);
    if (batch <= 0 || heads <= 0 || chunks <= 0 || groups <= 0 ||
        heads % groups != 0 || state.GetDim(3) != kTile ||
        state.GetDim(4) != kTile || c.GetDim(0) != batch ||
        c.GetDim(1) != chunks || c.GetDim(3) != kTile ||
        c.GetDim(4) != kTile) {
        return ge::GRAPH_FAILED;
    }

    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    if (aic == 0 || aiv < aic) return ge::GRAPH_FAILED;
    const uint64_t tasks64 = static_cast<uint64_t>(batch) * heads * chunks;
    if (tasks64 == 0 || tasks64 > UINT32_MAX) return ge::GRAPH_FAILED;
    const uint32_t tasks = static_cast<uint32_t>(tasks64);
    const uint32_t used = tasks < aic ? tasks : aic;

    Mamba2SsdStateProjectionTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_heads(static_cast<uint32_t>(heads));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_headsPerGroup(static_cast<uint32_t>(heads / groups));
    tiling.set_taskCount(tasks);
    tiling.set_usedCoreNum(used);
    tiling.set_workspaceBytesPerCore(
        static_cast<uint32_t>(kTile * kTile * sizeof(uint16_t)));
    if (!BuildCubeTiling(platform, tiling.cubeTilingData)) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(used);
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
        static_cast<size_t>(used) * kTile * kTile * sizeof(uint16_t);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *state = context->GetInputShape(0);
    auto *out = context->GetOutputShape(0);
    if (state == nullptr || out == nullptr || state->GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }
    out->SetDimNum(5);
    out->SetDim(0, state->GetDim(0));
    out->SetDim(1, state->GetDim(1));
    out->SetDim(2, state->GetDim(2));
    out->SetDim(3, kTile);
    out->SetDim(4, kTile);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Mamba2SsdStateProjection : public OpDef {
public:
    explicit Mamba2SsdStateProjection(const char *name) : OpDef(name)
    {
        this->Input("states_start").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_cube").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y_off").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc)
            .AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};
OP_ADD(Mamba2SsdStateProjection);
}  // namespace ops
