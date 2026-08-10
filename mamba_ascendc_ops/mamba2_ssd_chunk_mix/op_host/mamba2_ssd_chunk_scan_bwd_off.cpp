/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Host definition for the T=P=N=64 off-diagonal chunk-scan backward.
 */

#include "mamba2_ssd_chunk_scan_bwd_off_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kTile = 64;
constexpr uint32_t kTilingKey = 1;

size_t WorkspaceBytesPerCore()
{
    // Q[T,P], Q^T[P,T] and the FP16-cast state S[P,N].
    return static_cast<size_t>(3 * kTile * kTile) * sizeof(uint16_t);
}

bool BuildMatmulTiling(
    const platform_ascendc::PlatformAscendC &platform,
    optiling::TCubeTiling &tiling)
{
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(platform);
    return cubeTiling.SetAType(matmul_tiling::TPosition::GM,
                               matmul_tiling::CubeFormat::ND,
                               matmul_tiling::DataType::DT_FLOAT16) == 0 &&
           cubeTiling.SetBType(matmul_tiling::TPosition::GM,
                               matmul_tiling::CubeFormat::ND,
                               matmul_tiling::DataType::DT_FLOAT16) == 0 &&
           cubeTiling.SetCType(matmul_tiling::TPosition::GM,
                               matmul_tiling::CubeFormat::ND,
                               matmul_tiling::DataType::DT_FLOAT) == 0 &&
           cubeTiling.SetBiasType(matmul_tiling::TPosition::GM,
                                  matmul_tiling::CubeFormat::ND,
                                  matmul_tiling::DataType::DT_FLOAT) == 0 &&
           cubeTiling.SetShape(kTile, kTile, kTile) == 0 &&
           cubeTiling.SetOrgShape(kTile, kTile, kTile) == 0 &&
           cubeTiling.SetDim(1) == 0 &&
           cubeTiling.SetSingleShape(kTile, kTile, kTile) == 0 &&
           cubeTiling.SetFixSplit(kTile, kTile, kTile) == 0 &&
           cubeTiling.EnableBias(false) == 0 &&
           cubeTiling.SetBufferSpace(-1, -1, -1) == 0 &&
           cubeTiling.GetTiling(tiling) != -1;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto gyShape = context->GetInputTensor(0)->GetOriginShape();
    const auto stateShape = context->GetInputTensor(1)->GetOriginShape();
    const auto daShape = context->GetInputTensor(2)->GetOriginShape();
    const auto cShape = context->GetInputTensor(3)->GetOriginShape();
    if (gyShape.GetDimNum() != 5 || stateShape.GetDimNum() != 5 ||
        daShape.GetDimNum() != 4 || cShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = gyShape.GetDim(0);
    const int64_t heads = gyShape.GetDim(1);
    const int64_t chunks = gyShape.GetDim(2);
    const int64_t chunkSize = gyShape.GetDim(3);
    const int64_t headDim = gyShape.GetDim(4);
    const int64_t groups = cShape.GetDim(2);
    const bool compatible =
        batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
        heads % groups == 0 && chunkSize == kTile && headDim == kTile &&
        stateShape.GetDim(0) == batch && stateShape.GetDim(1) == heads &&
        stateShape.GetDim(2) == chunks &&
        stateShape.GetDim(3) == kTile && stateShape.GetDim(4) == kTile &&
        daShape.GetDim(0) == batch && daShape.GetDim(1) == heads &&
        daShape.GetDim(2) == chunks && daShape.GetDim(3) == kTile &&
        cShape.GetDim(0) == batch && cShape.GetDim(1) == chunks &&
        cShape.GetDim(3) == kTile && cShape.GetDim(4) == kTile;
    if (!compatible) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    // The installed kernel binary selects the architecture-specific MIX
    // topology: 910B uses 1 AIC + 2 AIVs, while Ascend 950 uses 1 AIC +
    // 1 AIV.  Host tiling only needs both engine types to be available;
    // rejecting anything below the 910B ratio incorrectly disables 950.
    if (aic == 0 || aiv == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tasks64 =
        static_cast<uint64_t>(batch) * heads * chunks;
    if (tasks64 == 0 || tasks64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tasks = static_cast<uint32_t>(tasks64);
    const uint32_t usedCores = tasks < aic ? tasks : aic;
    const size_t workspaceBytes = WorkspaceBytesPerCore();

    Mamba2SsdChunkScanBwdOffTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_heads(static_cast<uint32_t>(heads));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_taskCount(tasks);
    tiling.set_usedCoreNum(usedCores);
    tiling.set_workspaceBytesPerCore(
        static_cast<uint32_t>(workspaceBytes));
    tiling.set_headsPerGroup(static_cast<uint32_t>(heads / groups));
    if (!BuildMatmulTiling(platform, tiling.cubeTilingData)) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCores);
    context->SetTilingKey(kTilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
        static_cast<size_t>(usedCores) * workspaceBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *gyShape = context->GetInputShape(0);
    const auto *stateShape = context->GetInputShape(1);
    auto *dStateShape = context->GetOutputShape(0);
    auto *dCHeadShape = context->GetOutputShape(1);
    auto *gDaShape = context->GetOutputShape(2);
    if (gyShape == nullptr || stateShape == nullptr ||
        dStateShape == nullptr || dCHeadShape == nullptr ||
        gDaShape == nullptr || gyShape->GetDimNum() != 5 ||
        stateShape->GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }
    *dStateShape = *stateShape;
    *dCHeadShape = *gyShape;
    dCHeadShape->SetDim(4, stateShape->GetDim(4));
    gDaShape->SetDimNum(4);
    gDaShape->SetDim(0, gyShape->GetDim(0));
    gDaShape->SetDim(1, gyShape->GetDim(1));
    gDaShape->SetDim(2, gyShape->GetDim(2));
    gDaShape->SetDim(3, gyShape->GetDim(3));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Mamba2SsdChunkScanBwdOff : public OpDef {
public:
    explicit Mamba2SsdChunkScanBwdOff(const char *name) : OpDef(name)
    {
        this->Input("gy").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("states_start").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d_a_cumsum").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("d_states_start").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("d_c_head").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("g_d_a_cs_off").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        this->AICore().AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};

OP_ADD(Mamba2SsdChunkScanBwdOff);
} // namespace ops
