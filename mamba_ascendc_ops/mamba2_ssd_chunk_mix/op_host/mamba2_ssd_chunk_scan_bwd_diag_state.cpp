/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Host definition for the fixed T=P=N=64 diagonal + chunk-state backward.
 */

#include "mamba2_ssd_chunk_scan_bwd_diag_state_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kTile = 64;
constexpr uint32_t kTilingKey = 1;

size_t WorkspaceBytesPerCore()
{
    constexpr size_t halfMatrices = 31;
    constexpr size_t floatMatrices = 9;
    return halfMatrices * kTile * kTile * sizeof(uint16_t) +
           floatMatrices * kTile * kTile * sizeof(float);
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto gy = context->GetInputTensor(0)->GetOriginShape();
    const auto x = context->GetInputTensor(1)->GetOriginShape();
    const auto cs = context->GetInputTensor(2)->GetOriginShape();
    const auto b = context->GetInputTensor(3)->GetOriginShape();
    const auto c = context->GetInputTensor(4)->GetOriginShape();
    const auto du = context->GetInputTensor(5)->GetOriginShape();
    if (gy.GetDimNum() != 5 || x.GetDimNum() != 5 ||
        cs.GetDimNum() != 4 || b.GetDimNum() != 5 ||
        c.GetDimNum() != 5 || du.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = gy.GetDim(0);
    const int64_t heads = gy.GetDim(1);
    const int64_t chunks = gy.GetDim(2);
    const int64_t groups = b.GetDim(2);
    const bool compatible =
        batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
        heads % groups == 0 && gy.GetDim(3) == kTile &&
        gy.GetDim(4) == kTile && x.GetDim(0) == batch &&
        x.GetDim(1) == heads && x.GetDim(2) == chunks &&
        x.GetDim(3) == kTile && x.GetDim(4) == kTile &&
        cs.GetDim(0) == batch && cs.GetDim(1) == heads &&
        cs.GetDim(2) == chunks && cs.GetDim(3) == kTile &&
        b.GetDim(0) == batch && b.GetDim(1) == chunks &&
        b.GetDim(3) == kTile && b.GetDim(4) == kTile &&
        c.GetDim(0) == batch && c.GetDim(1) == chunks &&
        c.GetDim(2) == groups && c.GetDim(3) == kTile &&
        c.GetDim(4) == kTile && du.GetDim(0) == batch &&
        du.GetDim(1) == heads && du.GetDim(2) == chunks &&
        du.GetDim(3) == kTile && du.GetDim(4) == kTile;
    if (!compatible) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    // MIX topology is architecture-specific.  910B schedules two AIV
    // subcores per AIC, while Ascend 950PR schedules one stronger AIV per
    // AIC.  The device binary selects its matching task ratio; host tiling
    // only needs at least one Vector and one Cube core.
    if (aic == 0 || aiv == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tasks64 =
        static_cast<uint64_t>(batch) * groups * chunks;
    if (tasks64 == 0 || tasks64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tasks = static_cast<uint32_t>(tasks64);
    const uint32_t usedCores = tasks < aic ? tasks : aic;
    const size_t workspaceBytes = WorkspaceBytesPerCore();

    Mamba2SsdChunkScanBwdDiagStateTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_heads(static_cast<uint32_t>(heads));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_taskCount(tasks);
    tiling.set_usedCoreNum(usedCores);
    tiling.set_workspaceBytesPerCore(
        static_cast<uint32_t>(workspaceBytes));
    tiling.set_headsPerGroup(static_cast<uint32_t>(heads / groups));

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
    const auto *gy = context->GetInputShape(0);
    const auto *cs = context->GetInputShape(2);
    const auto *c = context->GetInputShape(4);
    auto *dx = context->GetOutputShape(0);
    auto *db = context->GetOutputShape(1);
    auto *dc = context->GetOutputShape(2);
    auto *gcs = context->GetOutputShape(3);
    if (gy == nullptr || cs == nullptr || c == nullptr || dx == nullptr ||
        db == nullptr || dc == nullptr || gcs == nullptr ||
        gy->GetDimNum() != 5 || cs->GetDimNum() != 4 ||
        c->GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }
    *dx = *gy;
    *db = *c;
    *dc = *c;
    *gcs = *cs;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    for (size_t index = 0; index < 4; ++index) {
        context->SetOutputDataType(index, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Mamba2SsdChunkScanBwdDiagState : public OpDef {
public:
    explicit Mamba2SsdChunkScanBwdDiagState(const char *name) : OpDef(name)
    {
        this->Input("gy").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d_a_cumsum").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d_chunk_states").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("d_xdt").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("d_b_group").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("d_c_diag_group").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("g_d_a_cs_diag_state").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        this->AICore().AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};

OP_ADD(Mamba2SsdChunkScanBwdDiagState);
} // namespace ops
