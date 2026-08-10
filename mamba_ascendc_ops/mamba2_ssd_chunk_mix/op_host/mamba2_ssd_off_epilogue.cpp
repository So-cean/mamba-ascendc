/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Fuses the inter-chunk C@state projection with decay, diagonal output,
 * D skip, SiLU gate and the final [B,L,H,P] layout write.
 */

#include "mamba2_ssd_off_epilogue_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kTile = 64;

size_t WorkspaceBytesPerCore(int64_t stateDim)
{
    // Two slots allow Vector to prepare/consume head n+1 while Cube computes
    // head n.  Each slot owns state^T[N,P] fp16 and y_off[T,P] fp32.
    return 2 * (static_cast<size_t>(stateDim * kTile) * sizeof(uint16_t) +
                static_cast<size_t>(kTile * kTile) * sizeof(float));
}

bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto cShape = context->GetInputTensor(0)->GetOriginShape();
    const auto stateShape = context->GetInputTensor(1)->GetOriginShape();
    const auto daShape = context->GetInputTensor(2)->GetOriginShape();
    const auto yShape = context->GetInputTensor(3)->GetOriginShape();
    const auto xShape = context->GetInputTensor(4)->GetOriginShape();
    const auto dShape = context->GetInputTensor(5)->GetOriginShape();
    const auto zShape = context->GetInputTensor(6)->GetOriginShape();
    if (cShape.GetDimNum() != 5 || stateShape.GetDimNum() != 5 ||
        daShape.GetDimNum() != 4 || yShape.GetDimNum() != 5 ||
        xShape.GetDimNum() != 4 || dShape.GetDimNum() != 2 ||
        zShape.GetDimNum() != 4 || !SameShape(xShape, zShape)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = stateShape.GetDim(0);
    const int64_t heads = stateShape.GetDim(1);
    const int64_t chunks = stateShape.GetDim(2);
    const int64_t headDim = stateShape.GetDim(3);
    const int64_t stateDim = stateShape.GetDim(4);
    const int64_t groups = cShape.GetDim(2);
    const int64_t chunkSize = cShape.GetDim(3);
    const int64_t seqlen = xShape.GetDim(1);
    const bool compatible =
        batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
        heads % groups == 0 && chunkSize == kTile && headDim == kTile &&
        (stateDim == kTile || stateDim == 2 * kTile) &&
        seqlen == chunks * chunkSize &&
        cShape.GetDim(0) == batch && cShape.GetDim(1) == chunks &&
        cShape.GetDim(4) == stateDim &&
        daShape.GetDim(0) == batch && daShape.GetDim(1) == heads &&
        daShape.GetDim(2) == chunks && daShape.GetDim(3) == chunkSize &&
        yShape.GetDim(0) == batch && yShape.GetDim(1) == heads &&
        yShape.GetDim(2) == chunks && yShape.GetDim(3) == chunkSize &&
        yShape.GetDim(4) == headDim &&
        xShape.GetDim(0) == batch && xShape.GetDim(2) == heads &&
        xShape.GetDim(3) == headDim && dShape.GetDim(0) == heads &&
        dShape.GetDim(1) == headDim;
    if (!compatible) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    // 910B exposes two Vector cores per Cube, while 950PR exposes one.  The
    // device binary selects the matching MIX task ratio and serializes the
    // two logical 32-row slabs when only one Vector core is present.
    if (aic == 0 || aiv < aic) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t aivPerAic = aiv >= 2 * aic ? 2U : 1U;
    const uint64_t headTasks64 =
        static_cast<uint64_t>(batch) * heads * chunks;
    const uint64_t groupTasks64 =
        static_cast<uint64_t>(batch) * groups * chunks;
    // Group mode preserves enough blocks to fill all AICs, then loops over
    // heads sharing C inside a block so the two-slot pipeline can overlap
    // Vector state/epilogue work with Cube matmul.
    const bool groupMode = heads / groups > 1 && groupTasks64 >= aic;
    const uint64_t tasks64 = groupMode ? groupTasks64 : headTasks64;
    if (tasks64 == 0 || tasks64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tasks = static_cast<uint32_t>(tasks64);
    const uint32_t usedCores = tasks < aic ? tasks : aic;
    const size_t workspaceBytes = WorkspaceBytesPerCore(stateDim);

    Mamba2SsdOffEpilogueTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_heads(static_cast<uint32_t>(heads));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_chunkSize(static_cast<uint32_t>(chunkSize));
    tiling.set_headDim(static_cast<uint32_t>(headDim));
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_stateDim(static_cast<uint32_t>(stateDim));
    tiling.set_taskCount(tasks);
    tiling.set_usedCoreNum(usedCores);
    tiling.set_workspaceBytesPerCore(static_cast<uint32_t>(workspaceBytes));
    tiling.set_headsPerGroup(static_cast<uint32_t>(heads / groups));
    tiling.set_taskMode(groupMode ? 1U : 0U);
    tiling.set_aivPerAic(aivPerAic);
    context->SetBlockDim(usedCores);
    context->SetTilingKey(aivPerAic == 1 ? 13 : 3);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
                        static_cast<size_t>(usedCores) * workspaceBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *xShape = context->GetInputShape(4);
    auto *outShape = context->GetOutputShape(0);
    if (xShape == nullptr || outShape == nullptr || xShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Mamba2SsdOffEpilogue : public OpDef {
public:
    explicit Mamba2SsdOffEpilogue(const char *name) : OpDef(name)
    {
        this->Input("c_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("states_start").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d_a_cumsum").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y_diag").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        this->AICore().AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};
OP_ADD(Mamba2SsdOffEpilogue);
} // namespace ops
