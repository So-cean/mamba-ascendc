/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Fuses recurrent state passing, C@state, decay, D skip, SiLU gate and final
 * layout.  One MIX block owns one [B,H] stream and keeps state in Vector UB.
 */

#include "mamba2_ssd_state_epilogue_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kTile = 64;

size_t WorkspaceBytesPerCore(int64_t chunkSize, int64_t stateDim,
                             uint32_t headsPerTask)
{
    return 2 * headsPerTask *
        (static_cast<size_t>(stateDim * kTile) * sizeof(uint16_t) +
         static_cast<size_t>(chunkSize * kTile) * sizeof(float));
}

bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) return false;
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) return false;
    }
    return true;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto stateShape = context->GetInputTensor(0)->GetOriginShape();
    const auto daShape = context->GetInputTensor(1)->GetOriginShape();
    const auto cShape = context->GetInputTensor(2)->GetOriginShape();
    const auto yShape = context->GetInputTensor(3)->GetOriginShape();
    const auto xShape = context->GetInputTensor(4)->GetOriginShape();
    const auto dShape = context->GetInputTensor(5)->GetOriginShape();
    const auto zShape = context->GetInputTensor(6)->GetOriginShape();
    const auto initialShape = context->GetInputTensor(7)->GetOriginShape();
    if (stateShape.GetDimNum() != 5 || daShape.GetDimNum() != 4 ||
        cShape.GetDimNum() != 5 || yShape.GetDimNum() != 5 ||
        xShape.GetDimNum() != 4 || dShape.GetDimNum() != 2 ||
        zShape.GetDimNum() != 4 || !SameShape(xShape, zShape)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = stateShape.GetDim(0);
    const int64_t heads = stateShape.GetDim(1);
    const int64_t chunks = stateShape.GetDim(2);
    const int64_t groups = cShape.GetDim(2);
    const int64_t chunkSize = cShape.GetDim(3);
    const bool stateNpLayout =
        chunkSize == 2 * kTile || cShape.GetDim(4) == kTile;
    const int64_t headDim = stateShape.GetDim(stateNpLayout ? 4 : 3);
    const int64_t stateDim = stateShape.GetDim(stateNpLayout ? 3 : 4);
    const int64_t seqlen = xShape.GetDim(1);
    const bool hasInitial = initialShape.GetDimNum() == 4;
    const bool initialCompatible = !hasInitial ||
        (initialShape.GetDim(0) == batch && initialShape.GetDim(1) == heads &&
         initialShape.GetDim(2) == (stateNpLayout ? stateDim : headDim) &&
         initialShape.GetDim(3) == (stateNpLayout ? headDim : stateDim));
    const bool compatible =
        batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
        heads % groups == 0 && headDim == kTile &&
        ((chunkSize == kTile &&
          (stateDim == kTile || stateDim == 2 * kTile)) ||
         (chunkSize == 2 * kTile && stateDim == 2 * kTile)) &&
        stateShape.GetDim(3) == (stateNpLayout ? stateDim : headDim) &&
        stateShape.GetDim(4) == (stateNpLayout ? headDim : stateDim) &&
        seqlen == chunks * chunkSize && initialCompatible &&
        daShape.GetDim(0) == batch && daShape.GetDim(1) == heads &&
        daShape.GetDim(2) == chunks && daShape.GetDim(3) == chunkSize &&
        cShape.GetDim(0) == batch && cShape.GetDim(1) == chunks &&
        cShape.GetDim(4) == stateDim &&
        yShape.GetDim(0) == batch && yShape.GetDim(1) == heads &&
        yShape.GetDim(2) == chunks && yShape.GetDim(3) == chunkSize &&
        yShape.GetDim(4) == headDim &&
        xShape.GetDim(0) == batch && xShape.GetDim(2) == heads &&
        xShape.GetDim(3) == headDim && dShape.GetDim(0) == heads &&
        dShape.GetDim(1) == headDim;
    if (!compatible) return ge::GRAPH_FAILED;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    if (aic == 0 || aiv < 2 * aic) return ge::GRAPH_FAILED;
    const uint32_t headsPerGroup = static_cast<uint32_t>(heads / groups);
    const uint64_t pairTasks = static_cast<uint64_t>(batch) * groups *
                               (headsPerGroup / 2);
    // Pair only N=128 streams when there are still enough tasks to fill every
    // Cube core.  Small shapes retain one-head tasks and their full grid.
    const uint32_t headsPerTask =
        chunkSize == kTile && stateDim == 2 * kTile &&
                headsPerGroup % 2 == 0 && pairTasks >= aic
            ? 2U : 1U;
    const uint64_t tasks64 = headsPerTask == 2
        ? pairTasks : static_cast<uint64_t>(batch) * heads;
    if (tasks64 == 0 || tasks64 > UINT32_MAX) return ge::GRAPH_FAILED;
    const uint32_t tasks = static_cast<uint32_t>(tasks64);
    const uint32_t usedCores = tasks < aic ? tasks : aic;
    const size_t workspaceBytes = WorkspaceBytesPerCore(chunkSize, stateDim,
                                                         headsPerTask);

    Mamba2SsdStateEpilogueTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_heads(static_cast<uint32_t>(heads));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_stateDim(static_cast<uint32_t>(stateDim));
    tiling.set_taskCount(tasks);
    tiling.set_usedCoreNum(usedCores);
    tiling.set_workspaceBytesPerCore(static_cast<uint32_t>(workspaceBytes));
    tiling.set_headsPerGroup(headsPerGroup);
    tiling.set_headsPerTask(headsPerTask);
    tiling.set_hasInitial(hasInitial ? 1U : 0U);
    context->SetBlockDim(usedCores);
    context->SetTilingKey(chunkSize == 2 * kTile ? 8 : 4);
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
    const auto *stateShape = context->GetInputShape(0);
    const auto *xShape = context->GetInputShape(4);
    auto *outShape = context->GetOutputShape(0);
    auto *finalShape = context->GetOutputShape(1);
    if (stateShape == nullptr || xShape == nullptr || outShape == nullptr ||
        finalShape == nullptr || stateShape->GetDimNum() != 5 ||
        xShape->GetDimNum() != 4) return ge::GRAPH_FAILED;
    *outShape = *xShape;
    finalShape->SetDimNum(4);
    finalShape->SetDim(0, stateShape->GetDim(0));
    finalShape->SetDim(1, stateShape->GetDim(1));
    finalShape->SetDim(2, stateShape->GetDim(3));
    finalShape->SetDim(3, stateShape->GetDim(4));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Mamba2SsdStateEpilogue : public OpDef {
public:
    explicit Mamba2SsdStateEpilogue(const char *name) : OpDef(name)
    {
        this->Input("chunk_states").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d_a_cumsum").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_cube").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y_diag").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("initial_states").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("final_state").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Mamba2SsdStateEpilogue);
} // namespace ops
