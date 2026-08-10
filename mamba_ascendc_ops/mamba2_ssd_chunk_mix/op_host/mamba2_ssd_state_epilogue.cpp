/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Fuses recurrent state passing, C@state, decay, D skip, SiLU gate and final
 * layout.  One MIX block owns one [B,H] stream and keeps state in Vector UB.
 */

#include "mamba2_ssd_state_epilogue_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kTile = 64;

size_t WorkspaceBytesPerCore(int64_t chunkSize, int64_t stateDim,
                             uint32_t headsPerTask,
                             size_t projectionElementBytes)
{
    return 2 * headsPerTask *
        (static_cast<size_t>(stateDim * kTile) * sizeof(uint16_t) +
         static_cast<size_t>(chunkSize * kTile) *
             projectionElementBytes);
}

bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) return false;
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) return false;
    }
    return true;
}

bool BuildArch35MatmulTiling(
    const platform_ascendc::PlatformAscendC &platform,
    int64_t outputCols, bool fp16Output,
    optiling::TCubeTiling &tiling)
{
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(platform);
    return cubeTiling.SetAType(matmul_tiling::TPosition::GM,
                               matmul_tiling::CubeFormat::ND,
                               matmul_tiling::DataType::DT_FLOAT16) == 0 &&
           cubeTiling.SetBType(matmul_tiling::TPosition::GM,
                               matmul_tiling::CubeFormat::ND,
                               matmul_tiling::DataType::DT_FLOAT16) == 0 &&
           cubeTiling.SetCType(
               matmul_tiling::TPosition::GM,
               matmul_tiling::CubeFormat::ND,
               fp16Output ? matmul_tiling::DataType::DT_FLOAT16
                          : matmul_tiling::DataType::DT_FLOAT) == 0 &&
           cubeTiling.SetBiasType(matmul_tiling::TPosition::GM,
                                  matmul_tiling::CubeFormat::ND,
                                  matmul_tiling::DataType::DT_FLOAT) == 0 &&
           cubeTiling.SetShape(kTile, outputCols, kTile) == 0 &&
           cubeTiling.SetOrgShape(kTile, outputCols, kTile) == 0 &&
           cubeTiling.SetDim(1) == 0 &&
           cubeTiling.SetSingleShape(kTile, outputCols, kTile) == 0 &&
           cubeTiling.SetFixSplit(kTile, outputCols, kTile) == 0 &&
           cubeTiling.EnableBias(false) == 0 &&
           cubeTiling.SetBufferSpace(-1, -1, -1) == 0 &&
           cubeTiling.GetTiling(tiling) != -1;
}

} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto stateShape = context->GetInputShape(0)->GetOriginShape();
    const auto daShape = context->GetInputShape(1)->GetOriginShape();
    const auto cShape = context->GetInputShape(2)->GetOriginShape();
    const auto yShape = context->GetInputShape(3)->GetOriginShape();
    const auto xShape = context->GetInputShape(4)->GetOriginShape();
    const auto dShape = context->GetInputShape(5)->GetOriginShape();
    const auto zShape = context->GetInputShape(6)->GetOriginShape();
    const auto initialShape = context->GetInputShape(7)->GetOriginShape();
    const bool inputGrouped = stateShape.GetDimNum() == 6;
    if ((!inputGrouped && stateShape.GetDimNum() != 5) ||
        daShape.GetDimNum() != 4 ||
        cShape.GetDimNum() != 5 || yShape.GetDimNum() != 5 ||
        xShape.GetDimNum() != 4 || dShape.GetDimNum() != 2 ||
        zShape.GetDimNum() != 4 || !SameShape(xShape, zShape)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = stateShape.GetDim(0);
    const int64_t chunks = inputGrouped ? stateShape.GetDim(1)
                                        : stateShape.GetDim(2);
    const int64_t groups = inputGrouped ? stateShape.GetDim(2)
                                        : cShape.GetDim(2);
    const int64_t chunkSize = cShape.GetDim(3);
    const bool stateNpLayout = inputGrouped ||
        chunkSize == 2 * kTile || cShape.GetDim(4) == kTile;
    const int64_t stateDim = inputGrouped
        ? stateShape.GetDim(3)
        : stateShape.GetDim(stateNpLayout ? 3 : 4);
    const int64_t headDim = inputGrouped
        ? stateShape.GetDim(5)
        : stateShape.GetDim(stateNpLayout ? 4 : 3);
    const int64_t headsPerGroup64 = inputGrouped
        ? stateShape.GetDim(4) : 0;
    const int64_t heads = inputGrouped
        ? groups * headsPerGroup64 : stateShape.GetDim(1);
    const int64_t seqlen = xShape.GetDim(1);
    const bool hasInitial = initialShape.GetDimNum() == 4;
    const bool initialCompatible = !hasInitial ||
        (initialShape.GetDim(0) == batch && initialShape.GetDim(1) == heads &&
         initialShape.GetDim(2) == (stateNpLayout ? stateDim : headDim) &&
         initialShape.GetDim(3) == (stateNpLayout ? headDim : stateDim));
    const bool stateShapeCompatible = inputGrouped
        ? (chunkSize == kTile && stateDim == kTile &&
           headsPerGroup64 == 4 &&
           stateShape.GetDim(1) == chunks &&
           stateShape.GetDim(2) == groups &&
           stateShape.GetDim(3) == stateDim &&
           stateShape.GetDim(5) == headDim)
        : (stateShape.GetDim(3) ==
               (stateNpLayout ? stateDim : headDim) &&
           stateShape.GetDim(4) ==
               (stateNpLayout ? headDim : stateDim));
    const bool compatible =
        batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
        heads % groups == 0 && headDim == kTile &&
        ((chunkSize == kTile &&
          (stateDim == kTile || stateDim == 2 * kTile)) ||
         (chunkSize == 2 * kTile && stateDim == 2 * kTile)) &&
        stateShapeCompatible &&
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
    const bool isAscend950 = mamba2_compat::IsAscend950(platform);
    // 950PR has one Vector core per Cube.  A grouped task lets that AIV own
    // four recurrent head states while its paired AIC performs one N=256
    // projection.  Legacy input remains a single-head 1:1 task.
    if (aic == 0 || aiv < aic) return ge::GRAPH_FAILED;
    const bool oneToOne = aiv < 2 * aic;
    // Arch35 exposes logical Vector capacity through GetCoreNumAiv(); that
    // count is not the MIX AIC:AIV pairing contract.  Select the grouped 1:1
    // kernel by SoC, exactly as the accepted standalone grouped projection.
    if (oneToOne && (chunkSize != kTile || stateDim != kTile)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t headsPerGroup = static_cast<uint32_t>(heads / groups);
    const uint64_t fourHeadTasks = static_cast<uint64_t>(batch) * groups *
                                   (headsPerGroup / 4);
    const uint64_t pairTasks = static_cast<uint64_t>(batch) * groups *
                               (headsPerGroup / 2);
    // On the 1AIC:2AIV architecture, aggregate as many same-group heads as
    // possible while retaining enough tasks to fill every Cube core.  H256
    // has four heads per group, so one 64x64 @ 64x256 projection replaces two
    // N=128 tasks and halves MIX rendezvous/scalar task scheduling.  Both AIV
    // subcores still own disjoint token/state slabs.  The 1:1 Arch35 path is
    // selected independently and is unchanged.
    const uint32_t headsPerTask = inputGrouped ? headsPerGroup :
        (!oneToOne && chunkSize == kTile &&
                 (stateDim == kTile || stateDim == 2 * kTile) &&
                 headsPerGroup % 4 == 0 && fourHeadTasks >= aic
             ? 4U
             : (!oneToOne && chunkSize == kTile &&
                        (stateDim == kTile || stateDim == 2 * kTile) &&
                        headsPerGroup % 2 == 0 && pairTasks >= aic
                    ? 2U : 1U));
    const uint64_t tasks64 = inputGrouped
        ? static_cast<uint64_t>(batch) * groups
        : (headsPerTask == 4 ? fourHeadTasks
           : (headsPerTask == 2 ? pairTasks
                                : static_cast<uint64_t>(batch) * heads));
    if (tasks64 == 0 || tasks64 > UINT32_MAX) return ge::GRAPH_FAILED;
    const uint32_t tasks = static_cast<uint32_t>(tasks64);
    const uint32_t usedCores = tasks < aic ? tasks : aic;
    const size_t projectionElementBytes = inputGrouped
        ? sizeof(uint16_t) : sizeof(float);
    const size_t workspaceBytes = WorkspaceBytesPerCore(
        chunkSize, stateDim, headsPerTask, projectionElementBytes);

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
    tiling.set_aivPerAic((inputGrouped && isAscend950) || oneToOne ? 1U : 2U);
    tiling.set_inputGrouped(inputGrouped ? 1U : 0U);
    if (((inputGrouped && isAscend950) || oneToOne) &&
        !BuildArch35MatmulTiling(platform, headsPerTask * kTile,
                                 inputGrouped && isAscend950,
                                 tiling.cubeTilingData)) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(usedCores);
    context->SetTilingKey(inputGrouped
        ? (isAscend950 ? 15 : 5)
        : (oneToOne ? 14 : (chunkSize == 2 * kTile ? 8 : 4)));
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
        finalShape == nullptr ||
        (stateShape->GetDimNum() != 5 && stateShape->GetDimNum() != 6) ||
        xShape->GetDimNum() != 4) return ge::GRAPH_FAILED;
    *outShape = *xShape;
    finalShape->SetDimNum(4);
    finalShape->SetDim(0, stateShape->GetDim(0));
    if (stateShape->GetDimNum() == 6) {
        finalShape->SetDim(
            1, stateShape->GetDim(2) * stateShape->GetDim(4));
        finalShape->SetDim(2, stateShape->GetDim(3));
        finalShape->SetDim(3, stateShape->GetDim(5));
    } else {
        finalShape->SetDim(1, stateShape->GetDim(1));
        finalShape->SetDim(2, stateShape->GetDim(3));
        finalShape->SetDim(3, stateShape->GetDim(4));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeTrain(gert::InferShapeContext *context)
{
    if (InferShape(context) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    const auto *xShape = context->GetInputShape(4);
    auto *preGateShape = context->GetOutputShape(2);
    if (xShape == nullptr || preGateShape == nullptr) return ge::GRAPH_FAILED;
    *preGateShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTrain(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeTrainStates(gert::InferShapeContext *context)
{
    if (InferShapeTrain(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto *stateShape = context->GetInputShape(0);
    auto *statesStartShape = context->GetOutputShape(3);
    if (stateShape == nullptr || statesStartShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // The device kernel writes the same Cube-native layout as chunk_states.
    // The PyTorch wrapper converts it to public [B,H,K,P,N] once in forward.
    *statesStartShape = *stateShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTrainStates(
    gert::InferDataTypeContext *context)
{
    if (InferDataTypeTrain(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(3, ge::DT_FLOAT16);
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
        this->AICore().AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};
OP_ADD(Mamba2SsdStateEpilogue);

class Mamba2SsdStateEpilogueTrain : public OpDef {
public:
    explicit Mamba2SsdStateEpilogueTrain(const char *name) : OpDef(name)
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
        this->Output("pre_gate").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeTrain)
            .SetInferDataType(ge::InferDataTypeTrain);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        this->AICore().AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};
OP_ADD(Mamba2SsdStateEpilogueTrain);

class Mamba2SsdStateEpilogueTrainStates : public OpDef {
public:
    explicit Mamba2SsdStateEpilogueTrainStates(const char *name) : OpDef(name)
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
        this->Output("pre_gate").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("states_start").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShapeTrainStates)
            .SetInferDataType(ge::InferDataTypeTrainStates);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        this->AICore().AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};
OP_ADD(Mamba2SsdStateEpilogueTrainStates);
} // namespace ops
