/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * First production gate for the chunk-local Mamba-2 MIX kernel.  This key is
 * intentionally restricted to 64x64x64 matrices until the three-stage Cube /
 * Vector pipeline has passed device precision and sanitizer gates.
 */

#include "mamba2_ssd_chunk_mix_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr int64_t kTile = 64;
// Per MIX group: two weighted-X slots and two W slots in fp16, followed by CB
// in fp32.  Key 3 additionally needs a legacy B[T,N] workspace because its
// state output remains [P,N]; Keys 2/7 consume preprocessed B^T directly.
size_t WorkspaceBytesPerCore(int64_t chunkSize, int64_t headDim,
                             int64_t stateDim)
{
    const int64_t legacyBElements =
        chunkSize == kTile && stateDim == 2 * kTile
            ? chunkSize * stateDim : 0;
    return static_cast<size_t>(legacyBElements +
                               2 * headDim * chunkSize +
                               2 * chunkSize * chunkSize) *
               sizeof(uint16_t) +
           static_cast<size_t>(chunkSize * chunkSize) * sizeof(float);
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
    const auto xShape = context->GetInputTensor(0)->GetOriginShape();
    const auto daShape = context->GetInputTensor(1)->GetOriginShape();
    const auto bShape = context->GetInputTensor(2)->GetOriginShape();
    const auto cShape = context->GetInputTensor(3)->GetOriginShape();
    if (xShape.GetDimNum() != 5 || daShape.GetDimNum() != 4 ||
        bShape.GetDimNum() != 5 || cShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = xShape.GetDim(0);
    const int64_t heads = xShape.GetDim(1);
    const int64_t chunks = xShape.GetDim(2);
    const int64_t chunkSize = xShape.GetDim(3);
    const int64_t headDim = xShape.GetDim(4);
    const int64_t groups = bShape.GetDim(2);
    const int64_t stateDim = cShape.GetDim(4);

    const bool compatible = batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
        heads % groups == 0 && headDim == kTile &&
        ((chunkSize == kTile &&
          (stateDim == kTile || stateDim == 2 * kTile)) ||
         (chunkSize == 2 * kTile && stateDim == 2 * kTile)) &&
        daShape.GetDim(0) == batch && daShape.GetDim(1) == heads &&
        daShape.GetDim(2) == chunks && daShape.GetDim(3) == chunkSize &&
        bShape.GetDim(0) == batch && bShape.GetDim(1) == chunks &&
        bShape.GetDim(3) == stateDim && bShape.GetDim(4) == chunkSize &&
        cShape.GetDim(0) == batch && cShape.GetDim(1) == chunks &&
        cShape.GetDim(2) == groups && cShape.GetDim(3) == chunkSize &&
        cShape.GetDim(4) == stateDim;
    if (!compatible) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    if (aic == 0 || aiv < 2 * aic) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t headTaskCount64 =
        static_cast<uint64_t>(batch) * heads * chunks;
    const uint64_t groupTaskCount64 =
        static_cast<uint64_t>(batch) * groups * chunks;
    if (headTaskCount64 == 0 || headTaskCount64 > UINT32_MAX ||
        groupTaskCount64 == 0 || groupTaskCount64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t headTaskCount = static_cast<uint32_t>(headTaskCount64);
    const uint32_t groupTaskCount = static_cast<uint32_t>(groupTaskCount64);
    // Once (B,G,K) already fills all Cube cores, keep B^T and CB resident in
    // the per-core workspace and process every head in that group.  For small
    // inputs retain head tasks so H/G contributes launch parallelism.
    const uint32_t taskMode =
        groupTaskCount >= aic && heads / groups > 1 ? 1U : 0U;
    const uint32_t taskCount = taskMode == 1 ? groupTaskCount : headTaskCount;
    const uint32_t usedCoreNum = taskCount < aic ? taskCount : aic;
    const size_t workspaceBytesPerCore =
        WorkspaceBytesPerCore(chunkSize, headDim, stateDim);

    Mamba2SsdChunkMixTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_heads(static_cast<uint32_t>(heads));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_chunkSize(static_cast<uint32_t>(chunkSize));
    tiling.set_headDim(static_cast<uint32_t>(headDim));
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_stateDim(static_cast<uint32_t>(stateDim));
    tiling.set_taskCount(taskCount);
    tiling.set_headTaskCount(headTaskCount);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_workspaceBytesPerCore(
        static_cast<uint32_t>(workspaceBytesPerCore));
    tiling.set_taskMode(taskMode);
    tiling.set_headsPerGroup(
        static_cast<uint32_t>(heads / groups));
    // T=128 keeps the recurrent state in Cube-native [N,P] layout.  ChunkMix
    // can then reuse B^T for B^T@weighted-X and StateEpilogue no longer has
    // to transpose [P,N] back to [N,P] on every chunk.
    const uint32_t stateNpLayout =
        (chunkSize == kTile && stateDim == kTile) ||
        chunkSize == 2 * kTile ? 1U : 0U;
    tiling.set_stateNpLayout(stateNpLayout);


    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(chunkSize == 2 * kTile ? 7 :
                          (stateDim == kTile ? 2 : 3));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
                        static_cast<size_t>(usedCoreNum) * workspaceBytesPerCore;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *xShape = context->GetInputShape(0);
    const auto *cShape = context->GetInputShape(3);
    auto *yShape = context->GetOutputShape(0);
    auto *stateShape = context->GetOutputShape(1);
    if (xShape == nullptr || cShape == nullptr || yShape == nullptr ||
        stateShape == nullptr || xShape->GetDimNum() != 5 ||
        cShape->GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    stateShape->SetDimNum(5);
    stateShape->SetDim(0, xShape->GetDim(0));
    stateShape->SetDim(1, xShape->GetDim(1));
    stateShape->SetDim(2, xShape->GetDim(2));
    const bool stateNpLayout =
        (xShape->GetDim(3) == kTile && cShape->GetDim(4) == kTile) ||
        xShape->GetDim(3) == 2 * kTile;
    stateShape->SetDim(3, stateNpLayout ? cShape->GetDim(4) :
                                         xShape->GetDim(4));
    stateShape->SetDim(4, stateNpLayout ? xShape->GetDim(4) :
                                         cShape->GetDim(4));
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

class Mamba2SsdChunkMix : public OpDef {
public:
    explicit Mamba2SsdChunkMix(const char *name) : OpDef(name)
    {
        this->Input("x_cube")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d_a_cumsum")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b_cube")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_cube")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y_diag")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("chunk_states")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Mamba2SsdChunkMix);
} // namespace ops
