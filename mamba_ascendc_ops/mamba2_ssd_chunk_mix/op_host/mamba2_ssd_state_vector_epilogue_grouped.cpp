// Copyright (c) 2026, mamba-ascendc authors.

#include "mamba2_ssd_state_vector_epilogue_grouped_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"

namespace {
constexpr int64_t kTile = 64;
constexpr int64_t kHeadsPerGroup = 4;

bool SameShape(const gert::Shape &a, const gert::Shape &b)
{
    if (a.GetDimNum() != b.GetDimNum()) return false;
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) return false;
    }
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *yShape = context->GetInputShape(0);
    const auto *daShape = context->GetInputShape(1);
    const auto *diagShape = context->GetInputShape(2);
    const auto *xShape = context->GetInputShape(3);
    const auto *dShape = context->GetInputShape(4);
    const auto *zShape = context->GetInputShape(5);
    if (yShape == nullptr || daShape == nullptr || diagShape == nullptr ||
        xShape == nullptr || dShape == nullptr || zShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto y = yShape->GetOriginShape();
    const auto da = daShape->GetOriginShape();
    const auto diag = diagShape->GetOriginShape();
    const auto x = xShape->GetOriginShape();
    const auto d = dShape->GetOriginShape();
    const auto z = zShape->GetOriginShape();
    const bool groupedDiag = diag.GetDimNum() == 6;
    if (y.GetDimNum() != 6 || da.GetDimNum() != 4 ||
        (!groupedDiag && diag.GetDimNum() != 5) || x.GetDimNum() != 4 ||
        d.GetDimNum() != 2 || z.GetDimNum() != 4 || !SameShape(x, z)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = y.GetDim(0);
    const int64_t chunks = y.GetDim(1);
    const int64_t groups = y.GetDim(2);
    const int64_t headsPerGroup = y.GetDim(4);
    const int64_t heads = groups * headsPerGroup;
    const bool diagCompatible = groupedDiag
        ? SameShape(diag, y)
        : diag.GetDim(0) == batch && diag.GetDim(1) == heads &&
              diag.GetDim(2) == chunks && diag.GetDim(3) == kTile &&
              diag.GetDim(4) == kTile;
    const bool compatible =
        batch > 0 && chunks > 0 && groups > 0 &&
        y.GetDim(3) == kTile && headsPerGroup == kHeadsPerGroup &&
        y.GetDim(5) == kTile && da.GetDim(0) == batch &&
        da.GetDim(1) == heads && da.GetDim(2) == chunks &&
        da.GetDim(3) == kTile && diagCompatible &&
        x.GetDim(0) == batch && x.GetDim(1) == chunks * kTile &&
        x.GetDim(2) == heads && x.GetDim(3) == kTile &&
        d.GetDim(0) == heads && d.GetDim(1) == kTile;
    if (!compatible) return ge::GRAPH_FAILED;

    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    if (!mamba2_compat::IsAscend950(platform)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t aiv = platform.GetCoreNumAiv();
    const uint64_t tasks64 = static_cast<uint64_t>(batch) * heads;
    if (aiv == 0 || tasks64 == 0 || tasks64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tasks = static_cast<uint32_t>(tasks64);
    const uint32_t used = tasks < aiv ? tasks : aiv;
    Mamba2SsdStateVectorEpilogueGroupedTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_heads(static_cast<uint32_t>(heads));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_headsPerGroup(static_cast<uint32_t>(headsPerGroup));
    tiling.set_taskCount(tasks);
    tiling.set_usedCoreNum(used);
    tiling.set_groupedDiag(groupedDiag ? 1U : 0U);
    context->SetBlockDim(used);
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *x = context->GetInputShape(3);
    auto *out = context->GetOutputShape(0);
    if (x == nullptr || out == nullptr) return ge::GRAPH_FAILED;
    *out = *x;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Mamba2SsdStateVectorEpilogueGrouped : public OpDef {
public:
    explicit Mamba2SsdStateVectorEpilogueGrouped(const char *name) : OpDef(name)
    {
        this->Input("y_grouped").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("d_a_cumsum").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y_diag").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        for (const char *name : {"x", "d", "z"}) {
            this->Input(name).ParamType(REQUIRED).DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc)
            .AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};
OP_ADD(Mamba2SsdStateVectorEpilogueGrouped);
}  // namespace ops
