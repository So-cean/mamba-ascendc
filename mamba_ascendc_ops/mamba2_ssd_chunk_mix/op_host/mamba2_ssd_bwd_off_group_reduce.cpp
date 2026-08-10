/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Host definition for group-owned dC/g_dA off-diagonal backward reduction.
 */

#include "mamba2_ssd_bwd_off_group_reduce_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kTile = 64;
constexpr uint32_t kTilingKey = 1;
constexpr size_t kWorkspaceBytesPerCore =
    static_cast<size_t>(2 * kTile * kTile) * sizeof(float);

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
    const auto qShape = context->GetInputTensor(0)->GetOriginShape();
    const auto stateShape = context->GetInputTensor(1)->GetOriginShape();
    const auto cShape = context->GetInputTensor(2)->GetOriginShape();
    if (qShape.GetDimNum() != 6 || stateShape.GetDimNum() != 6 ||
        cShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = qShape.GetDim(0);
    const int64_t groups = qShape.GetDim(1);
    const int64_t headsPerGroup = qShape.GetDim(2);
    const int64_t chunks = qShape.GetDim(3);
    const bool compatible =
        batch > 0 && groups > 0 && headsPerGroup > 0 && chunks > 0 &&
        qShape.GetDim(4) == kTile && qShape.GetDim(5) == kTile &&
        stateShape.GetDim(0) == batch && stateShape.GetDim(1) == groups &&
        stateShape.GetDim(2) == headsPerGroup &&
        stateShape.GetDim(3) == chunks &&
        stateShape.GetDim(4) == kTile && stateShape.GetDim(5) == kTile &&
        cShape.GetDim(0) == batch && cShape.GetDim(1) == chunks &&
        cShape.GetDim(2) == groups && cShape.GetDim(3) == kTile &&
        cShape.GetDim(4) == kTile;
    if (!compatible) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
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

    Mamba2SsdBwdOffGroupReduceTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_groups(static_cast<uint32_t>(groups));
    tiling.set_headsPerGroup(static_cast<uint32_t>(headsPerGroup));
    tiling.set_chunks(static_cast<uint32_t>(chunks));
    tiling.set_taskCount(tasks);
    tiling.set_usedCoreNum(usedCores);
    tiling.set_workspaceBytesPerCore(
        static_cast<uint32_t>(kWorkspaceBytesPerCore));
    if (!BuildMatmulTiling(platform, tiling.cubeTilingData)) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCores);
    context->SetTilingKey(kTilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
        static_cast<size_t>(usedCores) * kWorkspaceBytesPerCore;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *qShape = context->GetInputShape(0);
    auto *dCShape = context->GetOutputShape(0);
    auto *gDaShape = context->GetOutputShape(1);
    if (qShape == nullptr || dCShape == nullptr || gDaShape == nullptr ||
        qShape->GetDimNum() != 6) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = qShape->GetDim(0);
    const int64_t groups = qShape->GetDim(1);
    const int64_t heads = groups * qShape->GetDim(2);
    const int64_t chunks = qShape->GetDim(3);
    dCShape->SetDimNum(5);
    dCShape->SetDim(0, batch);
    dCShape->SetDim(1, chunks);
    dCShape->SetDim(2, kTile);
    dCShape->SetDim(3, groups);
    dCShape->SetDim(4, kTile);
    gDaShape->SetDimNum(4);
    gDaShape->SetDim(0, batch);
    gDaShape->SetDim(1, heads);
    gDaShape->SetDim(2, chunks);
    gDaShape->SetDim(3, kTile);
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
class Mamba2SsdBwdOffGroupReduce : public OpDef {
public:
    explicit Mamba2SsdBwdOffGroupReduce(const char *name) : OpDef(name)
    {
        this->Input("q_group").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("state_group").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("c_cube").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("d_c_group").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("g_d_a").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        this->AICore().AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};

OP_ADD(Mamba2SsdBwdOffGroupReduce);
} // namespace ops
