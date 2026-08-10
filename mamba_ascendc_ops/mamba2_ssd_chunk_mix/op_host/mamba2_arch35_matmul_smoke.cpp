// Internal-test-only Ascend 950PR Matmul path validation.

#include "mamba2_arch35_matmul_smoke_tiling.h"
#include "mamba2_cann_compat.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdio>

namespace {
constexpr int32_t kM = 64;
constexpr int32_t kHeadDim = 64;
constexpr int32_t kK = 64;
constexpr int32_t kHeads = 4;
constexpr int32_t kN = kHeads * kHeadDim;
} // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *aStorage = context->GetInputShape(0);
    const auto *bStorage = context->GetInputShape(1);
    if (aStorage == nullptr || bStorage == nullptr ||
        context->GetInputDesc(0) == nullptr ||
        context->GetInputDesc(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto aShape = aStorage->GetStorageShape();
    const auto bShape = bStorage->GetStorageShape();
    const bool batched =
        aShape.GetDimNum() == 3 && bShape.GetDimNum() == 3 &&
        aShape.GetDim(0) == kHeads && aShape.GetDim(1) == kM &&
        aShape.GetDim(2) == kK && bShape.GetDim(0) == kHeads &&
        bShape.GetDim(1) == kK && bShape.GetDim(2) == kHeadDim;
    const bool strided =
        aShape.GetDimNum() == 2 && bShape.GetDimNum() == 3 &&
        aShape.GetDim(0) == kM && aShape.GetDim(1) == kK &&
        bShape.GetDim(0) == kK && bShape.GetDim(1) == kHeads &&
        bShape.GetDim(2) == kHeadDim;
    if (!batched && !strided) {
        return ge::GRAPH_FAILED;
    }

    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    Mamba2Arch35MatmulSmokeTilingData tiling;
    if (batched) {
        matmul_tiling::BatchMatmulTiling cubeTiling(platform);
        if (cubeTiling.SetAType(matmul_tiling::TPosition::GM,
                                matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT16) != 0 ||
            cubeTiling.SetBType(matmul_tiling::TPosition::GM,
                                matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT16) != 0 ||
            cubeTiling.SetCType(matmul_tiling::TPosition::GM,
                                matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT) != 0 ||
            cubeTiling.SetBiasType(matmul_tiling::TPosition::GM,
                                   matmul_tiling::CubeFormat::ND,
                                   matmul_tiling::DataType::DT_FLOAT) != 0 ||
            cubeTiling.SetShape(kM, kHeadDim, kK) != 0 ||
            cubeTiling.SetOrgShape(kM, kHeadDim, kK) != 0 ||
            cubeTiling.SetBatchNum(kHeads) != 0 ||
            cubeTiling.EnableBias(false) != 0 ||
            cubeTiling.SetBufferSpace(-1, -1, -1) != 0 ||
            cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
            return ge::GRAPH_FAILED;
        }
    } else {
        matmul_tiling::MultiCoreMatmulTiling cubeTiling(platform);
        if (cubeTiling.SetAType(matmul_tiling::TPosition::GM,
                                matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT16) != 0 ||
            cubeTiling.SetBType(matmul_tiling::TPosition::GM,
                                matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT16) != 0 ||
            cubeTiling.SetCType(matmul_tiling::TPosition::GM,
                                matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT) != 0 ||
            cubeTiling.SetBiasType(matmul_tiling::TPosition::GM,
                                   matmul_tiling::CubeFormat::ND,
                                   matmul_tiling::DataType::DT_FLOAT) != 0 ||
            cubeTiling.SetShape(kM, kHeadDim, kK) != 0 ||
            // The physical B/C row is R*P wide, while one invocation
            // computes a P-wide head view at an r*P offset.
            cubeTiling.SetOrgShape(kM, kN, kK) != 0 ||
            cubeTiling.SetDim(1) != 0 ||
            cubeTiling.SetSingleShape(kM, kHeadDim, kK) != 0 ||
            cubeTiling.SetFixSplit(kM, kHeadDim, kK) != 0 ||
            cubeTiling.EnableBias(false) != 0 ||
            cubeTiling.SetBufferSpace(-1, -1, -1) != 0 ||
            cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
            return ge::GRAPH_FAILED;
        }
    }
    std::fprintf(
        stderr,
        "MAMBA2_ARCH35_TILING used=%d M=%d N=%d Ka=%d Kb=%d "
        "single=(%d,%d,%d) base=(%d,%d,%d) depth=(%d,%d) "
        "step=(%d,%d,%d,%d) db=(%d,%d,%d)\n",
        tiling.cubeTilingData.get_usedCoreNum(),
        tiling.cubeTilingData.get_M(),
        tiling.cubeTilingData.get_N(),
        tiling.cubeTilingData.get_Ka(),
        tiling.cubeTilingData.get_Kb(),
        tiling.cubeTilingData.get_singleCoreM(),
        tiling.cubeTilingData.get_singleCoreN(),
        tiling.cubeTilingData.get_singleCoreK(),
        tiling.cubeTilingData.get_baseM(),
        tiling.cubeTilingData.get_baseN(),
        tiling.cubeTilingData.get_baseK(),
        tiling.cubeTilingData.get_depthA1(),
        tiling.cubeTilingData.get_depthB1(),
        tiling.cubeTilingData.get_stepM(),
        tiling.cubeTilingData.get_stepN(),
        tiling.cubeTilingData.get_stepKa(),
        tiling.cubeTilingData.get_stepKb(),
        tiling.cubeTilingData.get_dbL0A(),
        tiling.cubeTilingData.get_dbL0B(),
        tiling.cubeTilingData.get_dbL0C());

    context->SetBlockDim(1);
    context->SetTilingKey(batched ? 2 : 1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *bShape = context->GetInputShape(1);
    auto *cShape = context->GetOutputShape(0);
    if (bShape == nullptr || cShape == nullptr || bShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    *cShape = *bShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Mamba2Arch35MatmulSmoke : public OpDef {
public:
    explicit Mamba2Arch35MatmulSmoke(const char *name) : OpDef(name)
    {
        this->Input("weight")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x_group")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y_group")
            .ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc)
            .AddConfig(MAMBA2_ASCEND950_CONFIG);
    }
};

OP_ADD(Mamba2Arch35MatmulSmoke);
} // namespace ops
