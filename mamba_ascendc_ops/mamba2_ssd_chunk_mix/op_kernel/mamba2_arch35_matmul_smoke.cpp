// Internal-test-only Ascend 950PR Matmul path validation.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {

constexpr uint32_t kTile = 64U;
constexpr uint32_t kHeads = 4U;
constexpr uint32_t kMatrixElements = kTile * kTile;
constexpr uint32_t kGroupElements = kTile * kHeads * kTile;
constexpr uint32_t kWide = kHeads * kTile;
constexpr MatmulConfig kMatmulConfig = GetBasicConfig(kTile, kTile, kTile);
constexpr MatmulConfig kBatchMatmulConfig = GetNormalConfig(
    false, false, false, BatchMode::BATCH_LESS_THAN_L1, true,
    IterateOrder::ORDER_M, ScheduleType::INNER_PRODUCT, true, false,
    BatchOutMode::MULTI_BATCH);

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BatchAType = MatmulType<TPosition::GM, CubeFormat::ND, half, false,
                             LayoutMode::NORMAL>;
using BatchBType = MatmulType<TPosition::GM, CubeFormat::ND, half, false,
                             LayoutMode::NORMAL>;
using BatchCType = MatmulType<TPosition::GM, CubeFormat::ND, float, false,
                             LayoutMode::NORMAL>;
using BatchBiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false,
                                LayoutMode::NORMAL>;

class Mamba2Arch35MatmulSmokeKernel {
public:
    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR xGroup, GM_ADDR yGroup)
    {
        weightGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(weight), kMatrixElements);
        xGroupGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(xGroup), kGroupElements);
        yGroupGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(yGroup), kGroupElements);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Validate a zero-copy per-head view of canonical [K,R,P].  A head
        // begins at r*P, while consecutive K rows are R*P elements apart.
        // The logical product is still [M,K]@[K,P], but the original N=R*P
        // shape carries the GM leading stride for both B and C.
        for (uint32_t head = 0; head < kHeads; ++head) {
            matmul_.SetOrgShape(kTile, kWide, kTile, kTile, kWide);
            matmul_.SetSingleShape(kTile, kTile, kTile);
            matmul_.SetTensorA(weightGm, false);
            matmul_.SetTensorB(xGroupGm[head * kTile], false);
            matmul_.IterateAll(yGroupGm[head * kTile], false);
            matmul_.End();
        }
    }

    __aicore__ inline void ProcessBatched()
    {
        if ASCEND_IS_AIV {
            return;
        }
        batchMatmul_.SetOrgShape(kTile, kTile, kTile, kTile, kTile);
        batchMatmul_.SetSingleShape(kTile, kTile, kTile);
        batchMatmul_.SetTensorA(weightGm, false);
        batchMatmul_.SetTensorB(xGroupGm, false);
        batchMatmul_.SetBatchNum(kHeads, kHeads);
        batchMatmul_.SetNBatchOutNum(kHeads);
        batchMatmul_.IterateBatch(
            yGroupGm, false, 0, false,
            kMatrixElements, kMatrixElements, kMatrixElements);
        batchMatmul_.End();
    }

    __aicore__ inline void InitMatmul(const TCubeTiling *cubeTiling)
    {
        if ASCEND_IS_AIC {
            // Follow CANN 9.0 upsample_linear1d_split.h: direct MatmulImpl
            // initialization is AIC-only, with no KFC registration or pipe.
            matmul_.Init(cubeTiling);
        }
    }

    __aicore__ inline void InitBatchMatmul(const TCubeTiling *cubeTiling)
    {
        if ASCEND_IS_AIC {
            batchMatmul_.Init(cubeTiling);
        }
    }

private:
    matmul::MatmulImpl<AType, BType, CType, BiasType, kMatmulConfig> matmul_;
    matmul::MatmulImpl<BatchAType, BatchBType, BatchCType, BatchBiasType,
                       kBatchMatmulConfig> batchMatmul_;
    GlobalTensor<half> weightGm;
    GlobalTensor<half> xGroupGm;
    GlobalTensor<float> yGroupGm;
};

} // namespace

extern "C" __global__ __aicore__ void mamba2_arch35_matmul_smoke(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    // Direct Arch35 MatmulImpl: both MIX sides initialize their local object,
    // while only AIC executes the actual Cube calculation.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA(tilingData, tiling);
    // Arch35 direct MatmulImpl kernels in this project instantiate TPipe at
    // the kernel entry even when the Cube path does not allocate queues from
    // it.  Keep the smoke test identical to that proven launch contract: the
    // pipe establishes the per-task local-memory runtime state for both
    // members of the MIX pair.
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        Mamba2Arch35MatmulSmokeKernel op;
        op.Init(a, b, c);
        op.InitMatmul(&tilingData.cubeTilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        Mamba2Arch35MatmulSmokeKernel op;
        op.Init(a, b, c);
        op.InitBatchMatmul(&tilingData.cubeTilingData);
        op.ProcessBatched();
    }
}
