// Copyright (c) 2026, mamba-ascendc authors.
//
// Group-contiguous projection.  One Cube task computes four
// heads at once: C[64,64] @ state[64,4,64] -> y[64,4,64].  The public
// head-major layout conversion deliberately remains outside this kernel;
// upstream and downstream grouped kernels consume this canonical form.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 64;
constexpr uint32_t kHeadsPerGroup = 4;
constexpr uint32_t kWide = kHeadsPerGroup * kTile;
constexpr uint32_t kMatrixElements = kTile * kTile;
constexpr uint32_t kGroupElements = kTile * kWide;
constexpr MatmulConfig kProjectionConfig =
    GetBasicConfig(kTile, kWide, kTile);

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

class KernelMamba2SsdStateProjectionGrouped {
public:
    __aicore__ inline void Init(
        GM_ADDR statesGrouped, GM_ADDR cCube, GM_ADDR yGrouped,
        const Mamba2SsdStateProjectionGroupedTilingData &tiling)
    {
        tiling_ = tiling;
        statesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(statesGrouped),
            static_cast<uint64_t>(tiling_.taskCount) * kGroupElements);
        cGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(cCube),
            static_cast<uint64_t>(tiling_.taskCount) * kMatrixElements);
        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(yGrouped),
            static_cast<uint64_t>(tiling_.taskCount) * kGroupElements);
        coreIdx_ = GetBlockIdx();
        if ASCEND_IS_AIC {
            matmul_.Init(&tiling_.cubeTilingData);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }
        for (uint32_t task = coreIdx_; task < tiling_.taskCount;
             task += tiling_.usedCoreNum) {
            matmul_.SetOrgShape(kTile, kWide, kTile, kTile, kWide);
            matmul_.SetSingleShape(kTile, kWide, kTile);
            matmul_.SetTensorA(
                cGm_[static_cast<uint64_t>(task) * kMatrixElements], false);
            matmul_.SetTensorB(
                statesGm_[static_cast<uint64_t>(task) * kGroupElements],
                false);
            matmul_.IterateAll(
                yGm_[static_cast<uint64_t>(task) * kGroupElements], false);
            matmul_.End();
        }
    }

private:
    matmul::MatmulImpl<AType, BType, CType, BiasType,
                       kProjectionConfig> matmul_;
    GlobalTensor<half> statesGm_;
    GlobalTensor<half> cGm_;
    GlobalTensor<half> yGm_;
    Mamba2SsdStateProjectionGroupedTilingData tiling_;
    uint32_t coreIdx_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_state_projection_grouped(
    GM_ADDR states_grouped, GM_ADDR c_cube, GM_ADDR y_grouped,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA(tilingData, tiling);
    // Required by the Arch35 direct MatmulImpl MIX launch contract even
    // though this Cube-only path does not allocate Vector queues.
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelMamba2SsdStateProjectionGrouped op;
        op.Init(states_grouped, c_cube, y_grouped, tilingData);
        op.Process();
    }
}
