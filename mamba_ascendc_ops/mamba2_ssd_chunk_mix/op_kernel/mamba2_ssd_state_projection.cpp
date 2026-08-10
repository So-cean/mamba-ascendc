// Copyright (c) 2026, mamba-ascendc authors.
//
// First launch of the 950PR State path.  AIV converts one public
// [P,N] FP32 state to Cube-native [N,P] FP16 in per-core GM workspace.
// Its paired AIC writes C[T,N] @ state[N,P] to a dedicated FP32 y_off
// tensor.  No AIV consumes y_off in this launch.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 64;
constexpr uint32_t kTileElements = kTile * kTile;
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements = kTransposeTile * kTransposeTile;
constexpr uint16_t kStateReady = 0x8;
constexpr uint16_t kCubeDone = 0x9;
constexpr MatmulConfig kProjectionConfig = GetBasicConfig(kTile, kTile, kTile);

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

class StateTransposeVector {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(floatIn_, 1,
                         kTransposeElements * sizeof(float));
        pipe->InitBuffer(halfOut_, 1,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(transposeOut_, 1,
                         kTransposeElements * sizeof(half));
    }

    __aicore__ inline void Run(const GlobalTensor<float> &state,
                               GlobalTensor<half> stateT)
    {
        for (uint32_t row = 0; row < kTile; row += kTransposeTile) {
            for (uint32_t col = 0; col < kTile; col += kTransposeTile) {
                auto fp32 = floatIn_.AllocTensor<float>();
                DataCopyParams load{
                    static_cast<uint16_t>(kTransposeTile),
                    static_cast<uint16_t>(kTransposeTile * sizeof(float) /
                                          DEFAULT_C0_SIZE),
                    static_cast<uint16_t>((kTile - kTransposeTile) *
                                          sizeof(float) /
                                          DEFAULT_C0_SIZE),
                    0};
                DataCopy(fp32, state[row * kTile + col], load);
                floatIn_.EnQue(fp32);
                fp32 = floatIn_.DeQue<float>();

                auto fp16 = halfOut_.AllocTensor<half>();
                Cast(fp16, fp32, RoundMode::CAST_RINT, kTransposeElements);
                floatIn_.FreeTensor(fp32);
                halfOut_.EnQue(fp16);
                fp16 = halfOut_.DeQue<half>();

                auto transposed = transposeOut_.AllocTensor<half>();
                Transpose(transposed, fp16);
                halfOut_.FreeTensor(fp16);
                transposeOut_.EnQue(transposed);
                transposed = transposeOut_.DeQue<half>();
                DataCopyExtParams store{
                    static_cast<uint16_t>(kTransposeTile),
                    static_cast<uint32_t>(kTransposeTile * sizeof(half)),
                    0,
                    static_cast<uint32_t>((kTile - kTransposeTile) *
                                          sizeof(half)),
                    0};
                DataCopyPad(stateT[col * kTile + row], transposed, store);
                transposeOut_.FreeTensor(transposed);
            }
        }
    }

private:
    TQue<TPosition::VECIN, 1> floatIn_;
    TQue<TPosition::VECOUT, 1> halfOut_;
    TQue<TPosition::VECOUT, 1> transposeOut_;
};

class KernelMamba2SsdStateProjection {
public:
    __aicore__ inline void Init(
        GM_ADDR statesStart, GM_ADDR cCube, GM_ADDR yOff,
        GM_ADDR workspace,
        const Mamba2SsdStateProjectionTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        statesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(statesStart),
            static_cast<uint64_t>(tiling_.taskCount) * kTileElements);
        const uint64_t groupTasks =
            static_cast<uint64_t>(tiling_.batch) * tiling_.chunks *
            tiling_.groups;
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube),
                             groupTasks * kTileElements);
        yOffGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(yOff),
            static_cast<uint64_t>(tiling_.taskCount) * kTileElements);
        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) *
                tiling_.workspaceBytesPerCore;
        stateHalfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(coreWorkspace), kTileElements);
        if ASCEND_IS_AIV {
            vector_.Init(pipe);
        }
        if ASCEND_IS_AIC {
            matmul_.Init(&tiling_.cubeTilingData);
        }
    }

    __aicore__ inline void Process()
    {
        bool hasPrevious = false;
        for (uint32_t task = coreIdx_; task < tiling_.taskCount;
             task += tiling_.usedCoreNum) {
            const uint32_t chunk = task % tiling_.chunks;
            const uint32_t stream = task / tiling_.chunks;
            const uint32_t head = stream % tiling_.heads;
            const uint32_t batch = stream / tiling_.heads;
            const uint32_t group = head / tiling_.headsPerGroup;
            const uint32_t groupTask =
                (batch * tiling_.chunks + chunk) * tiling_.groups + group;

            if ASCEND_IS_AIV {
                if (hasPrevious) {
                    CrossCoreWaitFlag<0x2>(kCubeDone);
                }
                vector_.Run(statesGm_[static_cast<uint64_t>(task) *
                                      kTileElements], stateHalfGm_);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(kStateReady);
            }
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<0x2>(kStateReady);
                matmul_.SetOrgShape(kTile, kTile, kTile, kTile, kTile);
                matmul_.SetSingleShape(kTile, kTile, kTile);
                matmul_.SetTensorA(
                    cGm_[static_cast<uint64_t>(groupTask) * kTileElements],
                    false);
                matmul_.SetTensorB(stateHalfGm_, false);
                matmul_.IterateAll(
                    yOffGm_[static_cast<uint64_t>(task) * kTileElements],
                    false);
                matmul_.End();
                CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeDone);
            }
            hasPrevious = true;
        }
        if ASCEND_IS_AIV {
            if (hasPrevious) {
                CrossCoreWaitFlag<0x2>(kCubeDone);
            }
        }
    }

private:
    StateTransposeVector vector_;
    matmul::MatmulImpl<AType, BType, CType, BiasType,
                       kProjectionConfig> matmul_;
    GlobalTensor<float> statesGm_;
    GlobalTensor<half> cGm_;
    GlobalTensor<float> yOffGm_;
    GlobalTensor<half> stateHalfGm_;
    Mamba2SsdStateProjectionTilingData tiling_;
    uint32_t coreIdx_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_state_projection(
    GM_ADDR states_start, GM_ADDR c_cube, GM_ADDR y_off,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelMamba2SsdStateProjection op;
        op.Init(states_start, c_cube, y_off, workspace, tilingData, &pipe);
        op.Process();
    }
}
