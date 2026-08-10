/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Cube computes C[T,N] @ state[N,P].  The paired Vector cores cast and
 * transpose state, apply the decay/skip/SiLU epilogue, and write directly in
 * public [B,L,H,P] order.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/pad/broadcast.h"
#include "mamba2_dynamic_matmul.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 64;
constexpr uint32_t kTileElements = kTile * kTile;
constexpr uint32_t kLogicalSlabCount = 2;
constexpr uint32_t kRowsPerSlab = kTile / kLogicalSlabCount;
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements = kTransposeTile * kTransposeTile;
constexpr uint16_t kStateReady0 = 0xB;
constexpr uint16_t kStateReady1 = 0xC;
constexpr uint16_t kCubeReady0 = 0xD;
constexpr uint16_t kCubeReady1 = 0xE;

class VectorOffEpilogue {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(stateFloatIn_, 1,
                         kTransposeElements * sizeof(float));
        pipe->InitBuffer(stateHalfOut_, 1,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(transposeOut_, 1,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(dAIn_, 1, kRowsPerSlab * sizeof(float));
        pipe->InitBuffer(dIn_, 1, kTile * sizeof(float));
        pipe->InitBuffer(yOffIn_, 1,
                         kRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(yDiagIn_, 1,
                         kRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(xIn_, 1,
                         kRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(zIn_, 1,
                         kRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(out_, 1,
                         kRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(matrix0_,
                         kRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(matrix1_,
                         kRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(broadcastTmp_,
                         2 * kRowsPerSlab * kTile * sizeof(uint8_t));
    }

    __aicore__ inline void PrepareState(
        const GlobalTensor<float> &state,
        GlobalTensor<half> &stateT,
        uint32_t stateDim, uint32_t rowBegin)
    {
        const uint32_t rowEnd = rowBegin + kRowsPerSlab;
        for (uint32_t row = rowBegin; row < rowEnd;
             row += kTransposeTile) {
            for (uint32_t col = 0; col < stateDim;
                 col += kTransposeTile) {
                FloatToHalfTransposeTile(
                    state[row * stateDim + col],
                    stateT[col * kTile + row], stateDim, kTile);
            }
        }
    }

    __aicore__ inline void Epilogue(
        const GlobalTensor<float> &yOff,
        const GlobalTensor<float> &yDiag,
        const GlobalTensor<float> &dA,
        const GlobalTensor<float> &x,
        const GlobalTensor<float> &d,
        const GlobalTensor<float> &z,
        GlobalTensor<float> out,
        uint32_t heads, uint32_t rowBegin)
    {
        constexpr uint32_t blockElements = kRowsPerSlab * kTile;
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        auto matrix0 = matrix0_.Get<float>();
        auto matrix1 = matrix1_.Get<float>();

        auto daLocal = dAIn_.AllocTensor<float>();
        DataCopy(daLocal, dA[rowBegin], kRowsPerSlab);
        dAIn_.EnQue(daLocal);
        daLocal = dAIn_.DeQue<float>();
        const uint32_t daSrcShape[2] = {kRowsPerSlab, 1U};
        const uint32_t matrixShape[2] = {kRowsPerSlab, kTile};
        Broadcast<float, 2, 1>(matrix0, daLocal, matrixShape,
                               daSrcShape, broadcastTmp);
        PipeBarrier<PIPE_V>();
        Exp(matrix0, matrix0, blockElements);
        PipeBarrier<PIPE_V>();

        auto yOffLocal = yOffIn_.AllocTensor<float>();
        DataCopy(yOffLocal, yOff[rowBegin * kTile], blockElements);
        yOffIn_.EnQue(yOffLocal);
        yOffLocal = yOffIn_.DeQue<float>();
        auto yDiagLocal = yDiagIn_.AllocTensor<float>();
        DataCopy(yDiagLocal, yDiag[rowBegin * kTile], blockElements);
        yDiagIn_.EnQue(yDiagLocal);
        yDiagLocal = yDiagIn_.DeQue<float>();
        auto outLocal = out_.AllocTensor<float>();
        Mul(yOffLocal, yOffLocal, matrix0, blockElements);
        PipeBarrier<PIPE_V>();
        Add(outLocal, yDiagLocal, yOffLocal, blockElements);
        yOffIn_.FreeTensor(yOffLocal);
        yDiagIn_.FreeTensor(yDiagLocal);
        dAIn_.FreeTensor(daLocal);
        PipeBarrier<PIPE_V>();

        auto dLocal = dIn_.AllocTensor<float>();
        DataCopy(dLocal, d, kTile);
        dIn_.EnQue(dLocal);
        dLocal = dIn_.DeQue<float>();
        const uint32_t dSrcShape[2] = {1U, kTile};
        Broadcast<float, 2, 0>(matrix0, dLocal, matrixShape,
                               dSrcShape, broadcastTmp);
        dIn_.FreeTensor(dLocal);
        PipeBarrier<PIPE_V>();

        DataCopyParams rawLoad{
            static_cast<uint16_t>(kRowsPerSlab),
            static_cast<uint16_t>(kTile * sizeof(float) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((heads - 1) * kTile * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0};
        auto xLocal = xIn_.AllocTensor<float>();
        DataCopy(xLocal, x[rowBegin * heads * kTile], rawLoad);
        xIn_.EnQue(xLocal);
        xLocal = xIn_.DeQue<float>();
        Mul(xLocal, xLocal, matrix0, blockElements);
        PipeBarrier<PIPE_V>();
        Add(outLocal, outLocal, xLocal, blockElements);
        xIn_.FreeTensor(xLocal);
        PipeBarrier<PIPE_V>();

        auto zLocal = zIn_.AllocTensor<float>();
        DataCopy(zLocal, z[rowBegin * heads * kTile], rawLoad);
        zIn_.EnQue(zLocal);
        zLocal = zIn_.DeQue<float>();
        Muls(matrix1, zLocal, -1.0f, blockElements);
        PipeBarrier<PIPE_V>();
        Exp(matrix1, matrix1, blockElements);
        PipeBarrier<PIPE_V>();
        Adds(matrix1, matrix1, 1.0f, blockElements);
        PipeBarrier<PIPE_V>();
        Div(zLocal, zLocal, matrix1, blockElements);
        PipeBarrier<PIPE_V>();
        Mul(outLocal, outLocal, zLocal, blockElements);
        zIn_.FreeTensor(zLocal);

        out_.EnQue(outLocal);
        outLocal = out_.DeQue<float>();
        DataCopyParams rawStore{
            static_cast<uint16_t>(kRowsPerSlab),
            static_cast<uint16_t>(kTile * sizeof(float) / DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((heads - 1) * kTile * sizeof(float) /
                                  DEFAULT_C0_SIZE)};
        DataCopy(out[rowBegin * heads * kTile], outLocal, rawStore);
        out_.FreeTensor(outLocal);
    }

private:
    __aicore__ inline void FloatToHalfTransposeTile(
        const GlobalTensor<float> &src, GlobalTensor<half> dst,
        uint32_t srcStride, uint32_t dstStride)
    {
        auto floatLocal = stateFloatIn_.AllocTensor<float>();
        DataCopyParams load{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((srcStride - kTransposeTile) *
                                  sizeof(float) / DEFAULT_C0_SIZE),
            0};
        DataCopy(floatLocal, src, load);
        stateFloatIn_.EnQue(floatLocal);
        floatLocal = stateFloatIn_.DeQue<float>();
        auto halfLocal = stateHalfOut_.AllocTensor<half>();
        Cast(halfLocal, floatLocal, RoundMode::CAST_RINT,
             kTransposeElements);
        stateFloatIn_.FreeTensor(floatLocal);
        stateHalfOut_.EnQue(halfLocal);
        halfLocal = stateHalfOut_.DeQue<half>();
        auto transposed = transposeOut_.AllocTensor<half>();
        Transpose(transposed, halfLocal);
        stateHalfOut_.FreeTensor(halfLocal);
        transposeOut_.EnQue(transposed);
        transposed = transposeOut_.DeQue<half>();
        DataCopyParams store{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((dstStride - kTransposeTile) *
                                  sizeof(half) / DEFAULT_C0_SIZE)};
        DataCopy(dst, transposed, store);
        transposeOut_.FreeTensor(transposed);
    }

    TQue<TPosition::VECIN, 1> stateFloatIn_;
    TQue<TPosition::VECOUT, 1> stateHalfOut_;
    TQue<TPosition::VECOUT, 1> transposeOut_;
    TQue<TPosition::VECIN, 1> dAIn_;
    TQue<TPosition::VECIN, 1> dIn_;
    TQue<TPosition::VECIN, 1> yOffIn_;
    TQue<TPosition::VECIN, 1> yDiagIn_;
    TQue<TPosition::VECIN, 1> xIn_;
    TQue<TPosition::VECIN, 1> zIn_;
    TQue<TPosition::VECOUT, 1> out_;
    TBuf<TPosition::VECCALC> matrix0_;
    TBuf<TPosition::VECCALC> matrix1_;
    TBuf<TPosition::VECCALC> broadcastTmp_;
};

class KernelMamba2SsdOffEpilogue {
public:
    __aicore__ inline void Init(
        GM_ADDR cCube, GM_ADDR statesStart, GM_ADDR dACumsum,
        GM_ADDR yDiag, GM_ADDR x, GM_ADDR d, GM_ADDR z, GM_ADDR out,
        GM_ADDR workspace, const Mamba2SsdOffEpilogueTilingData &tiling,
        TPipe *pipe)
    {
        tiling_ = tiling;
        const uint64_t headTasks = static_cast<uint64_t>(tiling_.batch) *
                                   tiling_.heads * tiling_.chunks;
        const uint64_t groupMatrices = static_cast<uint64_t>(tiling_.batch) *
                                       tiling_.chunks * tiling_.groups;
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube),
                             groupMatrices * kTile * tiling_.stateDim);
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(statesStart),
                                 headTasks * kTile * tiling_.stateDim);
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum),
                              headTasks * kTile);
        yDiagGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yDiag),
                                 headTasks * kTileElements);
        const uint64_t rawElements = static_cast<uint64_t>(tiling_.batch) *
                                     tiling_.chunks * kTile *
                                     tiling_.heads * kTile;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), rawElements);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z), rawElements);
        dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d),
                             static_cast<uint64_t>(tiling_.heads) * kTile);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out),
                               rawElements);

        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) * tiling_.workspaceBytesPerCore;
        stateTHalfElements_ = tiling_.stateDim * kTile;
        const uint32_t twoStateBytes =
            2 * stateTHalfElements_ * sizeof(half);
        workspaceHalf_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(coreWorkspace),
            2 * stateTHalfElements_);
        workspaceFloat_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(coreWorkspace + twoStateBytes),
            2 * kTileElements);
        if ASCEND_IS_AIC {
            matmul_.Init(*pipe);
        }
        if ASCEND_IS_AIV {
            vector_.Init(pipe);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t task = coreIdx_; task < tiling_.taskCount;
             task += tiling_.usedCoreNum) {
            uint32_t batch = 0;
            uint32_t chunk = 0;
            uint32_t group = 0;
            uint32_t headBegin = 0;
            uint32_t headCount = 0;
            DecodeTask(task, batch, chunk, group, headBegin, headCount);
            const uint32_t cIndex =
                (batch * tiling_.chunks + chunk) * tiling_.groups + group;
            const uint64_t cOffset = static_cast<uint64_t>(cIndex) *
                                     kTile * tiling_.stateDim;

            if ASCEND_IS_AIV {
                const uint32_t preload = headCount < 2 ? headCount : 2;
                for (uint32_t headOffset = 0; headOffset < preload;
                     ++headOffset) {
                    PrepareHead(batch, chunk, headBegin + headOffset,
                                headOffset);
                }
                for (uint32_t headOffset = 0; headOffset < headCount;
                     ++headOffset) {
                    const uint32_t slot = headOffset & 1U;
                    const uint32_t head = headBegin + headOffset;
                    WaitCubeReady(slot);
                    RunEpilogue(batch, chunk, head, slot);
                    const uint32_t nextOffset = headOffset + 2;
                    if (nextOffset < headCount) {
                        PrepareHead(batch, chunk, headBegin + nextOffset,
                                    slot);
                    }
                }
            }
            if ASCEND_IS_AIC {
                for (uint32_t headOffset = 0; headOffset < headCount;
                     ++headOffset) {
                    const uint32_t slot = headOffset & 1U;
                    WaitStateReady(slot);
                    matmul_.ComputeBlock(
                        cGm_[cOffset],
                        workspaceHalf_[slot * stateTHalfElements_],
                        workspaceFloat_[slot * kTileElements],
                        tiling_.stateDim, tiling_.stateDim,
                        kTile, kTile);
                    SetCubeReady(slot);
                }
            }
        }
    }

private:
    __aicore__ inline void PrepareHead(uint32_t batch, uint32_t chunk,
                                       uint32_t head, uint32_t slot)
    {
        const uint64_t headTask =
            (static_cast<uint64_t>(batch) * tiling_.heads + head) *
                tiling_.chunks + chunk;
        auto stateT = workspaceHalf_[slot * stateTHalfElements_];
        for (uint32_t slab = GetSubBlockIdx(); slab < kLogicalSlabCount;
             slab += tiling_.aivPerAic) {
            vector_.PrepareState(
                stateGm_[headTask * kTile * tiling_.stateDim], stateT,
                tiling_.stateDim, slab * kRowsPerSlab);
        }
        SetStateReady(slot);
    }

    __aicore__ inline void RunEpilogue(uint32_t batch, uint32_t chunk,
                                       uint32_t head, uint32_t slot)
    {
        const uint64_t headTask =
            (static_cast<uint64_t>(batch) * tiling_.heads + head) *
                tiling_.chunks + chunk;
        const uint64_t rawOffset =
            (static_cast<uint64_t>(batch) * tiling_.chunks * kTile +
             static_cast<uint64_t>(chunk) * kTile) * tiling_.heads * kTile +
            static_cast<uint64_t>(head) * kTile;
        for (uint32_t slab = GetSubBlockIdx(); slab < kLogicalSlabCount;
             slab += tiling_.aivPerAic) {
            vector_.Epilogue(
                workspaceFloat_[slot * kTileElements],
                yDiagGm_[headTask * kTileElements],
                dAGm_[headTask * kTile], xGm_[rawOffset],
                dGm_[static_cast<uint64_t>(head) * kTile], zGm_[rawOffset],
                outGm_[rawOffset], tiling_.heads, slab * kRowsPerSlab);
        }
    }

    __aicore__ inline void SetStateReady(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(kStateReady0);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(kStateReady1);
        }
    }

    __aicore__ inline void WaitStateReady(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreWaitFlag<0x2>(kStateReady0);
        } else {
            CrossCoreWaitFlag<0x2>(kStateReady1);
        }
    }

    __aicore__ inline void SetCubeReady(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeReady0);
        } else {
            CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeReady1);
        }
    }

    __aicore__ inline void WaitCubeReady(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreWaitFlag<0x2>(kCubeReady0);
        } else {
            CrossCoreWaitFlag<0x2>(kCubeReady1);
        }
    }

    __aicore__ inline void DecodeTask(uint32_t task, uint32_t &batch,
                                      uint32_t &chunk, uint32_t &group,
                                      uint32_t &headBegin,
                                      uint32_t &headCount) const
    {
        chunk = task % tiling_.chunks;
        const uint32_t outer = task / tiling_.chunks;
        if (tiling_.taskMode == 1) {
            group = outer % tiling_.groups;
            batch = outer / tiling_.groups;
            headBegin = group * tiling_.headsPerGroup;
            headCount = tiling_.headsPerGroup;
        } else {
            const uint32_t head = outer % tiling_.heads;
            batch = outer / tiling_.heads;
            group = head / tiling_.headsPerGroup;
            headBegin = head;
            headCount = 1;
        }
    }

    Mamba2DynamicMatmul<half, float> matmul_;
    VectorOffEpilogue vector_;
    GlobalTensor<half> cGm_;
    GlobalTensor<float> stateGm_;
    GlobalTensor<float> dAGm_;
    GlobalTensor<float> yDiagGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> dGm_;
    GlobalTensor<float> zGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<half> workspaceHalf_;
    GlobalTensor<float> workspaceFloat_;
    Mamba2SsdOffEpilogueTilingData tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t stateTHalfElements_ = 0;
};
} // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_off_epilogue(
    GM_ADDR c_cube, GM_ADDR states_start, GM_ADDR d_a_cumsum,
    GM_ADDR y_diag, GM_ADDR x, GM_ADDR d, GM_ADDR z, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(13, KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelMamba2SsdOffEpilogue op;
    op.Init(c_cube, states_start, d_a_cumsum, y_diag, x, d, z, out,
            workspace, tilingData, &pipe);
    if (TILING_KEY_IS(3) || TILING_KEY_IS(13)) {
        op.Process();
    }
}
