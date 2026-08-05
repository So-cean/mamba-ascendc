/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Key 2: one AIC plus two AIVs execute one or more 64x64 Mamba-2 chunk
 * tasks.  Cube performs all three matrix multiplications.  Vector performs
 * the two fp16 transposes, exponential decay and causal masking.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/pad/broadcast.h"
#include "mamba2_dynamic_matmul.h"
#ifndef MAMBA2_CHUNK_MIX_COMPONENT_ONLY
#include "mamba2_ssd_chunk_mix128.h"
#endif

using namespace AscendC;

namespace {

constexpr uint32_t kTile = 64;
constexpr uint32_t kTileElements = kTile * kTile;
constexpr uint32_t kRowsPerAiv = kTile / 2;
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements = kTransposeTile * kTransposeTile;

constexpr uint16_t kVectorCbInputReady = 0x8;
constexpr uint16_t kCubeCbReady = 0x9;
constexpr uint16_t kHeadReady0 = 0xA;
constexpr uint16_t kHeadReady1 = 0xB;
constexpr uint16_t kHeadDone0 = 0xC;
constexpr uint16_t kHeadDone1 = 0xD;

// Workspace offsets are in half elements except CB, whose byte offset is
// converted before binding the float GlobalTensor.
constexpr uint32_t kBtHalfOffset = 0;

class VectorChunkKernel {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(transposeIn_, 1, kTransposeElements * sizeof(half));
        pipe->InitBuffer(transposeOut_, 1, kTransposeElements * sizeof(half));
        pipe->InitBuffer(dAIn_, 1, kTile * sizeof(float));
        pipe->InitBuffer(cbBlockIn_, 1, kRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(halfBlockOut_, 1,
                         kRowsPerAiv * kTile * sizeof(half));
        pipe->InitBuffer(floatBlock0_,
                         kRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(floatBlock1_,
                         kRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(causalMask_,
                         kRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(decayHalf_, kTile * sizeof(half));
        pipe->InitBuffer(decayTileHalf_,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(broadcastTmp_,
                         2 * kRowsPerAiv * kTile * sizeof(uint8_t));

        // The triangular mask is invariant across every head/task handled by
        // this AIV.  Build it once instead of issuing two Duplicate commands
        // for every output row of every task.
        auto mask = causalMask_.Get<float>();
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        for (uint32_t localRow = 0; localRow < kRowsPerAiv; ++localRow) {
            auto rowMask = mask[localRow * kTile];
            Duplicate(rowMask, 0.0f, kTile);
            PipeBarrier<PIPE_V>();
            Duplicate(rowMask, 1.0f, rowBegin + localRow + 1);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareB(const GlobalTensor<half> &b,
                                    GlobalTensor<half> &bT,
                                    uint32_t stateDim)
    {
        const uint32_t subBlock = GetSubBlockIdx();
        const uint32_t rowBegin = subBlock * kRowsPerAiv;
        const uint32_t rowEnd = rowBegin + kRowsPerAiv;

        // B[T,N] -> B^T[N,T], with the two AIVs owning disjoint source-row
        // ranges and therefore disjoint destination-column ranges.
        for (uint32_t row = rowBegin; row < rowEnd; row += kTransposeTile) {
            for (uint32_t col = 0; col < stateDim; col += kTransposeTile) {
                TransposeHalfTile(b[row * stateDim + col],
                                  bT[col * kTile + row], stateDim, kTile);
            }
        }
    }

    __aicore__ inline void PrepareLegacyB(
        const GlobalTensor<half> &bT, GlobalTensor<half> &b,
        uint32_t stateDim)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        const uint32_t rowEnd = rowBegin + kRowsPerAiv;

        // Only Key 3 needs B[T,N] for [P,T]@[T,N].  The public input is
        // B^T[N,T], so transpose the two AIV-owned token-row ranges into the
        // legacy workspace without placing B^T in workspace first.
        for (uint32_t row = rowBegin; row < rowEnd; row += kTransposeTile) {
            for (uint32_t col = 0; col < stateDim; col += kTransposeTile) {
                TransposeHalfTile(bT[col * kTile + row],
                                  b[row * stateDim + col],
                                  kTile, stateDim);
            }
        }
    }

    __aicore__ inline void PrepareHead(const GlobalTensor<half> &x,
                                       const GlobalTensor<float> &dA,
                                       GlobalTensor<half> &weightedX,
                                       uint32_t stateNpLayout)
    {
        const uint32_t subBlock = GetSubBlockIdx();
        const uint32_t rowBegin = subBlock * kRowsPerAiv;
        const uint32_t rowEnd = rowBegin + kRowsPerAiv;

        dALocal_ = dAIn_.AllocTensor<float>();
        DataCopy(dALocal_, dA, kTile);
        dAIn_.EnQue(dALocal_);
        dALocal_ = dAIn_.DeQue<float>();

        auto decay = floatBlock0_.Get<float>();
        const float lastDA = dALocal_.GetValue(kTile - 1);
        Muls(decay, dALocal_, -1.0f, kTile);
        PipeBarrier<PIPE_V>();
        Adds(decay, decay, lastDA, kTile);
        PipeBarrier<PIPE_V>();
        Exp(decay, decay, kTile);
        PipeBarrier<PIPE_V>();

        auto decayHalf = decayHalf_.Get<half>();
        Cast(decayHalf, decay, RoundMode::CAST_RINT, kTile);
        PipeBarrier<PIPE_V>();

        // The persistent state path consumes B^T[N,T] @ weighted-X[T,P] and
        // therefore keeps weighted-X in ND.  The state_passing fallback still
        // consumes weighted-X^T[P,T] @ B[T,N].
        for (uint32_t row = rowBegin; row < rowEnd; row += kTransposeTile) {
            for (uint32_t col = 0; col < kTile; col += kTransposeTile) {
                if (stateNpLayout != 0) {
                    WeightedHalfTile(
                        x[row * kTile + col],
                        weightedX[row * kTile + col], decayHalf[row],
                        kTile, kTile);
                } else {
                    WeightedTransposeHalfTile(
                        x[row * kTile + col],
                        weightedX[col * kTile + row], decayHalf[row],
                        kTile, kTile);
                }
            }
        }
    }

    __aicore__ inline void BuildWeights(const GlobalTensor<float> &cb,
                                        GlobalTensor<half> &w)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        const uint32_t rowEnd = rowBegin + kRowsPerAiv;
        constexpr uint32_t blockElements = kRowsPerAiv * kTile;
        auto colDA = floatBlock0_.Get<float>();
        auto rowDA = floatBlock1_.Get<float>();
        auto causalMask = causalMask_.Get<float>();
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t colSrcShape[2] = {1U, kTile};
        const uint32_t rowSrcShape[2] = {kRowsPerAiv, 1U};
        const uint32_t dstShape[2] = {kRowsPerAiv, kTile};
        Broadcast<float, 2, 0>(colDA, dALocal_, dstShape,
                               colSrcShape, broadcastTmp);
        PipeBarrier<PIPE_V>();
        Broadcast<float, 2, 1>(rowDA, dALocal_[rowBegin], dstShape,
                               rowSrcShape, broadcastTmp);
        PipeBarrier<PIPE_V>();
        Sub(colDA, rowDA, colDA, blockElements);
        PipeBarrier<PIPE_V>();
        Mul(colDA, colDA, causalMask, blockElements);
        PipeBarrier<PIPE_V>();
        Exp(colDA, colDA, blockElements);
        PipeBarrier<PIPE_V>();

        auto cbBlock = cbBlockIn_.AllocTensor<float>();
        DataCopy(cbBlock, cb[rowBegin * kTile], blockElements);
        cbBlockIn_.EnQue(cbBlock);
        cbBlock = cbBlockIn_.DeQue<float>();
        Mul(cbBlock, cbBlock, colDA, blockElements);
        PipeBarrier<PIPE_V>();
        Mul(cbBlock, cbBlock, causalMask, blockElements);
        PipeBarrier<PIPE_V>();
        auto halfBlock = halfBlockOut_.AllocTensor<half>();
        Cast(halfBlock, cbBlock, RoundMode::CAST_RINT, blockElements);
        cbBlockIn_.FreeTensor(cbBlock);
        halfBlockOut_.EnQue(halfBlock);
        halfBlock = halfBlockOut_.DeQue<half>();
        DataCopy(w[rowBegin * kTile], halfBlock, blockElements);
        halfBlockOut_.FreeTensor(halfBlock);
        dAIn_.FreeTensor(dALocal_);
    }

private:
    __aicore__ inline void WeightedHalfTile(
        const GlobalTensor<half> &src, GlobalTensor<half> dst,
        const LocalTensor<half> &rowDecay, uint32_t srcStride,
        uint32_t dstStride)
    {
        auto local = transposeIn_.AllocTensor<half>();
        DataCopyParams loadParams{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((srcStride - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(local, src, loadParams);
        transposeIn_.EnQue(local);
        local = transposeIn_.DeQue<half>();

        auto decayTile = decayTileHalf_.Get<half>();
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t decaySrcShape[2] = {kTransposeTile, 1U};
        const uint32_t tileShape[2] = {kTransposeTile, kTransposeTile};
        Broadcast<half, 2, 1>(decayTile, rowDecay, tileShape,
                              decaySrcShape, broadcastTmp);
        PipeBarrier<PIPE_V>();
        Mul(local, local, decayTile, kTransposeElements);
        PipeBarrier<PIPE_V>();

        auto output = transposeOut_.AllocTensor<half>();
        Muls(output, local, static_cast<half>(1.0f), kTransposeElements);
        transposeIn_.FreeTensor(local);
        transposeOut_.EnQue(output);
        output = transposeOut_.DeQue<half>();
        DataCopyParams storeParams{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((dstStride - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE)};
        DataCopy(dst, output, storeParams);
        transposeOut_.FreeTensor(output);
    }

    __aicore__ inline void WeightedTransposeHalfTile(
        const GlobalTensor<half> &src, GlobalTensor<half> dst,
        const LocalTensor<half> &rowDecay, uint32_t srcStride,
        uint32_t dstStride)
    {
        auto transposeInLocal = transposeIn_.AllocTensor<half>();
        DataCopyParams loadParams{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((srcStride - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(transposeInLocal, src, loadParams);
        transposeIn_.EnQue(transposeInLocal);
        transposeInLocal = transposeIn_.DeQue<half>();

        auto decayTile = decayTileHalf_.Get<half>();
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t decaySrcShape[2] = {kTransposeTile, 1U};
        const uint32_t tileShape[2] = {kTransposeTile, kTransposeTile};
        Broadcast<half, 2, 1>(decayTile, rowDecay, tileShape,
                              decaySrcShape, broadcastTmp);
        PipeBarrier<PIPE_V>();
        Mul(transposeInLocal, transposeInLocal, decayTile,
            kTransposeElements);
        PipeBarrier<PIPE_V>();

        auto transposeOutLocal = transposeOut_.AllocTensor<half>();
        Transpose(transposeOutLocal, transposeInLocal);
        transposeIn_.FreeTensor(transposeInLocal);
        transposeOut_.EnQue(transposeOutLocal);
        transposeOutLocal = transposeOut_.DeQue<half>();
        DataCopyParams storeParams{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((dstStride - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE)};
        DataCopy(dst, transposeOutLocal, storeParams);
        transposeOut_.FreeTensor(transposeOutLocal);
    }

    __aicore__ inline void TransposeHalfTile(const GlobalTensor<half> &src,
                                             GlobalTensor<half> dst,
                                             uint32_t srcStride,
                                             uint32_t dstStride)
    {
        auto transposeInLocal = transposeIn_.AllocTensor<half>();
        DataCopyParams loadParams{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((srcStride - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(transposeInLocal, src, loadParams);
        transposeIn_.EnQue(transposeInLocal);
        transposeInLocal = transposeIn_.DeQue<half>();

        auto transposeOutLocal = transposeOut_.AllocTensor<half>();
        Transpose(transposeOutLocal, transposeInLocal);
        transposeIn_.FreeTensor(transposeInLocal);
        transposeOut_.EnQue(transposeOutLocal);
        transposeOutLocal = transposeOut_.DeQue<half>();

        DataCopyParams storeParams{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) / DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((dstStride - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE)};
        DataCopy(dst, transposeOutLocal, storeParams);
        transposeOut_.FreeTensor(transposeOutLocal);
    }

    TQue<TPosition::VECIN, 1> transposeIn_;
    TQue<TPosition::VECOUT, 1> transposeOut_;
    TQue<TPosition::VECIN, 1> dAIn_;
    TQue<TPosition::VECIN, 1> cbBlockIn_;
    TQue<TPosition::VECOUT, 1> halfBlockOut_;
    TBuf<TPosition::VECCALC> floatBlock0_;
    TBuf<TPosition::VECCALC> floatBlock1_;
    TBuf<TPosition::VECCALC> causalMask_;
    TBuf<TPosition::VECCALC> decayHalf_;
    TBuf<TPosition::VECCALC> decayTileHalf_;
    TBuf<TPosition::VECCALC> broadcastTmp_;

    LocalTensor<float> dALocal_;
};

#ifndef MAMBA2_CHUNK_MIX_COMPONENT_ONLY
template <uint32_t MatmulK>
class KernelMamba2SsdChunkMix {
public:
    __aicore__ inline void Init(GM_ADDR xCube, GM_ADDR dACumsum,
                                GM_ADDR bCube, GM_ADDR cCube,
                                GM_ADDR yDiag, GM_ADDR chunkStates,
                                GM_ADDR workspace,
                                const Mamba2SsdChunkMixTilingData &tiling,
                                TPipe *pipe)
    {
        tiling_ = tiling;
        const uint64_t taskMatrixElements =
            static_cast<uint64_t>(tiling_.headTaskCount) * kTileElements;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(xCube),
                             taskMatrixElements);
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum),
                              static_cast<uint64_t>(tiling_.headTaskCount) * kTile);
        const uint64_t bcMatrices = static_cast<uint64_t>(tiling_.batch) *
                                    tiling_.chunks * tiling_.groups;
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bCube),
                             bcMatrices * kTile * tiling_.stateDim);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube),
                             bcMatrices * kTile * tiling_.stateDim);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yDiag),
                             taskMatrixElements);
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(chunkStates),
                                 static_cast<uint64_t>(tiling_.headTaskCount) *
                                     kTile * tiling_.stateDim);

        legacyBHalfElements_ = tiling_.stateNpLayout == 0
            ? kTile * tiling_.stateDim : 0;
        weightedXtHalfOffset_[0] = legacyBHalfElements_;
        weightedXtHalfOffset_[1] = weightedXtHalfOffset_[0] + kTileElements;
        wHalfOffset_[0] = weightedXtHalfOffset_[1] + kTileElements;
        wHalfOffset_[1] = wHalfOffset_[0] + kTileElements;
        cbByteOffset_ =
            (wHalfOffset_[1] + kTileElements) * sizeof(half);

        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) * tiling_.workspaceBytesPerCore;
        workspaceHalf_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(coreWorkspace),
                                       cbByteOffset_ / sizeof(half));
        cbWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(coreWorkspace + cbByteOffset_),
            kTileElements);

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
            uint32_t b = 0;
            uint32_t k = 0;
            uint32_t group = 0;
            uint32_t headBegin = 0;
            uint32_t headCount = 0;
            DecodeTask(task, b, k, group, headBegin, headCount);
            const uint32_t bcIndex =
                (b * tiling_.chunks + k) * tiling_.groups + group;

            const uint64_t bcOffset = static_cast<uint64_t>(bcIndex) *
                                      kTile * tiling_.stateDim;
            auto bT = bGm_[bcOffset];
            auto cMat = cGm_[bcOffset];
            auto legacyB = workspaceHalf_[kBtHalfOffset];

            if ASCEND_IS_AIV {
                if (tiling_.stateNpLayout == 0) {
                    vector_.PrepareLegacyB(bT, legacyB, tiling_.stateDim);
                }
                CrossCoreWaitFlag<0x2>(kCubeCbReady);
                for (uint32_t headOffset = 0; headOffset < headCount;
                     ++headOffset) {
                    const uint32_t slot = headOffset & 1U;
                    if (headOffset >= 2) {
                        WaitHeadDone(slot);
                    }
                    const uint32_t h = headBegin + headOffset;
                    const uint32_t headTask =
                        (b * tiling_.heads + h) * tiling_.chunks + k;
                    auto weightedXT =
                        workspaceHalf_[weightedXtHalfOffset_[slot]];
                    auto w = workspaceHalf_[wHalfOffset_[slot]];
                    vector_.PrepareHead(
                        xGm_[static_cast<uint64_t>(headTask) * kTileElements],
                        dAGm_[static_cast<uint64_t>(headTask) * kTile],
                        weightedXT, tiling_.stateNpLayout);
                    vector_.BuildWeights(cbWorkspace_, w);
                    SetHeadReady(slot);
                }
                // Consume the completion counters for the one or two slots
                // that remain live after the producer loop.
                if (headCount == 1) {
                    WaitHeadDone(0);
                } else if (headCount >= 2) {
                    WaitHeadDone((headCount - 2) & 1U);
                    WaitHeadDone((headCount - 1) & 1U);
                }
            }

            if ASCEND_IS_AIC {
                matmul_.ComputeBlock(cMat, bT, cbWorkspace_,
                                     tiling_.stateDim, tiling_.stateDim,
                                     kTile, kTile);
                CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeCbReady);
                for (uint32_t headOffset = 0; headOffset < headCount;
                     ++headOffset) {
                    const uint32_t slot = headOffset & 1U;
                    const uint32_t h = headBegin + headOffset;
                    const uint32_t headTask =
                        (b * tiling_.heads + h) * tiling_.chunks + k;
                    auto x = xGm_[static_cast<uint64_t>(headTask) *
                                  kTileElements];
                    auto y = yGm_[static_cast<uint64_t>(headTask) *
                                  kTileElements];
                    auto state = stateGm_[static_cast<uint64_t>(headTask) *
                                          kTile * tiling_.stateDim];
                    auto weightedXT =
                        workspaceHalf_[weightedXtHalfOffset_[slot]];
                    auto w = workspaceHalf_[wHalfOffset_[slot]];
                    WaitHeadReady(slot);
                    matmul_.ComputeBlock(w, x, y,
                                         kTile, kTile, kTile, kTile);
                    if (tiling_.stateNpLayout != 0) {
                        // B^T[N,T] @ weighted-X[T,P] writes Cube-native
                        // [N,P].  N=128 is two independent 64-row blocks.
                        for (uint32_t n = 0; n < tiling_.stateDim;
                             n += kTile) {
                            matmul_.ComputeBlock(
                                bT[n * kTile], weightedXT,
                                state[n * kTile], kTile, kTile, kTile,
                                kTile);
                        }
                    } else {
                        // Legacy [P,T] @ [T,N] -> [P,N] for the unfused
                        // state_passing fallback.
                        matmul_.ComputeBlock(
                            weightedXT, legacyB, state,
                            kTile, kTile, tiling_.stateDim, tiling_.stateDim,
                            tiling_.stateDim);
                    }
                    SetHeadDone(slot);
                }
            }
        }
    }

private:
    __aicore__ inline void SetHeadReady(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(kHeadReady0);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(kHeadReady1);
        }
    }

    __aicore__ inline void WaitHeadReady(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreWaitFlag<0x2>(kHeadReady0);
        } else {
            CrossCoreWaitFlag<0x2>(kHeadReady1);
        }
    }

    __aicore__ inline void SetHeadDone(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(kHeadDone0);
        } else {
            CrossCoreSetFlag<0x2, PIPE_FIX>(kHeadDone1);
        }
    }

    __aicore__ inline void WaitHeadDone(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreWaitFlag<0x2>(kHeadDone0);
        } else {
            CrossCoreWaitFlag<0x2>(kHeadDone1);
        }
    }

    __aicore__ inline void DecodeTask(uint32_t task, uint32_t &b,
                                      uint32_t &k, uint32_t &group,
                                      uint32_t &headBegin,
                                      uint32_t &headCount) const
    {
        k = task % tiling_.chunks;
        const uint32_t outer = task / tiling_.chunks;
        if (tiling_.taskMode == 1) {
            group = outer % tiling_.groups;
            b = outer / tiling_.groups;
            headBegin = group * tiling_.headsPerGroup;
            headCount = tiling_.headsPerGroup;
        } else {
            const uint32_t h = outer % tiling_.heads;
            b = outer / tiling_.heads;
            group = h / tiling_.headsPerGroup;
            headBegin = h;
            headCount = 1;
        }
    }

    Mamba2DynamicMatmul<half, float, MatmulK> matmul_;
    VectorChunkKernel vector_;
    GlobalTensor<half> xGm_;
    GlobalTensor<float> dAGm_;
    GlobalTensor<half> bGm_;
    GlobalTensor<half> cGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> stateGm_;
    GlobalTensor<half> workspaceHalf_;
    GlobalTensor<float> cbWorkspace_;
    Mamba2SsdChunkMixTilingData tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t legacyBHalfElements_ = 0;
    uint32_t weightedXtHalfOffset_[2] = {0, 0};
    uint32_t wHalfOffset_[2] = {0, 0};
    uint32_t cbByteOffset_ = 0;
};
#endif

} // namespace

#ifndef MAMBA2_CHUNK_MIX_COMPONENT_ONLY
extern "C" __global__ __aicore__ void mamba2_ssd_chunk_mix(
    GM_ADDR x_cube, GM_ADDR d_a_cumsum, GM_ADDR b_cube, GM_ADDR c_cube,
    GM_ADDR y_diag, GM_ADDR chunk_states, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(2)) {
        KernelMamba2SsdChunkMix<16> op;
        op.Init(x_cube, d_a_cumsum, b_cube, c_cube, y_diag, chunk_states,
                workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelMamba2SsdChunkMix<64> op;
        op.Init(x_cube, d_a_cumsum, b_cube, c_cube, y_diag, chunk_states,
                workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        Mamba2Chunk128::KernelMamba2SsdChunkMix128 op;
        op.Init(x_cube, d_a_cumsum, b_cube, c_cube, y_diag, chunk_states,
                workspace, tilingData, &pipe);
        op.Process();
    }
}
#endif
