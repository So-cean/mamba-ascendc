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
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements = kTransposeTile * kTransposeTile;

constexpr uint16_t kCubeCbReady = 0x9;
constexpr uint16_t kHeadReady0 = 0xA;
constexpr uint16_t kHeadReady1 = 0xB;
constexpr uint16_t kHeadDone0 = 0xC;
constexpr uint16_t kHeadDone1 = 0xD;
constexpr uint16_t kGroupedStateReady = 0xE;

// Workspace offsets are in half elements except CB, whose byte offset is
// converted before binding the float GlobalTensor.
constexpr uint32_t kBtHalfOffset = 0;

template <uint32_t RowsPerAiv>
class VectorChunkKernel {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(transposeIn_, 1, kTransposeElements * sizeof(half));
        pipe->InitBuffer(transposeOut_, 1, kTransposeElements * sizeof(half));
        pipe->InitBuffer(dAIn_, 1, kTile * sizeof(float));
        pipe->InitBuffer(cbBlockIn_, 1, RowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(halfBlockOut_, 1,
                         RowsPerAiv * kTile * sizeof(half));
        pipe->InitBuffer(weightedBlockIn_, 1,
                         RowsPerAiv * kTile * sizeof(half));
        pipe->InitBuffer(floatBlock0_,
                         RowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(floatBlock1_,
                         RowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(causalMask_,
                         RowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(decayHalf_, kTile * sizeof(half));
        pipe->InitBuffer(decayTileHalf_,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(weightedScaleHalf_,
                         RowsPerAiv * kTile * sizeof(half));
        pipe->InitBuffer(broadcastTmp_,
                         2 * RowsPerAiv * kTile * sizeof(uint8_t));
        pipe->InitBuffer(stateScatterIn_, 1,
                         kTileElements * sizeof(float));

        // The triangular mask is invariant across every head/task handled by
        // this AIV.  Build it once instead of issuing two Duplicate commands
        // for every output row of every task.
        auto mask = causalMask_.Get<float>();
        const uint32_t rowBegin = GetSubBlockIdx() * RowsPerAiv;
        for (uint32_t localRow = 0; localRow < RowsPerAiv; ++localRow) {
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
        const uint32_t rowBegin = subBlock * RowsPerAiv;
        const uint32_t rowEnd = rowBegin + RowsPerAiv;

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
        const uint32_t rowBegin = GetSubBlockIdx() * RowsPerAiv;
        const uint32_t rowEnd = rowBegin + RowsPerAiv;

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
                                       uint32_t stateNpLayout,
                                       uint32_t weightedXStride = kTile,
                                       uint32_t xStride = kTile)
    {
        const uint32_t subBlock = GetSubBlockIdx();
        const uint32_t rowBegin = subBlock * RowsPerAiv;
        const uint32_t rowEnd = rowBegin + RowsPerAiv;

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

        // The persistent state path consumes B^T[N,T] @ weighted-X[T,P].
        // Load the complete AIV-owned row block once, scale it in UB and use
        // one strided store into the group-packed [T,R*P] destination.  The
        // old 16x16 loop issued many small MTE2/MTE3 transactions per head.
        if (stateNpLayout != 0) {
            WeightedHalfBlock(
                x[rowBegin * xStride],
                weightedX[rowBegin * weightedXStride],
                decayHalf[rowBegin], weightedXStride, xStride);
            return;
        }

        // The state_passing fallback consumes weighted-X^T[P,T] @ B[T,N].
        for (uint32_t row = rowBegin; row < rowEnd; row += kTransposeTile) {
            for (uint32_t col = 0; col < kTile; col += kTransposeTile) {
                WeightedTransposeHalfTile(
                    x[row * kTile + col],
                    weightedX[col * kTile + row], decayHalf[row],
                    kTile, kTile);
            }
        }
    }

    __aicore__ inline void BuildWeights(const GlobalTensor<float> &cb,
        GlobalTensor<half> &w)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * RowsPerAiv;
        constexpr uint32_t blockElements = RowsPerAiv * kTile;
        auto colDA = floatBlock0_.Get<float>();
        auto rowDA = floatBlock1_.Get<float>();
        auto causalMask = causalMask_.Get<float>();
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t colSrcShape[2] = {1U, kTile};
        const uint32_t rowSrcShape[2] = {RowsPerAiv, 1U};
        const uint32_t dstShape[2] = {RowsPerAiv, kTile};
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

    __aicore__ inline void ScatterPackedState(
        const GlobalTensor<float> &packed,
        GlobalTensor<float> &state,
        uint32_t packedStride)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * RowsPerAiv;
        constexpr uint32_t slabElements = RowsPerAiv * kTile;
        auto local = stateScatterIn_.AllocTensor<float>();
        DataCopyParams loadParams{
            static_cast<uint16_t>(RowsPerAiv),
            static_cast<uint16_t>(kTile * sizeof(float) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((packedStride - kTile) * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(local, packed[rowBegin * packedStride], loadParams);
        stateScatterIn_.EnQue(local);
        local = stateScatterIn_.DeQue<float>();
        DataCopy(state[rowBegin * kTile], local, slabElements);
        stateScatterIn_.FreeTensor(local);
    }

private:
    __aicore__ inline void WeightedHalfBlock(
        const GlobalTensor<half> &src, GlobalTensor<half> dst,
        const LocalTensor<half> &rowDecay, uint32_t dstStride,
        uint32_t srcStride)
    {
        constexpr uint32_t blockElements = RowsPerAiv * kTile;
        auto input = weightedBlockIn_.AllocTensor<half>();
        DataCopyParams loadParams{
            static_cast<uint16_t>(RowsPerAiv),
            static_cast<uint16_t>(kTile * sizeof(half) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((srcStride - kTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(input, src, loadParams);
        weightedBlockIn_.EnQue(input);
        input = weightedBlockIn_.DeQue<half>();

        auto scale = weightedScaleHalf_.Get<half>();
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t srcShape[2] = {RowsPerAiv, 1U};
        const uint32_t dstShape[2] = {RowsPerAiv, kTile};
        Broadcast<half, 2, 1>(scale, rowDecay, dstShape, srcShape,
                              broadcastTmp);
        PipeBarrier<PIPE_V>();
        auto output = halfBlockOut_.AllocTensor<half>();
        Mul(output, input, scale, blockElements);
        weightedBlockIn_.FreeTensor(input);
        PipeBarrier<PIPE_V>();
        halfBlockOut_.EnQue(output);
        output = halfBlockOut_.DeQue<half>();

        DataCopyParams storeParams{
            static_cast<uint16_t>(RowsPerAiv),
            static_cast<uint16_t>(kTile * sizeof(half) / DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((dstStride - kTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE)};
        DataCopy(dst, output, storeParams);
        halfBlockOut_.FreeTensor(output);
    }

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
    TQue<TPosition::VECIN, 1> weightedBlockIn_;
    TBuf<TPosition::VECCALC> floatBlock0_;
    TBuf<TPosition::VECCALC> floatBlock1_;
    TBuf<TPosition::VECCALC> causalMask_;
    TBuf<TPosition::VECCALC> decayHalf_;
    TBuf<TPosition::VECCALC> decayTileHalf_;
    TBuf<TPosition::VECCALC> weightedScaleHalf_;
    TBuf<TPosition::VECCALC> broadcastTmp_;
    TQue<TPosition::VECIN, 1> stateScatterIn_;

    LocalTensor<float> dALocal_;
};

#ifndef MAMBA2_CHUNK_MIX_COMPONENT_ONLY
constexpr MatmulConfig kArch35MatmulConfig = GetBasicConfig(64, 64, 64);
using Arch35AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Arch35BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Arch35CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using Arch35BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using Arch35BatchAType = MatmulType<TPosition::GM, CubeFormat::ND, half,
                                    false, LayoutMode::NORMAL>;
using Arch35BatchBType = MatmulType<TPosition::GM, CubeFormat::ND, half,
                                    false, LayoutMode::NORMAL>;
using Arch35BatchCType = MatmulType<TPosition::GM, CubeFormat::ND, float,
                                    false, LayoutMode::NORMAL>;
using Arch35BatchBiasType = MatmulType<TPosition::GM, CubeFormat::ND, float,
                                       false, LayoutMode::NORMAL>;

class Arch35Matmul64 {
public:
    __aicore__ inline void Init(const TCubeTiling *cubeTiling)
    {
        if ASCEND_IS_AIC {
            object_.Init(cubeTiling);
        }
    }

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &a,
        const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        object_.SetOrgShape(kTile, kTile, kTile, kTile, kTile);
        object_.SetSingleShape(kTile, kTile, kTile);
        object_.SetTensorA(a, false);
        object_.SetTensorB(b, false);
        object_.IterateAll(c, false);
        object_.End();
    }

    __aicore__ inline void ComputeBlockNoEnd(
        const GlobalTensor<half> &a,
        const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        object_.SetOrgShape(kTile, kTile, kTile, kTile, kTile);
        object_.SetSingleShape(kTile, kTile, kTile);
        object_.SetTensorA(a, false);
        object_.SetTensorB(b, false);
        object_.IterateAll(c, false);
    }

    __aicore__ inline void ComputeBlockStridedNoEnd(
        const GlobalTensor<half> &a,
        const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        object_.SetOrgShape(kTile, 4 * kTile, kTile,
                            kTile, 4 * kTile);
        object_.SetSingleShape(kTile, kTile, kTile);
        object_.SetTensorA(a, false);
        object_.SetTensorB(b, false);
        object_.IterateAll(c, false);
    }

    __aicore__ inline void End()
    {
        object_.End();
    }

private:
    matmul::MatmulImpl<Arch35AType, Arch35BType, Arch35CType,
                       Arch35BiasType, kArch35MatmulConfig> object_;
};

constexpr MatmulConfig kArch35GroupedStateConfig =
    GetBasicConfig(64, 256, 64);

class Arch35GroupedStateMatmul {
public:
    __aicore__ inline void Init(const TCubeTiling *cubeTiling)
    {
        if ASCEND_IS_AIC {
            object_.Init(cubeTiling);
        }
    }

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &a,
        const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        object_.SetOrgShape(kTile, 4 * kTile, kTile,
                            kTile, 4 * kTile);
        object_.SetSingleShape(kTile, 4 * kTile, kTile);
        object_.SetTensorA(a, false);
        object_.SetTensorB(b, false);
        object_.IterateAll(c, false);
        object_.End();
    }

private:
    matmul::MatmulImpl<Arch35AType, Arch35BType, Arch35CType,
                       Arch35BiasType,
                       kArch35GroupedStateConfig> object_;
};

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
constexpr MatmulConfig kArch35BatchDiagConfig = GetNormalConfig(
    false, false, false, BatchMode::BATCH_LESS_THAN_L1, true,
    IterateOrder::ORDER_M, ScheduleType::INNER_PRODUCT, true, false,
    BatchOutMode::MULTI_BATCH);

class Arch35BatchDiagMatmul {
public:
    __aicore__ inline void Init(const TCubeTiling *cubeTiling)
    {
        if ASCEND_IS_AIC {
            object_.Init(cubeTiling);
        }
    }

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &packedW,
        const GlobalTensor<half> &x,
        const GlobalTensor<float> &y,
        uint32_t xMatrixStride,
        uint32_t yMatrixStride)
    {
        object_.SetOrgShape(kTile, kTile, kTile, kTile, kTile);
        object_.SetSingleShape(kTile, kTile, kTile);
        object_.SetTensorA(packedW, false);
        object_.SetTensorB(x, false);
        object_.SetBatchNum(4, 4);
        object_.SetNBatchOutNum(4);
        object_.IterateBatch(
            y, false, 0, false,
            kTileElements, xMatrixStride, yMatrixStride);
        object_.End();
    }

private:
    matmul::MatmulImpl<Arch35BatchAType, Arch35BatchBType,
                       Arch35BatchCType, Arch35BatchBiasType,
                       kArch35BatchDiagConfig> object_;
};
#else
// BatchOutMode::MULTI_BATCH is a CANN 9 / Ascend 950 interface.  Keep the
// class shape available so the common template parses under CANN 8.2, while
// the 910B host never selects tiling key 10.
class Arch35BatchDiagMatmul {
public:
    __aicore__ inline void Init(const TCubeTiling *) {}

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &, const GlobalTensor<half> &,
        const GlobalTensor<float> &, uint32_t, uint32_t) {}
};
#endif

class Arch35ChunkMixMatmuls {
public:
    __aicore__ inline void Init(
        const TCubeTiling *cubeTiling,
        const TCubeTiling *groupedStateCubeTiling, TPipe *)
    {
        matmul_.Init(cubeTiling);
        groupedStateMatmul_.Init(groupedStateCubeTiling);
    }

    __aicore__ inline void InitBatchDiag(
        const TCubeTiling *batchDiagCubeTiling)
    {
        batchDiagMatmul_.Init(batchDiagCubeTiling);
    }

    __aicore__ inline void Compute64(
        const GlobalTensor<half> &a, const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        matmul_.ComputeBlock(a, b, c);
    }

    __aicore__ inline void Compute64NoEnd(
        const GlobalTensor<half> &a, const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        matmul_.ComputeBlockNoEnd(a, b, c);
    }

    __aicore__ inline void Compute64StridedNoEnd(
        const GlobalTensor<half> &a, const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        matmul_.ComputeBlockStridedNoEnd(a, b, c);
    }

    __aicore__ inline void End64()
    {
        matmul_.End();
    }

    __aicore__ inline void ComputeWide(
        const GlobalTensor<half> &a, const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        groupedStateMatmul_.ComputeBlock(a, b, c);
    }

    __aicore__ inline void ComputeBatchDiag(
        const GlobalTensor<half> &packedW,
        const GlobalTensor<half> &x,
        const GlobalTensor<float> &y,
        uint32_t xMatrixStride,
        uint32_t yMatrixStride)
    {
        batchDiagMatmul_.ComputeBlock(
            packedW, x, y, xMatrixStride, yMatrixStride);
    }

private:
    Arch35Matmul64 matmul_;
    Arch35GroupedStateMatmul groupedStateMatmul_;
    Arch35BatchDiagMatmul batchDiagMatmul_;
};

// Ascend 950PR path. Every dense product is 64x64x64.  Direct MatmulImpl keeps
// the Cube side independent from the AIV producer so two GM workspace slots
// can overlap Vector preparation for head h+1 with Cube work for head h.
template <bool UseBatchDiag = false>
class KernelMamba2SsdChunkMixArch35 {
public:
    __aicore__ inline void Init(
        GM_ADDR xCube, GM_ADDR dACumsum, GM_ADDR bCube, GM_ADDR cCube,
        GM_ADDR yDiag, GM_ADDR chunkStates, GM_ADDR workspace,
        const Mamba2SsdChunkMixTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        groupedInput_ = tiling_.inputGroupedX != 0;
        const uint64_t taskMatrixElements =
            static_cast<uint64_t>(tiling_.headTaskCount) * kTileElements;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(xCube),
                             taskMatrixElements);
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum),
                              static_cast<uint64_t>(tiling_.headTaskCount) *
                                  kTile);
        const uint64_t bcMatrices = static_cast<uint64_t>(tiling_.batch) *
                                    tiling_.chunks * tiling_.groups;
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bCube),
                             bcMatrices * kTileElements);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube),
                             bcMatrices * kTileElements);
        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(yDiag),
                             taskMatrixElements);
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(chunkStates),
                                 taskMatrixElements);

        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) * tiling_.workspaceBytesPerCore;
        packedWeightedX_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(coreWorkspace),
            4 * kTileElements);
        constexpr uint32_t packedWHalfTiles = UseBatchDiag ? 4U : 2U;
        packedW_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(
                coreWorkspace + 4 * kTileElements * sizeof(half)),
            packedWHalfTiles * kTileElements);
        constexpr uint32_t packedHalfTiles = 4U + packedWHalfTiles;
        constexpr uint32_t packedStateFloatTiles = UseBatchDiag ? 0U : 4U;
        packedState_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(
                coreWorkspace + packedHalfTiles * kTileElements *
                    sizeof(half)),
            4 * kTileElements);
        cbWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(
                coreWorkspace + packedHalfTiles * kTileElements *
                    sizeof(half) +
                    packedStateFloatTiles * kTileElements * sizeof(float)),
            kTileElements);

        matmuls_.Init(&tiling_.cubeTilingData,
                      &tiling_.groupedStateCubeTilingData, pipe);
        if (UseBatchDiag) {
            matmuls_.InitBatchDiag(&tiling_.batchDiagCubeTilingData);
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
            const uint64_t bcOffset =
                static_cast<uint64_t>(bcIndex) * kTileElements;
            auto bT = bGm_[bcOffset];
            auto cMat = cGm_[bcOffset];
            const bool groupedState = headCount == 4;

            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<0x2>(kCubeCbReady);
                if (groupedState) {
#ifdef MAMBA2_CHUNK_MIX_GROUPED_OUTPUT
                    const uint64_t groupMatrixBase =
                        static_cast<uint64_t>(bcIndex) * 4U * kTileElements;
                    for (uint32_t headOffset = 0; headOffset < headCount;
                         ++headOffset) {
                        const uint32_t slot = UseBatchDiag
                            ? headOffset : (headOffset & 1U);
                        if (!UseBatchDiag && headOffset >= 2) {
                            WaitHeadDone(slot);
                        }
                        const uint32_t h = headBegin + headOffset;
                        const uint32_t headTask =
                            (b * tiling_.heads + h) * tiling_.chunks + k;
                        auto x = groupedInput_
                            ? xGm_[groupMatrixBase + headOffset * kTile]
                            : xGm_[static_cast<uint64_t>(headTask) *
                                   kTileElements];
                        auto weightedX =
                            packedWeightedX_[headOffset * kTile];
                        auto packedW = packedW_[slot * kTileElements];
                        vector_.PrepareHead(
                            x,
                            dAGm_[static_cast<uint64_t>(headTask) * kTile],
                            weightedX, 1U, 4U * kTile,
                            groupedInput_ ? 4U * kTile : kTile);
                        vector_.BuildWeights(
                            cbWorkspace_, packedW);
                        if (!UseBatchDiag) {
                            SetHeadReady(slot);
                        }
                    }
                    if (UseBatchDiag) {
                        SetHeadReady(0);
                        WaitHeadDone(0);
                    } else {
                        WaitHeadDone((headCount - 2) & 1U);
                        WaitHeadDone((headCount - 1) & 1U);
                    }
                    CrossCoreWaitFlag<0x2>(kGroupedStateReady);
#else
                    for (uint32_t headOffset = 0; headOffset < headCount;
                         ++headOffset) {
                        const uint32_t slot = headOffset & 1U;
                        if (headOffset >= 2) {
                            WaitHeadDone(slot);
                        }
                        const uint32_t h = headBegin + headOffset;
                        const uint32_t headTask =
                            (b * tiling_.heads + h) * tiling_.chunks + k;
                        auto x = xGm_[static_cast<uint64_t>(headTask) *
                                      kTileElements];
                        auto weightedX =
                            packedWeightedX_[headOffset * kTile];
                        auto packedW = packedW_[slot * kTileElements];
                        vector_.PrepareHead(
                            x,
                            dAGm_[static_cast<uint64_t>(headTask) * kTile],
                            weightedX, 1U, 4U * kTile);
                        vector_.BuildWeights(cbWorkspace_, packedW);
                        SetHeadReady(slot);
                    }
                    WaitHeadDone((headCount - 2) & 1U);
                    WaitHeadDone((headCount - 1) & 1U);
                    CrossCoreWaitFlag<0x2>(kGroupedStateReady);
                    for (uint32_t headOffset = 0; headOffset < headCount;
                         ++headOffset) {
                        const uint32_t h = headBegin + headOffset;
                        const uint32_t headTask =
                            (b * tiling_.heads + h) * tiling_.chunks + k;
                        auto state = stateGm_[
                            static_cast<uint64_t>(headTask) * kTileElements];
                        vector_.ScatterPackedState(
                            packedState_[headOffset * kTile], state,
                            4U * kTile);
                    }
#endif
                }
                else {
                    for (uint32_t headOffset = 0; headOffset < headCount;
                         ++headOffset) {
                        const uint32_t slot = headOffset & 1U;
                        if (headOffset >= 2) {
                            WaitHeadDone(slot);
                        }
                        const uint32_t h = headBegin + headOffset;
                        const uint32_t headTask =
                            (b * tiling_.heads + h) * tiling_.chunks + k;
                        auto weightedX =
                            packedWeightedX_[slot * kTileElements];
                        auto packedW = packedW_[slot * kTileElements];
                        vector_.PrepareHead(
                            xGm_[static_cast<uint64_t>(headTask) *
                                 kTileElements],
                            dAGm_[static_cast<uint64_t>(headTask) * kTile],
                            weightedX, 1U);
                        vector_.BuildWeights(
                            cbWorkspace_, packedW);
                        SetHeadReady(slot);
                    }
                    if (headCount == 1) {
                        WaitHeadDone(0);
                    } else if (headCount >= 2) {
                        WaitHeadDone((headCount - 2) & 1U);
                        WaitHeadDone((headCount - 1) & 1U);
                    }
                }
            }

            if ASCEND_IS_AIC {
                matmuls_.Compute64(cMat, bT, cbWorkspace_);
                CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeCbReady);
                if (groupedState) {
#ifdef MAMBA2_CHUNK_MIX_GROUPED_OUTPUT
                    const uint64_t groupMatrixBase =
                        static_cast<uint64_t>(bcIndex) * 4U * kTileElements;
                    if (UseBatchDiag) {
                        const uint32_t headTask =
                            (b * tiling_.heads + headBegin) *
                                tiling_.chunks + k;
                        auto x = xGm_[static_cast<uint64_t>(headTask) *
                                      kTileElements];
                        auto y = yGm_[static_cast<uint64_t>(headTask) *
                                      kTileElements];
                        WaitHeadReady(0);
                        const uint32_t headMatrixStride =
                            tiling_.chunks * kTileElements;
                        matmuls_.ComputeBatchDiag(
                            packedW_, x, y,
                            headMatrixStride, headMatrixStride);
                        SetHeadDone(0);
                    } else {
                        for (uint32_t headOffset = 0; headOffset < headCount;
                             ++headOffset) {
                            const uint32_t slot = headOffset & 1U;
                            const uint32_t h = headBegin + headOffset;
                            const uint32_t headTask =
                                (b * tiling_.heads + h) * tiling_.chunks + k;
                            auto x = groupedInput_
                                ? xGm_[groupMatrixBase + headOffset * kTile]
                                : xGm_[static_cast<uint64_t>(headTask) *
                                       kTileElements];
                            auto y = groupedInput_
                                ? yGm_[groupMatrixBase + headOffset * kTile]
                                : yGm_[static_cast<uint64_t>(headTask) *
                                       kTileElements];
                            WaitHeadReady(slot);
                            if (groupedInput_) {
                                matmuls_.Compute64StridedNoEnd(
                                    packedW_[slot * kTileElements], x, y);
                            } else {
                                matmuls_.Compute64NoEnd(
                                    packedW_[slot * kTileElements], x, y);
                            }
                            SetHeadDone(slot);
                        }
                        matmuls_.End64();
                    }
                    matmuls_.ComputeWide(
                        bT, packedWeightedX_,
                        stateGm_[static_cast<uint64_t>(bcIndex) *
                                 4U * kTileElements]);
#else
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
                        WaitHeadReady(slot);
                        matmuls_.Compute64(
                            packedW_[slot * kTileElements], x, y);
                        SetHeadDone(slot);
                    }
                    matmuls_.ComputeWide(
                        bT, packedWeightedX_, packedState_);
#endif
                    // Keep the PIPE_FIX completion handshake even when the
                    // wide GEMM writes directly to the group-owned output.
                    // Besides pairing the AIC/AIV event counter, this prevents
                    // the next task from reusing packedWeightedX_ before the
                    // current Fixpipe store has consumed it.  The grouped path
                    // still avoids the expensive four-head GM scatter.
                    CrossCoreSetFlag<0x2, PIPE_FIX>(kGroupedStateReady);
                }
                else {
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
                        auto state = stateGm_[
                            static_cast<uint64_t>(headTask) * kTileElements];
                        WaitHeadReady(slot);
                        matmuls_.Compute64(
                            packedW_[slot * kTileElements], x, y);
                        matmuls_.Compute64(
                            bT, packedWeightedX_[slot * kTileElements],
                            state);
                        SetHeadDone(slot);
                    }
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

    __aicore__ inline void DecodeTask(
        uint32_t task, uint32_t &b, uint32_t &k, uint32_t &group,
        uint32_t &headBegin, uint32_t &headCount) const
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

    Arch35ChunkMixMatmuls matmuls_;
    VectorChunkKernel<kTile> vector_;
    GlobalTensor<half> xGm_;
    GlobalTensor<float> dAGm_;
    GlobalTensor<half> bGm_;
    GlobalTensor<half> cGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> stateGm_;
    GlobalTensor<half> packedWeightedX_;
    GlobalTensor<half> packedW_;
    GlobalTensor<float> packedState_;
    GlobalTensor<float> cbWorkspace_;
    Mamba2SsdChunkMixTilingData tiling_;
    uint32_t coreIdx_ = 0;
    bool groupedInput_ = false;
};

template <uint32_t MatmulK, uint32_t MaxN = 128,
          bool GroupedStateOutput = false>
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

        groupedWideState_ = GroupedStateOutput && tiling_.taskMode == 1 &&
            tiling_.headsPerGroup == 4 && tiling_.stateDim == kTile;
        legacyBHalfElements_ = tiling_.stateNpLayout == 0
            ? kTile * tiling_.stateDim : 0;
        weightedXtHalfOffset_[0] = legacyBHalfElements_;
        if (groupedWideState_) {
            // The four heads are columns of one row-major [T, 4P] matrix.
            // Their bases are therefore 0, P, 2P and 3P.  Advancing by a
            // complete T*P tile would overlap later rows and overrun the
            // packed weighted-X workspace.
            weightedXtHalfOffset_[1] =
                weightedXtHalfOffset_[0] + kTile;
            weightedXtHalfOffset_[2] =
                weightedXtHalfOffset_[0] + 2U * kTile;
            weightedXtHalfOffset_[3] =
                weightedXtHalfOffset_[0] + 3U * kTile;
        } else {
            weightedXtHalfOffset_[1] =
                weightedXtHalfOffset_[0] + kTileElements;
            weightedXtHalfOffset_[2] =
                weightedXtHalfOffset_[1] + kTileElements;
            weightedXtHalfOffset_[3] =
                weightedXtHalfOffset_[2] + kTileElements;
        }
        const uint32_t weightedSlots = groupedWideState_ ? 4U : 2U;
        wHalfOffset_[0] = weightedXtHalfOffset_[0] +
            weightedSlots * kTileElements;
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
                for (uint32_t headOffset = 0; headOffset < headCount;
                     ++headOffset) {
                    const uint32_t slot = headOffset & 1U;
                    if (headOffset >= 2) {
                        WaitHeadDone(slot);
                    }
                    const uint32_t h = headBegin + headOffset;
                    const uint32_t headTask =
                        (b * tiling_.heads + h) * tiling_.chunks + k;
                    const uint32_t weightedSlot = groupedWideState_
                        ? headOffset : slot;
                    auto weightedXT =
                        workspaceHalf_[weightedXtHalfOffset_[weightedSlot]];
                    auto w = workspaceHalf_[wHalfOffset_[slot]];
                    vector_.PrepareHead(
                        xGm_[static_cast<uint64_t>(headTask) * kTileElements],
                        dAGm_[static_cast<uint64_t>(headTask) * kTile],
                        weightedXT, tiling_.stateNpLayout,
                        groupedWideState_ ? 4U * kTile : kTile);
                    // weighted-X depends only on x/dA.  Prepare head 0 while
                    // Cube computes the shared C@B tile, and rendezvous only
                    // before BuildWeights consumes that tile.  Later heads
                    // already overlap with the preceding diagonal GEMM.
                    if (headOffset == 0) {
                        CrossCoreWaitFlag<0x2>(kCubeCbReady);
                    }
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
                if (groupedWideState_) {
                    CrossCoreWaitFlag<0x2>(kGroupedStateReady);
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
                    const uint32_t weightedSlot = groupedWideState_
                        ? headOffset : slot;
                    auto weightedXT =
                        workspaceHalf_[weightedXtHalfOffset_[weightedSlot]];
                    auto w = workspaceHalf_[wHalfOffset_[slot]];
                    WaitHeadReady(slot);
                    matmul_.ComputeBlock(w, x, y,
                                         kTile, kTile, kTile, kTile);
                    if (!groupedWideState_ && tiling_.stateNpLayout != 0) {
                        // B^T[N,T] @ weighted-X[T,P] writes Cube-native
                        // [N,P].  N=128 is two independent 64-row blocks.
                        for (uint32_t n = 0; n < tiling_.stateDim;
                             n += kTile) {
                            matmul_.ComputeBlock(
                                bT[n * kTile], weightedXT,
                                state[n * kTile], kTile, kTile, kTile,
                                kTile);
                        }
                    } else if (!groupedWideState_) {
                        // Legacy [P,T] @ [T,N] -> [P,N] for the unfused
                        // state_passing fallback.
                        matmul_.ComputeBlock(
                            weightedXT, legacyB, state,
                            kTile, kTile, tiling_.stateDim, tiling_.stateDim,
                            tiling_.stateDim);
                    }
                    SetHeadDone(slot);
                }
                if (groupedWideState_) {
                    const uint64_t groupedStateOffset =
                        static_cast<uint64_t>(bcIndex) * kTile * 4U * kTile;
                    matmul_.ComputeBlock(
                        bT, workspaceHalf_[weightedXtHalfOffset_[0]],
                        stateGm_[groupedStateOffset], kTile, kTile,
                        4U * kTile, 4U * kTile, 4U * kTile);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(kGroupedStateReady);
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

    Mamba2DynamicMatmul<half, float, MatmulK, MaxN> matmul_;
    VectorChunkKernel<kTile / 2> vector_;
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
    uint32_t weightedXtHalfOffset_[4] = {0, 0, 0, 0};
    uint32_t wHalfOffset_[2] = {0, 0};
    uint32_t cbByteOffset_ = 0;
    bool groupedWideState_ = false;
};
#endif

} // namespace

#ifndef MAMBA2_CHUNK_MIX_COMPONENT_ONLY
#ifdef MAMBA2_CHUNK_MIX_GROUPED_OUTPUT
extern "C" __global__ __aicore__ void mamba2_ssd_chunk_mix_grouped(
#else
extern "C" __global__ __aicore__ void mamba2_ssd_chunk_mix(
#endif
    GM_ADDR x_cube, GM_ADDR d_a_cumsum, GM_ADDR b_cube, GM_ADDR c_cube,
    GM_ADDR y_diag, GM_ADDR chunk_states, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(9, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(10, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(11, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(9)) {
        KernelMamba2SsdChunkMixArch35<> op;
        op.Init(x_cube, d_a_cumsum, b_cube, c_cube, y_diag, chunk_states,
                workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        KernelMamba2SsdChunkMixArch35<true> op;
        op.Init(x_cube, d_a_cumsum, b_cube, c_cube, y_diag, chunk_states,
                workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        KernelMamba2SsdChunkMix<64, 256, true> op;
        op.Init(x_cube, d_a_cumsum, b_cube, c_cube, y_diag, chunk_states,
                workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
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
