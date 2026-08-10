/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * T=P=N=64 MIX kernel for the diagonal scan and chunk-state backward paths.
 * Cube owns the seven dense 64x64 GEMMs; the paired AIVs own decay/masking,
 * fp16 staging/transposes, output fusion and dA-cumsum reductions.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/pad/broadcast.h"
#include "mamba2_dynamic_matmul.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 64;
constexpr uint32_t kPackedHeads = 4;
constexpr uint32_t kPackedWidth = kPackedHeads * kTile;
constexpr uint32_t kTileElements = kTile * kTile;
// CANN builds one binary per compute unit.  Ascend 950PR (NPU arch 3510;
// older CANN headers also expose __DAV_310R6__) uses a 1-AIC/1-AIV MIX
// group; 910B retains the original 1-AIC/2-AIV group.  Derive Vector row
// ownership from that topology so the math and workspace layout stay shared.
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
constexpr uint32_t kAivPerAic = 1;
#else
constexpr uint32_t kAivPerAic = 2;
#endif
static_assert(kAivPerAic == 1 || kAivPerAic == 2,
              "unsupported MIX Vector/Cube ratio");
static_assert(kTile % kAivPerAic == 0,
              "Vector row ownership must cover a full tile");
constexpr uint32_t kRowsPerAiv = kTile / kAivPerAic;
constexpr uint32_t kBlockElements = kRowsPerAiv * kTile;
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements =
    kTransposeTile * kTransposeTile;

// Packed-four-head mode retains per-head W/dW/dCb/R/U until the group-level
// Cube reductions.  The larger per-core GM workspace removes per-head
// dB/dC Fixpipe plus GM read-add-write from the hot path.
constexpr uint32_t kHalfMatrices = 31;
constexpr uint32_t kFloatMatrices = 9;
constexpr uint32_t kFloatWorkspaceByteOffset =
    kHalfMatrices * kTileElements * sizeof(half);

// CANN 8.2 on 910B exposes 0..15.  Keep custom cross-core stages in 8..14;
// the direct Mmad/DataCopy implementation and runtime reserve lower IDs.
constexpr uint16_t kPrepareReady = 0x8;
constexpr uint16_t kCbReady = 0x9;
constexpr uint16_t kWeightsReady = 0xA;
constexpr uint16_t kDwReady = 0xB;
constexpr uint16_t kDcbReady = 0xC;
constexpr uint16_t kFinalReady = 0xD;
constexpr uint16_t kVectorDone = 0xE;

enum HalfWorkspace : uint32_t {
    kGyHalf = 0,
    kXT = 1,
    kBTime = 2,
    kUHalf = 3,
    kUTHalf = 4,
    kW = 5,
    kWT = 6,
    kDcb = 7,
    kDcbT = 8,
    kR = 9,
};

enum FloatWorkspace : uint32_t {
    kCb = 0,
    kDw = 1,
    kDr = 2,
    kDbState = 3,
    kDbHead = 4,
    kDcHead = 5,
};

// Per-head matrix slots used only when headsPerGroup == 4.  Slots 0..2 keep
// the existing gy/x/B temporaries so the Vector implementation is shared.
constexpr uint32_t kPackedUBase = 3;
constexpr uint32_t kPackedUTBase = 7;
constexpr uint32_t kPackedWBase = 11;
constexpr uint32_t kPackedWTBase = 15;
constexpr uint32_t kPackedDcbBase = 19;
constexpr uint32_t kPackedDcbTBase = 23;
constexpr uint32_t kPackedRBase = 27;
constexpr uint32_t kPackedDwBase = 1;
constexpr uint32_t kPackedDrBase = 5;

class VectorDiagStateBwd {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(transposeIn_, 1,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(transposeOut_, 1,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(dAIn_, 1, kTile * sizeof(float));
        pipe->InitBuffer(floatBlockIn_, 1,
                         kBlockElements * sizeof(float));
        pipe->InitBuffer(halfBlockIn_, 1,
                         kBlockElements * sizeof(half));
        pipe->InitBuffer(halfBlockOut_, 1,
                         kBlockElements * sizeof(half));
        pipe->InitBuffer(fullFloatIn_, 1,
                         kTileElements * sizeof(float));
        pipe->InitBuffer(fullHalfIn_, 1,
                         kTileElements * sizeof(half));
        pipe->InitBuffer(gOut_, 1, kRowsPerAiv * sizeof(float));

        pipe->InitBuffer(floatBlock0_,
                         kBlockElements * sizeof(float));
        pipe->InitBuffer(floatBlock1_,
                         kBlockElements * sizeof(float));
        pipe->InitBuffer(halfDecayBlock_,
                         kBlockElements * sizeof(half));
        pipe->InitBuffer(decayHalf_, kTile * sizeof(half));
        pipe->InitBuffer(product_,
                         kTileElements * sizeof(float));
        pipe->InitBuffer(reduceSparse_,
                         kTile * 8 * sizeof(float));
        pipe->InitBuffer(compact0_, kTile * sizeof(float));
        pipe->InitBuffer(compact1_, kTile * sizeof(float));
        pipe->InitBuffer(causalMask_,
                         kBlockElements * sizeof(float));
        pipe->InitBuffer(broadcastTmp_,
                         2 * kBlockElements * sizeof(uint8_t));

        // The causal mask depends only on the fixed 64-token tile and this
        // AIV's owned row slab.  Build it once per kernel instead of issuing 64
        // Duplicate instructions for every head of every group task.
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        auto causalMask = causalMask_.Get<float>();
        for (uint32_t localRow = 0; localRow < kRowsPerAiv; ++localRow) {
            auto maskRow = causalMask[localRow * kTile];
            Duplicate(maskRow, 0.0f, kTile);
            PipeBarrier<PIPE_V>();
            Duplicate(maskRow, 1.0f, rowBegin + localRow + 1);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareShared(
        const GlobalTensor<half> &bCube,
        GlobalTensor<half> bTime)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        // B is group-shared and supplied as [N,T].  Materialize [T,N] once
        // for all heads owned by this group task.
        for (uint32_t row = rowBegin; row < rowBegin + kRowsPerAiv;
             row += kTransposeTile) {
            for (uint32_t col = 0; col < kTile;
                 col += kTransposeTile) {
                TransposeHalfTile(
                    bCube[row * kTile + col],
                    bTime[col * kTile + row]);
            }
        }
    }

    __aicore__ inline void PrepareHead(
        const GlobalTensor<float> &gy,
        const GlobalTensor<half> &x,
        const GlobalTensor<float> &dA,
        const GlobalTensor<float> &dChunkStates,
        GlobalTensor<half> gyHalf,
        GlobalTensor<half> xT,
        GlobalTensor<half> uHalf,
        GlobalTensor<half> uTHalf,
        GlobalTensor<half> r,
        uint32_t uTLeadingDimension = kTile)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;

        LoadDA(dA);

        auto gyLocal = floatBlockIn_.AllocTensor<float>();
        CopyIn(gyLocal, gy, rowBegin * kTile, kBlockElements);
        floatBlockIn_.EnQue(gyLocal);
        gyLocal = floatBlockIn_.DeQue<float>();
        auto gyHalfLocal = halfBlockOut_.AllocTensor<half>();
        Cast(gyHalfLocal, gyLocal, RoundMode::CAST_RINT,
             kBlockElements);
        floatBlockIn_.FreeTensor(gyLocal);
        halfBlockOut_.EnQue(gyHalfLocal);
        gyHalfLocal = halfBlockOut_.DeQue<half>();
        CopyOut(gyHalf, gyHalfLocal, rowBegin * kTile,
                kBlockElements);
        halfBlockOut_.FreeTensor(gyHalfLocal);

        auto xLocal = halfBlockIn_.AllocTensor<half>();
        CopyIn(xLocal, x, rowBegin * kTile, kBlockElements);
        halfBlockIn_.EnQue(xLocal);
        xLocal = halfBlockIn_.DeQue<half>();
        for (uint32_t localRow = 0; localRow < kRowsPerAiv;
             localRow += kTransposeTile) {
            for (uint32_t col = 0; col < kTile;
                 col += kTransposeTile) {
                TransposeLocalTile(
                    xLocal[localRow * kTile + col],
                    xT[col * kTile + rowBegin + localRow]);
            }
        }

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
        auto decayBlock = halfDecayBlock_.Get<half>();
        auto tmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t rowShape[2] = {kRowsPerAiv, 1U};
        const uint32_t blockShape[2] = {kRowsPerAiv, kTile};
        Broadcast<half, 2, 1>(decayBlock, decayHalf[rowBegin],
                              blockShape, rowShape, tmp);
        PipeBarrier<PIPE_V>();
        Mul(xLocal, xLocal, decayBlock, kBlockElements);
        PipeBarrier<PIPE_V>();
        // Materialize the Vector result through VECOUT before MTE3.  Direct
        // VECIN->GM copy can observe the pre-Mul payload on this toolchain.
        auto rLocal = halfBlockOut_.AllocTensor<half>();
        Muls(rLocal, xLocal, static_cast<half>(1.0f), kBlockElements);
        PipeBarrier<PIPE_V>();
        halfBlockIn_.FreeTensor(xLocal);
        halfBlockOut_.EnQue(rLocal);
        rLocal = halfBlockOut_.DeQue<half>();
        CopyOut(r, rLocal, rowBegin * kTile, kBlockElements);
        halfBlockOut_.FreeTensor(rLocal);

        auto uLocal = floatBlockIn_.AllocTensor<float>();
        CopyIn(uLocal, dChunkStates, rowBegin * kTile,
               kBlockElements);
        floatBlockIn_.EnQue(uLocal);
        uLocal = floatBlockIn_.DeQue<float>();
        auto uHalfLocal = halfBlockOut_.AllocTensor<half>();
        Cast(uHalfLocal, uLocal, RoundMode::CAST_RINT,
             kBlockElements);
        floatBlockIn_.FreeTensor(uLocal);
        halfBlockOut_.EnQue(uHalfLocal);
        uHalfLocal = halfBlockOut_.DeQue<half>();
        for (uint32_t localRow = 0; localRow < kRowsPerAiv;
             localRow += kTransposeTile) {
            for (uint32_t col = 0; col < kTile;
                 col += kTransposeTile) {
                TransposeLocalTile(
                    uHalfLocal[localRow * kTile + col],
                    uTHalf[col * uTLeadingDimension + rowBegin + localRow],
                    uTLeadingDimension);
            }
        }
        CopyOut(uHalf, uHalfLocal, rowBegin * kTile,
                kBlockElements);
        halfBlockOut_.FreeTensor(uHalfLocal);
    }

    // The baseline path keeps dA resident from PrepareHead through Finalize.
    // The four-head packed path cannot do that because dAIn_ intentionally has
    // one UB slot.  Release after BuildDcb and reload the 256-byte vector just
    // before Finalize; retaining four complete head workspaces is still much
    // more valuable than retaining these four tiny vectors.
    __aicore__ inline void LoadDA(const GlobalTensor<float> &dA)
    {
        dALocal_ = dAIn_.AllocTensor<float>();
        DataCopy(dALocal_, dA, kTile);
        dAIn_.EnQue(dALocal_);
        dALocal_ = dAIn_.DeQue<float>();
    }

    __aicore__ inline void ReleaseDA()
    {
        dAIn_.FreeTensor(dALocal_);
    }

    __aicore__ inline void BuildWeights(
        const GlobalTensor<float> &cb,
        GlobalTensor<half> w,
        GlobalTensor<half> wT)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        BuildCausalDecay(rowBegin);

        auto cbLocal = floatBlockIn_.AllocTensor<float>();
        CopyIn(cbLocal, cb, rowBegin * kTile, kBlockElements);
        floatBlockIn_.EnQue(cbLocal);
        cbLocal = floatBlockIn_.DeQue<float>();
        Mul(cbLocal, cbLocal, floatBlock0_.Get<float>(),
            kBlockElements);
        PipeBarrier<PIPE_V>();
        auto wLocal = halfBlockOut_.AllocTensor<half>();
        Cast(wLocal, cbLocal, RoundMode::CAST_RINT,
             kBlockElements);
        floatBlockIn_.FreeTensor(cbLocal);
        halfBlockOut_.EnQue(wLocal);
        wLocal = halfBlockOut_.DeQue<half>();
        CopyOut(w, wLocal, rowBegin * kTile, kBlockElements);
        for (uint32_t localRow = 0; localRow < kRowsPerAiv;
             localRow += kTransposeTile) {
            for (uint32_t col = 0; col < kTile;
                 col += kTransposeTile) {
                TransposeLocalTile(
                    wLocal[localRow * kTile + col],
                    wT[col * kTile + rowBegin + localRow]);
            }
        }
        halfBlockOut_.FreeTensor(wLocal);
    }

    __aicore__ inline void BuildDcb(
        const GlobalTensor<float> &dW,
        GlobalTensor<half> dCb,
        GlobalTensor<half> dCbT)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        // BuildWeights generated this head's causal decay in floatBlock0_.
        // The AIC dW phase does not touch AIV UB, so keep the matrix live
        // across the cross-core wait instead of repeating two broadcasts,
        // Sub, Exp and mask Mul for dCb.

        auto local = floatBlockIn_.AllocTensor<float>();
        CopyIn(local, dW, rowBegin * kTile, kBlockElements);
        floatBlockIn_.EnQue(local);
        local = floatBlockIn_.DeQue<float>();
        Mul(local, local, floatBlock0_.Get<float>(), kBlockElements);
        PipeBarrier<PIPE_V>();
        auto halfLocal = halfBlockOut_.AllocTensor<half>();
        Cast(halfLocal, local, RoundMode::CAST_RINT,
             kBlockElements);
        floatBlockIn_.FreeTensor(local);
        halfBlockOut_.EnQue(halfLocal);
        halfLocal = halfBlockOut_.DeQue<half>();
        CopyOut(dCb, halfLocal, rowBegin * kTile, kBlockElements);
        for (uint32_t localRow = 0; localRow < kRowsPerAiv;
             localRow += kTransposeTile) {
            for (uint32_t col = 0; col < kTile;
                 col += kTransposeTile) {
                TransposeLocalTile(
                    halfLocal[localRow * kTile + col],
                    dCbT[col * kTile + rowBegin + localRow]);
            }
        }
        halfBlockOut_.FreeTensor(halfLocal);
    }

    __aicore__ inline void Finalize(
        GlobalTensor<float> dX,
        GlobalTensor<float> dBGroup,
        GlobalTensor<float> dCGroup,
        GlobalTensor<float> gCs,
        const GlobalTensor<float> &dW,
        const GlobalTensor<half> &w,
        const GlobalTensor<float> &dR,
        const GlobalTensor<float> &dBState,
        const GlobalTensor<float> &dBHead,
        const GlobalTensor<float> &dCHead,
        const GlobalTensor<half> &r,
        uint32_t headInGroup,
        bool accumulateGroup = true,
        uint32_t dRLeadingDimension = kTile)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;

        // dX = dX_diag + dR * exp(cs[-1] - cs).
        auto xLocal = floatBlockIn_.AllocTensor<float>();
        CopyIn(xLocal, dX, rowBegin * kTile, kBlockElements);
        floatBlockIn_.EnQue(xLocal);
        xLocal = floatBlockIn_.DeQue<float>();
        auto xAccum = product_.Get<float>();
        // Use a Vector move rather than UB->UB DataCopy.  The latter runs on
        // a DMA pipe; freeing/reusing the one-slot input queue before that
        // DMA completed produced nondeterministic partial rows.
        Muls(xAccum, xLocal, 1.0f, kBlockElements);
        PipeBarrier<PIPE_V>();
        floatBlockIn_.FreeTensor(xLocal);

        auto decayVector = compact0_.Get<float>();
        const float lastDA = dALocal_.GetValue(kTile - 1);
        Muls(decayVector, dALocal_, -1.0f, kTile);
        PipeBarrier<PIPE_V>();
        Adds(decayVector, decayVector, lastDA, kTile);
        PipeBarrier<PIPE_V>();
        Exp(decayVector, decayVector, kTile);
        PipeBarrier<PIPE_V>();
        auto decayBlock = floatBlock1_.Get<float>();
        auto tmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t rowShape[2] = {kRowsPerAiv, 1U};
        const uint32_t blockShape[2] = {kRowsPerAiv, kTile};
        Broadcast<float, 2, 1>(decayBlock, decayVector[rowBegin],
                               blockShape, rowShape, tmp);
        PipeBarrier<PIPE_V>();

        auto drLocal = floatBlockIn_.AllocTensor<float>();
        CopyInMatrix(
            drLocal, dR, rowBegin, kRowsPerAiv, kTile,
            dRLeadingDimension);
        floatBlockIn_.EnQue(drLocal);
        drLocal = floatBlockIn_.DeQue<float>();
        Mul(drLocal, drLocal, decayBlock, kBlockElements);
        PipeBarrier<PIPE_V>();
        Add(drLocal, drLocal, xAccum, kBlockElements);
        PipeBarrier<PIPE_V>();
        CopyOut(dX, drLocal, rowBegin * kTile, kBlockElements);
        floatBlockIn_.FreeTensor(drLocal);

        if (accumulateGroup) {
        // Accumulate dB_diag + dB_state directly into the group output.
        auto dbLocal = floatBlockIn_.AllocTensor<float>();
        CopyIn(dbLocal, dBHead, rowBegin * kTile, kBlockElements);
        floatBlockIn_.EnQue(dbLocal);
        dbLocal = floatBlockIn_.DeQue<float>();
        Muls(xAccum, dbLocal, 1.0f, kBlockElements);
        PipeBarrier<PIPE_V>();
        floatBlockIn_.FreeTensor(dbLocal);
        auto dbStateLocal = floatBlockIn_.AllocTensor<float>();
        CopyIn(dbStateLocal, dBState, rowBegin * kTile,
               kBlockElements);
        floatBlockIn_.EnQue(dbStateLocal);
        dbStateLocal = floatBlockIn_.DeQue<float>();
        Add(dbStateLocal, dbStateLocal, xAccum, kBlockElements);
        PipeBarrier<PIPE_V>();
        if (headInGroup != 0) {
            Muls(xAccum, dbStateLocal, 1.0f, kBlockElements);
            PipeBarrier<PIPE_V>();
            floatBlockIn_.FreeTensor(dbStateLocal);
            auto groupLocal = floatBlockIn_.AllocTensor<float>();
            CopyIn(groupLocal, dBGroup, rowBegin * kTile,
                   kBlockElements);
            floatBlockIn_.EnQue(groupLocal);
            groupLocal = floatBlockIn_.DeQue<float>();
            Add(groupLocal, groupLocal, xAccum, kBlockElements);
            PipeBarrier<PIPE_V>();
            CopyOut(dBGroup, groupLocal, rowBegin * kTile,
                    kBlockElements);
            floatBlockIn_.FreeTensor(groupLocal);
        } else {
            CopyOut(dBGroup, dbStateLocal, rowBegin * kTile,
                    kBlockElements);
            floatBlockIn_.FreeTensor(dbStateLocal);
        }

        // dC has only the diagonal branch in this operator.  Accumulate it
        // by group here; the off-diagonal group contribution is added by the
        // public backward path.
        auto dcLocal = floatBlockIn_.AllocTensor<float>();
        CopyIn(dcLocal, dCHead, rowBegin * kTile, kBlockElements);
        floatBlockIn_.EnQue(dcLocal);
        dcLocal = floatBlockIn_.DeQue<float>();
        if (headInGroup != 0) {
            Muls(xAccum, dcLocal, 1.0f, kBlockElements);
            PipeBarrier<PIPE_V>();
            floatBlockIn_.FreeTensor(dcLocal);
            auto groupLocal = floatBlockIn_.AllocTensor<float>();
            CopyIn(groupLocal, dCGroup, rowBegin * kTile,
                   kBlockElements);
            floatBlockIn_.EnQue(groupLocal);
            groupLocal = floatBlockIn_.DeQue<float>();
            Add(groupLocal, groupLocal, xAccum, kBlockElements);
            PipeBarrier<PIPE_V>();
            CopyOut(dCGroup, groupLocal, rowBegin * kTile,
                    kBlockElements);
            floatBlockIn_.FreeTensor(groupLocal);
        } else {
            CopyOut(dCGroup, dcLocal, rowBegin * kTile,
                    kBlockElements);
            floatBlockIn_.FreeTensor(dcLocal);
        }

        }

        // g_diag[t] = row_sum(dW*W)[t] - col_sum(dW*W)[t].
        auto fullFloat = fullFloatIn_.AllocTensor<float>();
        CopyIn(fullFloat, dW, 0, kTileElements);
        fullFloatIn_.EnQue(fullFloat);
        fullFloat = fullFloatIn_.DeQue<float>();
        auto fullHalf = fullHalfIn_.AllocTensor<half>();
        CopyIn(fullHalf, w, 0, kTileElements);
        fullHalfIn_.EnQue(fullHalf);
        fullHalf = fullHalfIn_.DeQue<half>();
        auto product = product_.Get<float>();
        Cast(product, fullHalf, RoundMode::CAST_NONE, kTileElements);
        fullHalfIn_.FreeTensor(fullHalf);
        PipeBarrier<PIPE_V>();
        Mul(product, fullFloat, product, kTileElements);
        fullFloatIn_.FreeTensor(fullFloat);
        PipeBarrier<PIPE_V>();

        auto sparse = reduceSparse_.Get<float>();
        // One Vector repeat reduces one 64-element row.  Issue the complete
        // architecture-owned row slab in one instruction instead of a scalar
        // loop with one barrier per row.
        // WholeReduceSum writes one contiguous scalar per repeat on 910B;
        // using the sparse 32-byte offsets from the single-repeat form would
        // skip 7/8 of the results.
        auto diag = compact0_.Get<float>();
        WholeReduceSum<float, true>(
            diag, product[rowBegin * kTile],
            static_cast<int32_t>(kTile),
            static_cast<int32_t>(kRowsPerAiv), 1, 1,
            static_cast<int32_t>(kTile * sizeof(float) /
                                 DEFAULT_C0_SIZE));
        PipeBarrier<PIPE_V>();

        // Column sums are row-wise vector additions.  Six in-place tree
        // stages reduce 64 rows to product[0, :] without the previous 32
        // Gather + WholeReduceSum + barrier pairs on each AIV.
        for (uint32_t rows = kTile / 2; rows > 0; rows >>= 1) {
            Add(product, product, product[rows * kTile],
                rows * kTile);
            PipeBarrier<PIPE_V>();
        }
        Sub(diag, diag, product[rowBegin], kRowsPerAiv);
        PipeBarrier<PIPE_V>();

        // g_state[t] = -row_sum(dR*R)[t], with the complete sum added
        // back to the last cumsum element.  Every participating AIV reduces
        // all rows, so no atomic or AIV-to-AIV hand-off is required.
        fullFloat = fullFloatIn_.AllocTensor<float>();
        CopyInMatrix(
            fullFloat, dR, 0, kTile, kTile, dRLeadingDimension);
        fullFloatIn_.EnQue(fullFloat);
        fullFloat = fullFloatIn_.DeQue<float>();
        fullHalf = fullHalfIn_.AllocTensor<half>();
        CopyIn(fullHalf, r, 0, kTileElements);
        fullHalfIn_.EnQue(fullHalf);
        fullHalf = fullHalfIn_.DeQue<half>();
        Cast(product, fullHalf, RoundMode::CAST_NONE, kTileElements);
        fullHalfIn_.FreeTensor(fullHalf);
        PipeBarrier<PIPE_V>();
        Mul(product, fullFloat, product, kTileElements);
        fullFloatIn_.FreeTensor(fullFloat);
        PipeBarrier<PIPE_V>();
        auto stateRows = compact1_.Get<float>();
        WholeReduceSum<float, true>(
            stateRows, product, static_cast<int32_t>(kTile),
            static_cast<int32_t>(kTile), 1, 1,
            static_cast<int32_t>(kTile * sizeof(float) /
                                 DEFAULT_C0_SIZE));
        PipeBarrier<PIPE_V>();

        auto out = gOut_.AllocTensor<float>();
        Sub(out, diag, stateRows[rowBegin], kRowsPerAiv);
        PipeBarrier<PIPE_V>();
        if (rowBegin + kRowsPerAiv == kTile) {
            WholeReduceSum<float, true>(
                sparse[0], stateRows, static_cast<int32_t>(kTile),
                1, 1, 1,
                static_cast<int32_t>(kTile * sizeof(float) /
                                     DEFAULT_C0_SIZE));
            PipeBarrier<PIPE_V>();
            out.SetValue(kRowsPerAiv - 1,
                         out.GetValue(kRowsPerAiv - 1) +
                             sparse.GetValue(0));
        }
        gOut_.EnQue(out);
        out = gOut_.DeQue<float>();
        CopyOut(gCs, out, rowBegin, kRowsPerAiv);
        gOut_.FreeTensor(out);
        dAIn_.FreeTensor(dALocal_);
    }

private:
    __aicore__ inline void BuildCausalDecay(uint32_t rowBegin)
    {
        auto colDA = floatBlock0_.Get<float>();
        auto rowDA = floatBlock1_.Get<float>();
        auto tmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t colShape[2] = {1U, kTile};
        const uint32_t rowShape[2] = {kRowsPerAiv, 1U};
        const uint32_t blockShape[2] = {kRowsPerAiv, kTile};
        Broadcast<float, 2, 0>(colDA, dALocal_, blockShape,
                               colShape, tmp);
        PipeBarrier<PIPE_V>();
        Broadcast<float, 2, 1>(rowDA, dALocal_[rowBegin],
                               blockShape, rowShape, tmp);
        PipeBarrier<PIPE_V>();
        Sub(colDA, rowDA, colDA, kBlockElements);
        PipeBarrier<PIPE_V>();
        Exp(colDA, colDA, kBlockElements);
        PipeBarrier<PIPE_V>();
        Mul(colDA, colDA, causalMask_.Get<float>(), kBlockElements);
        PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void CopyIn(
        const LocalTensor<T> &dst, const GlobalTensor<T> &src,
        uint64_t offset, uint32_t elements)
    {
        DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], copy, pad);
    }

    template <typename T>
    __aicore__ inline void CopyOut(
        GlobalTensor<T> dst, const LocalTensor<T> &src,
        uint64_t offset, uint32_t elements)
    {
        // CopyOut is also used for tensors produced in VECCALC/VECIN.  A
        // PIPE_V barrier only orders Vector instructions; it does not make
        // the result visible to MTE3.  Without this hard event, MTE3 can
        // write the previous contents of the reused UB region.
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copy);
        // The caller may immediately reuse the same UB queue through either
        // Vector or MTE2.  Complete the store before that reuse.
        event_t eventIdToV = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdToV);
        event_t eventIdToMte2 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdToMte2);
    }

    template <typename T>
    __aicore__ inline void CopyInMatrix(
        const LocalTensor<T> &dst, const GlobalTensor<T> &src,
        uint32_t rowBegin, uint32_t rows, uint32_t cols,
        uint32_t leadingDimension)
    {
        DataCopyExtParams copy{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((leadingDimension - cols) * sizeof(T)),
            0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[rowBegin * leadingDimension], copy, pad);
    }

    __aicore__ inline void TransposeLocalTile(
        const LocalTensor<half> &src, GlobalTensor<half> dst,
        uint32_t dstLeadingDimension = kTile)
    {
        auto packed = transposeIn_.AllocTensor<half>();
        DataCopyParams load{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((kTile - kTransposeTile) *
                                  sizeof(half) / DEFAULT_C0_SIZE),
            0};
        DataCopy(packed, src, load);
        transposeIn_.EnQue(packed);
        packed = transposeIn_.DeQue<half>();
        auto transposed = transposeOut_.AllocTensor<half>();
        Transpose(transposed, packed);
        transposeIn_.FreeTensor(packed);
        transposeOut_.EnQue(transposed);
        transposed = transposeOut_.DeQue<half>();
        DataCopyExtParams store{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint32_t>(kTransposeTile * sizeof(half)),
            0,
            static_cast<uint32_t>((dstLeadingDimension - kTransposeTile) *
                                  sizeof(half)),
            0};
        DataCopyPad(dst, transposed, store);
        transposeOut_.FreeTensor(transposed);
    }

    __aicore__ inline void TransposeHalfTile(
        const GlobalTensor<half> &src, GlobalTensor<half> dst)
    {
        auto input = transposeIn_.AllocTensor<half>();
        DataCopyParams load{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((kTile - kTransposeTile) *
                                  sizeof(half) / DEFAULT_C0_SIZE),
            0};
        DataCopy(input, src, load);
        transposeIn_.EnQue(input);
        input = transposeIn_.DeQue<half>();
        auto output = transposeOut_.AllocTensor<half>();
        Transpose(output, input);
        transposeIn_.FreeTensor(input);
        transposeOut_.EnQue(output);
        output = transposeOut_.DeQue<half>();
        DataCopyParams store{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((kTile - kTransposeTile) *
                                  sizeof(half) / DEFAULT_C0_SIZE)};
        DataCopy(dst, output, store);
        transposeOut_.FreeTensor(output);
    }

    TQue<TPosition::VECIN, 1> transposeIn_;
    TQue<TPosition::VECOUT, 1> transposeOut_;
    TQue<TPosition::VECIN, 1> dAIn_;
    TQue<TPosition::VECIN, 1> floatBlockIn_;
    TQue<TPosition::VECIN, 1> halfBlockIn_;
    TQue<TPosition::VECOUT, 1> halfBlockOut_;
    TQue<TPosition::VECIN, 1> fullFloatIn_;
    TQue<TPosition::VECIN, 1> fullHalfIn_;
    TQue<TPosition::VECOUT, 1> gOut_;
    TBuf<TPosition::VECCALC> floatBlock0_;
    TBuf<TPosition::VECCALC> floatBlock1_;
    TBuf<TPosition::VECCALC> halfDecayBlock_;
    TBuf<TPosition::VECCALC> decayHalf_;
    TBuf<TPosition::VECCALC> product_;
    TBuf<TPosition::VECCALC> reduceSparse_;
    TBuf<TPosition::VECCALC> compact0_;
    TBuf<TPosition::VECCALC> compact1_;
    TBuf<TPosition::VECCALC> causalMask_;
    TBuf<TPosition::VECCALC> broadcastTmp_;
    LocalTensor<float> dALocal_;
};

class KernelMamba2SsdChunkScanBwdDiagState {
public:
    __aicore__ inline void Init(
        GM_ADDR gy, GM_ADDR x, GM_ADDR dA, GM_ADDR b, GM_ADDR c,
        GM_ADDR dChunkStates, GM_ADDR dX, GM_ADDR dB, GM_ADDR dC,
        GM_ADDR gCs, GM_ADDR workspace,
        const Mamba2SsdChunkScanBwdDiagStateTilingData &tiling,
        TPipe *pipe)
    {
        tiling_ = tiling;
        const uint64_t headTasks =
            static_cast<uint64_t>(tiling_.batch) * tiling_.heads *
            tiling_.chunks;
        const uint64_t groupTasks = tiling_.taskCount;
        gyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gy),
                              headTasks * kTileElements);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x),
                             headTasks * kTileElements);
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA),
                              headTasks * kTile);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b),
                             groupTasks * kTileElements);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c),
                             groupTasks * kTileElements);
        dChunkGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dChunkStates),
            headTasks * kTileElements);
        dXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dX),
                              headTasks * kTileElements);
        dBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dB),
                              groupTasks * kTileElements);
        dCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dC),
                              groupTasks * kTileElements);
        gCsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gCs),
                               headTasks * kTile);

        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) *
                tiling_.workspaceBytesPerCore;
        halfWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(coreWorkspace),
            kHalfMatrices * kTileElements);
        floatWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(
                coreWorkspace + kFloatWorkspaceByteOffset),
            kFloatMatrices * kTileElements);
        if ASCEND_IS_AIC {
            matmul_.Init(*pipe);
        }
        if ASCEND_IS_AIV {
            vector_.Init(pipe);
        }
    }

    __aicore__ inline void Process()
    {
        if (tiling_.headsPerGroup == 4) {
            ProcessPackedFourHeads();
            return;
        }
        for (uint32_t groupTask = coreIdx_; groupTask < tiling_.taskCount;
             groupTask += tiling_.usedCoreNum) {
            const uint32_t group = groupTask % tiling_.groups;
            const uint32_t chunk =
                (groupTask / tiling_.groups) % tiling_.chunks;
            const uint32_t batch =
                groupTask / (tiling_.groups * tiling_.chunks);
            const uint64_t groupOffset =
                static_cast<uint64_t>(groupTask) * kTileElements;

            auto gyHalf = halfWorkspace_[kGyHalf * kTileElements];
            auto xT = halfWorkspace_[kXT * kTileElements];
            auto bTime = halfWorkspace_[kBTime * kTileElements];
            auto uHalf = halfWorkspace_[kUHalf * kTileElements];
            auto uT = halfWorkspace_[kUTHalf * kTileElements];
            auto w = halfWorkspace_[kW * kTileElements];
            auto wT = halfWorkspace_[kWT * kTileElements];
            auto dCb = halfWorkspace_[kDcb * kTileElements];
            auto dCbT = halfWorkspace_[kDcbT * kTileElements];
            auto r = halfWorkspace_[kR * kTileElements];
            auto cb = floatWorkspace_[kCb * kTileElements];
            auto dW = floatWorkspace_[kDw * kTileElements];
            auto dR = floatWorkspace_[kDr * kTileElements];
            auto dBState = floatWorkspace_[kDbState * kTileElements];
            auto dBHead = floatWorkspace_[kDbHead * kTileElements];
            auto dCHead = floatWorkspace_[kDcHead * kTileElements];

            // Both B^T staging and C@B are group-shared.  Keep them in the
            // per-core workspace while streaming all heads in the group.
            if ASCEND_IS_AIV {
                vector_.PrepareShared(bGm_[groupOffset], bTime);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(kPrepareReady);
            }
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<0x2>(kPrepareReady);
                matmul_.ComputeBlock(
                    cGm_[groupOffset], bGm_[groupOffset], cb,
                    kTile, kTile, kTile, kTile);
                CrossCoreSetFlag<0x2, PIPE_FIX>(kCbReady);
            }

            for (uint32_t headInGroup = 0;
                 headInGroup < tiling_.headsPerGroup; ++headInGroup) {
                const uint32_t head =
                    group * tiling_.headsPerGroup + headInGroup;
                const uint32_t headTask =
                    (batch * tiling_.heads + head) * tiling_.chunks + chunk;
                const uint64_t matrixOffset =
                    static_cast<uint64_t>(headTask) * kTileElements;
                const uint64_t vectorOffset =
                    static_cast<uint64_t>(headTask) * kTile;

                if ASCEND_IS_AIV {
                    vector_.PrepareHead(
                        gyGm_[matrixOffset], xGm_[matrixOffset],
                        dAGm_[vectorOffset], dChunkGm_[matrixOffset],
                        gyHalf, xT, uHalf, uT, r);
                    if (headInGroup == 0) {
                        CrossCoreWaitFlag<0x2>(kCbReady);
                    }
                    vector_.BuildWeights(cb, w, wT);
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(kWeightsReady);
                    CrossCoreWaitFlag<0x2>(kDwReady);
                    vector_.BuildDcb(dW, dCb, dCbT);
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(kDcbReady);
                    CrossCoreWaitFlag<0x2>(kFinalReady);
                    vector_.Finalize(
                        dXGm_[matrixOffset], dBGm_[groupOffset],
                        dCGm_[groupOffset], gCsGm_[vectorOffset],
                        dW, w, dR, dBState, dBHead, dCHead, r,
                        headInGroup);
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(kVectorDone);
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<0x2>(kWeightsReady);
                    matmul_.ComputeBlock(
                        wT, gyHalf, dXGm_[matrixOffset],
                        kTile, kTile, kTile, kTile);
                    matmul_.ComputeBlock(
                        gyHalf, xT, dW, kTile, kTile, kTile, kTile);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(kDwReady);
                    CrossCoreWaitFlag<0x2>(kDcbReady);
                    matmul_.ComputeBlock(
                        dCb, bTime, dCHead,
                        kTile, kTile, kTile, kTile);
                    matmul_.ComputeBlock(
                        dCbT, cGm_[groupOffset], dBHead,
                        kTile, kTile, kTile, kTile);
                    matmul_.ComputeBlock(
                        bTime, uT, dR, kTile, kTile, kTile, kTile);
                    matmul_.ComputeBlock(
                        r, uHalf, dBState, kTile, kTile, kTile, kTile);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(kFinalReady);
                    CrossCoreWaitFlag<0x2>(kVectorDone);
                }
            }
        }
    }

private:
    __aicore__ inline void ProcessPackedFourHeads()
    {
        for (uint32_t groupTask = coreIdx_; groupTask < tiling_.taskCount;
             groupTask += tiling_.usedCoreNum) {
            const uint32_t group = groupTask % tiling_.groups;
            const uint32_t chunk =
                (groupTask / tiling_.groups) % tiling_.chunks;
            const uint32_t batch =
                groupTask / (tiling_.groups * tiling_.chunks);
            const uint64_t groupOffset =
                static_cast<uint64_t>(groupTask) * kTileElements;

            auto gyHalf = halfWorkspace_[kGyHalf * kTileElements];
            auto xT = halfWorkspace_[kXT * kTileElements];
            auto bTime = halfWorkspace_[kBTime * kTileElements];
            auto cb = floatWorkspace_[kCb * kTileElements];

            if ASCEND_IS_AIV {
                vector_.PrepareShared(bGm_[groupOffset], bTime);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(kPrepareReady);
            }
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<0x2>(kPrepareReady);
                matmul_.ComputeBlock(
                    cGm_[groupOffset], bGm_[groupOffset], cb,
                    kTile, kTile, kTile, kTile);
                CrossCoreSetFlag<0x2, PIPE_FIX>(kCbReady);
            }

            // Only the two head-specific GEMMs remain in the serial head
            // stage.  dCb/R/U are retained for group-level L0C accumulation.
            for (uint32_t headInGroup = 0; headInGroup < 4;
                 ++headInGroup) {
                const uint32_t head = group * 4 + headInGroup;
                const uint32_t headTask =
                    (batch * tiling_.heads + head) * tiling_.chunks + chunk;
                const uint64_t matrixOffset =
                    static_cast<uint64_t>(headTask) * kTileElements;
                const uint64_t vectorOffset =
                    static_cast<uint64_t>(headTask) * kTile;
                auto uHalf = halfWorkspace_[
                    (kPackedUBase + headInGroup) * kTileElements];
                // UT is row-interleaved as one Kx(4N) matrix so Cube can
                // produce all four dR heads with one wide GEMM/Fixpipe.
                auto uT = halfWorkspace_[
                    kPackedUTBase * kTileElements + headInGroup * kTile];
                auto w = halfWorkspace_[
                    (kPackedWBase + headInGroup) * kTileElements];
                auto wT = halfWorkspace_[
                    (kPackedWTBase + headInGroup) * kTileElements];
                auto dCb = halfWorkspace_[
                    (kPackedDcbBase + headInGroup) * kTileElements];
                auto dCbT = halfWorkspace_[
                    (kPackedDcbTBase + headInGroup) * kTileElements];
                auto r = halfWorkspace_[
                    (kPackedRBase + headInGroup) * kTileElements];
                auto dW = floatWorkspace_[
                    (kPackedDwBase + headInGroup) * kTileElements];

                if ASCEND_IS_AIV {
                    vector_.PrepareHead(
                        gyGm_[matrixOffset], xGm_[matrixOffset],
                        dAGm_[vectorOffset], dChunkGm_[matrixOffset],
                        gyHalf, xT, uHalf, uT, r, kPackedWidth);
                    if (headInGroup == 0) {
                        CrossCoreWaitFlag<0x2>(kCbReady);
                    }
                    vector_.BuildWeights(cb, w, wT);
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(kWeightsReady);
                    CrossCoreWaitFlag<0x2>(kDwReady);
                    vector_.BuildDcb(dW, dCb, dCbT);
                    vector_.ReleaseDA();
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<0x2>(kWeightsReady);
                    matmul_.ComputeBlock(
                        wT, gyHalf, dXGm_[matrixOffset],
                        kTile, kTile, kTile, kTile);
                    matmul_.ComputeBlock(
                        gyHalf, xT, dW,
                        kTile, kTile, kTile, kTile);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(kDwReady);
                }
            }

            if ASCEND_IS_AIV {
                // Every per-head dCb store has completed before publishing
                // the group inputs to Cube.
                CrossCoreSetFlag<0x2, PIPE_MTE3>(kDcbReady);
                CrossCoreWaitFlag<0x2>(kFinalReady);
                for (uint32_t headInGroup = 0; headInGroup < 4;
                     ++headInGroup) {
                    const uint32_t head = group * 4 + headInGroup;
                    const uint32_t headTask =
                        (batch * tiling_.heads + head) * tiling_.chunks +
                        chunk;
                    const uint64_t matrixOffset =
                        static_cast<uint64_t>(headTask) * kTileElements;
                    const uint64_t vectorOffset =
                        static_cast<uint64_t>(headTask) * kTile;
                    vector_.LoadDA(dAGm_[vectorOffset]);
                    vector_.Finalize(
                        dXGm_[matrixOffset], dBGm_[groupOffset],
                        dCGm_[groupOffset], gCsGm_[vectorOffset],
                        floatWorkspace_[
                            (kPackedDwBase + headInGroup) * kTileElements],
                        halfWorkspace_[
                            (kPackedWBase + headInGroup) * kTileElements],
                        floatWorkspace_[
                            kPackedDrBase * kTileElements +
                            headInGroup * kTile],
                        floatWorkspace_[kDbState * kTileElements],
                        floatWorkspace_[kDbHead * kTileElements],
                        floatWorkspace_[kDcHead * kTileElements],
                        halfWorkspace_[
                            (kPackedRBase + headInGroup) * kTileElements],
                        headInGroup, false, kPackedWidth);
                }
                CrossCoreSetFlag<0x2, PIPE_MTE3>(kVectorDone);
                // Do not let Vector overwrite the packed workspace for the
                // next group until Cube has consumed every retained matrix.
                CrossCoreWaitFlag<0x2>(kCbReady);
            }

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<0x2>(kDcbReady);
                // B @ [U0^T U1^T U2^T U3^T] produces every dR with one
                // 64x64x256 Cube dispatch and one Fixpipe.
                matmul_.ComputeBlock(
                    bTime, halfWorkspace_[kPackedUTBase * kTileElements],
                    floatWorkspace_[kPackedDrBase * kTileElements],
                    kTile, kTile, kPackedWidth, kPackedWidth,
                    kPackedWidth);
                CrossCoreSetFlag<0x2, PIPE_FIX>(kFinalReady);

                // dB = sum_h(dCb_h^T @ C + R_h @ U_h).  Retain FP32 L0C
                // across all eight contributions and perform one Fixpipe.
                auto dBAccumulator = matmul_.AllocAccumulator();
                for (uint32_t headInGroup = 0; headInGroup < 4;
                     ++headInGroup) {
                    matmul_.AccumulateBlock(
                        dBAccumulator,
                        halfWorkspace_[
                            (kPackedDcbTBase + headInGroup) * kTileElements],
                        cGm_[groupOffset], kTile, kTile, kTile,
                        headInGroup == 0, kTile);
                    matmul_.AccumulateBlock(
                        dBAccumulator,
                        halfWorkspace_[
                            (kPackedRBase + headInGroup) * kTileElements],
                        halfWorkspace_[
                            (kPackedUBase + headInGroup) * kTileElements],
                        kTile, kTile, kTile, false, kTile);
                }
                matmul_.StoreAccumulator(
                    dBAccumulator, dBGm_[groupOffset], kTile, kTile);

                // dC = sum_h(dCb_h @ B).  B^T was materialized once for the
                // complete group task.
                auto dCAccumulator = matmul_.AllocAccumulator();
                for (uint32_t headInGroup = 0; headInGroup < 4;
                     ++headInGroup) {
                    matmul_.AccumulateBlock(
                        dCAccumulator,
                        halfWorkspace_[
                            (kPackedDcbBase + headInGroup) * kTileElements],
                        bTime, kTile, kTile, kTile,
                        headInGroup == 0, kTile);
                }
                matmul_.StoreAccumulator(
                    dCAccumulator, dCGm_[groupOffset], kTile, kTile);
                CrossCoreWaitFlag<0x2>(kVectorDone);
                CrossCoreSetFlag<0x2, PIPE_FIX>(kCbReady);
            }
        }
    }

    Mamba2DynamicMatmul<half, float, 64, kPackedWidth> matmul_;
    VectorDiagStateBwd vector_;
    GlobalTensor<float> gyGm_;
    GlobalTensor<half> xGm_;
    GlobalTensor<float> dAGm_;
    GlobalTensor<half> bGm_;
    GlobalTensor<half> cGm_;
    GlobalTensor<float> dChunkGm_;
    GlobalTensor<float> dXGm_;
    GlobalTensor<float> dBGm_;
    GlobalTensor<float> dCGm_;
    GlobalTensor<float> gCsGm_;
    GlobalTensor<half> halfWorkspace_;
    GlobalTensor<float> floatWorkspace_;
    Mamba2SsdChunkScanBwdDiagStateTilingData tiling_;
    uint32_t coreIdx_ = 0;
};
} // namespace

extern "C" __global__ __aicore__ void
mamba2_ssd_chunk_scan_bwd_diag_state(
    GM_ADDR gy, GM_ADDR x_cube, GM_ADDR d_a_cumsum,
    GM_ADDR b_cube, GM_ADDR c_cube, GM_ADDR d_chunk_states,
    GM_ADDR d_xdt, GM_ADDR d_b_group, GM_ADDR d_c_diag_group,
    GM_ADDR g_d_a_cs_diag_state, GM_ADDR workspace, GM_ADDR tiling)
{
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelMamba2SsdChunkScanBwdDiagState op;
        op.Init(gy, x_cube, d_a_cumsum, b_cube, c_cube,
                d_chunk_states, d_xdt, d_b_group, d_c_diag_group,
                g_d_a_cs_diag_state, workspace, tilingData, &pipe);
        op.Process();
    }
}
