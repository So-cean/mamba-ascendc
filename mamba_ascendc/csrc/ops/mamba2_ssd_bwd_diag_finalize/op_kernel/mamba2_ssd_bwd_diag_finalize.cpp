// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"
#include "lib/pad/broadcast.h"

namespace {
constexpr int64_t kTile = 64;
constexpr int64_t kElements = kTile * kTile;

class KernelMamba2SsdBwdDiagFinalize {
public:
    __aicore__ inline void Init(
        GM_ADDR dxDiag, GM_ADDR dR, GM_ADDR dBDiag, GM_ADDR dBState,
        GM_ADDR dCDiag, GM_ADDR dCOff, GM_ADDR dW, GM_ADDR w, GM_ADDR r,
        GM_ADDR dACumsum, GM_ADDR dx, GM_ADDR dBGroup,
        GM_ADDR dCGroup, GM_ADDR gCs,
        int64_t batch, int64_t heads, int64_t chunks, int64_t groups,
        int64_t headsPerGroup, int64_t headBlockSize,
        int64_t groupedDiag, int64_t groupedDR, int64_t hasDCOff,
        int64_t usedCoreNum)
    {
        dxDiagGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dxDiag));
        dRGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dR));
        dBDiagGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dBDiag));
        dBStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dBState));
        dCDiagGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dCDiag));
        dCOffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dCOff));
        dWGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dW));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w));
        rGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(r));
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum));
        dxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dx));
        dBGroupGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dBGroup));
        dCGroupGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dCGroup));
        gCsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gCs));
        batch_ = batch;
        heads_ = heads;
        chunks_ = chunks;
        groups_ = groups;
        headsPerGroup_ = headsPerGroup;
        headBlockSize_ = headBlockSize;
        groupedDiag_ = groupedDiag;
        groupedDR_ = groupedDR;
        hasDCOff_ = hasDCOff;
        usedCoreNum_ = usedCoreNum;

        pipe_.InitBuffer(
            halfBuf0_, headBlockSize_ * kElements * sizeof(half));
        pipe_.InitBuffer(
            halfBuf1_, headBlockSize_ * kElements * sizeof(half));
        pipe_.InitBuffer(
            halfBuf2_, headBlockSize_ * kElements * sizeof(half));
        pipe_.InitBuffer(floatBuf0_, kElements * sizeof(float));
        pipe_.InitBuffer(floatBuf1_, kElements * sizeof(float));
        pipe_.InitBuffer(productBuf_, kElements * sizeof(float));
        pipe_.InitBuffer(dBAccBuf_, kElements * sizeof(float));
        pipe_.InitBuffer(dCAccBuf_, kElements * sizeof(float));
        pipe_.InitBuffer(
            dABuf_, headBlockSize_ * kTile * sizeof(float));
        pipe_.InitBuffer(decayBuf_, kTile * sizeof(float));
        pipe_.InitBuffer(diagRowsBuf_, kTile * sizeof(float));
        pipe_.InitBuffer(
            stateRowsBuf_, headBlockSize_ * kTile * sizeof(float));
        pipe_.InitBuffer(
            gBuf_, headBlockSize_ * kTile * sizeof(float));
        pipe_.InitBuffer(broadcastTmp_, 2 * kElements);

        vToMte2Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE2));
        mte2ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3Event_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToVEvent_ = static_cast<event_t>(
            pipe_.FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * chunks_ * groups_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            const int64_t group = task % groups_;
            const int64_t chunk = (task / groups_) % chunks_;
            const int64_t batch = task / (groups_ * chunks_);
            ProcessGroup(batch, group, chunk);
        }
    }

private:
    template <typename T>
    __aicore__ inline void CopyInHeadBlock(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::GlobalTensor<T> &src,
        int64_t firstOffset, int64_t elementsPerHead,
        int64_t headCount)
    {
        // Public inputs are [B,H,K,...].  For a fixed chunk, consecutive
        // heads are separated by K matrices/vectors in GM.  One strided DMA
        // compacts up to four heads in UB, turning 8-KiB matrix reads into a
        // 16/24/32-KiB burst without changing their logical layout.
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(headCount),
            static_cast<uint32_t>(elementsPerHead * sizeof(T)),
            static_cast<uint32_t>(
                (chunks_ - 1) * elementsPerHead * sizeof(T)),
            0, 0};
        AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(dst, src[firstOffset], params, pad);
    }

    template <typename T>
    __aicore__ inline void CopyIn(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::GlobalTensor<T> &src,
        int64_t offset, int64_t elements)
    {
        AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline void CopyInGroupedDR(
        const AscendC::LocalTensor<half> &dst,
        int64_t batch, int64_t chunk, int64_t group,
        int64_t firstHeadInGroup, int64_t headCount)
    {
        // dR is producer-native [B,K,G,T,R,P].  Gather each head's P-vector
        // across the T rows into a head-major UB matrix.  No GM transpose or
        // Vector transpose is required; MTE2 performs the layout exchange
        // while compacting the current head block.
        for (int64_t headOffset = 0; headOffset < headCount; ++headOffset) {
            const int64_t headInGroup = firstHeadInGroup + headOffset;
            const int64_t groupTask =
                (batch * chunks_ + chunk) * groups_ + group;
            const int64_t firstOffset =
                (groupTask * kTile * headsPerGroup_ + headInGroup) * kTile;
            AscendC::DataCopyExtParams params{
                static_cast<uint16_t>(kTile),
                static_cast<uint32_t>(kTile * sizeof(half)),
                static_cast<uint32_t>(
                    (headsPerGroup_ - 1) * kTile * sizeof(half)),
                0, 0};
            AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(
                dst[headOffset * kElements], dRGm_[firstOffset],
                params, pad);
        }
    }

    template <typename T>
    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<T> &dst,
        const AscendC::LocalTensor<T> &src,
        int64_t offset, int64_t elements)
    {
        AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(dst[offset], src, params);
    }

    __aicore__ inline void CopyOutHeadBlockVector(
        const AscendC::GlobalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        int64_t firstOffset, int64_t headCount)
    {
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(headCount),
            static_cast<uint32_t>(kTile * sizeof(float)),
            0,
            static_cast<uint32_t>(
                (chunks_ - 1) * kTile * sizeof(float)),
            0};
        AscendC::DataCopyPad(dst[firstOffset], src, params);
    }

    __aicore__ inline void CopyOutGroupMatrix(
        const AscendC::GlobalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        int64_t batch, int64_t chunk, int64_t group)
    {
        // Destination is [B, K, T, G, N].  A group's N-vector is contiguous,
        // while consecutive T rows are separated by the other groups.
        const int64_t offset =
            ((batch * chunks_ + chunk) * kTile * groups_ + group) * kTile;
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(kTile),
            static_cast<uint32_t>(kTile * sizeof(float)),
            0,
            static_cast<uint32_t>((groups_ - 1) * kTile * sizeof(float)),
            0};
        AscendC::DataCopyPad(dst[offset], src, params);
    }

    __aicore__ inline void CopyInGroupMatrix(
        const AscendC::LocalTensor<float> &dst,
        const AscendC::GlobalTensor<float> &src,
        int64_t batch, int64_t chunk, int64_t group)
    {
        // Source is the public [B,K,T,G,N] layout produced by Off finalize.
        // Gather one group's N-vector from every T row directly into a
        // contiguous UB matrix, avoiding a full-device transpose.
        const int64_t offset =
            ((batch * chunks_ + chunk) * kTile * groups_ + group) * kTile;
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(kTile),
            static_cast<uint32_t>(kTile * sizeof(float)),
            static_cast<uint32_t>((groups_ - 1) * kTile * sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline void BeginLoad()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_);
    }

    __aicore__ inline void EndLoad()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void BeginStore()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
    }

    __aicore__ inline void FinishStoreForVectorReuse()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void ProcessGroup(
        int64_t batch, int64_t group, int64_t chunk)
    {
        auto half0 = halfBuf0_.Get<half>();
        auto half1 = halfBuf1_.Get<half>();
        auto half2 = halfBuf2_.Get<half>();
        auto float0 = floatBuf0_.Get<float>();
        auto float1 = floatBuf1_.Get<float>();
        auto product = productBuf_.Get<float>();
        auto dBAcc = dBAccBuf_.Get<float>();
        auto dCAcc = dCAccBuf_.Get<float>();
        auto dA = dABuf_.Get<float>();
        auto decay = decayBuf_.Get<float>();
        auto diagRows = diagRowsBuf_.Get<float>();
        auto stateRows = stateRowsBuf_.Get<float>();
        auto g = gBuf_.Get<float>();

        if (groupedDiag_ != 0) {
            const int64_t groupMatrixOffset =
                ((batch * groups_ + group) * chunks_ + chunk) * kElements;
            BeginLoad();
            CopyIn(half0, dBDiagGm_, groupMatrixOffset, kElements);
            CopyIn(half1, dCDiagGm_, groupMatrixOffset, kElements);
            if (hasDCOff_ != 0) {
                CopyInGroupMatrix(float0, dCOffGm_, batch, chunk, group);
            }
            EndLoad();
            AscendC::Cast(
                dBAcc, half0, AscendC::RoundMode::CAST_NONE, kElements);
            AscendC::Cast(
                dCAcc, half1, AscendC::RoundMode::CAST_NONE, kElements);
            AscendC::PipeBarrier<PIPE_V>();
            if (hasDCOff_ != 0) {
                AscendC::Add(dCAcc, dCAcc, float0, kElements);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            AscendC::Duplicate(dBAcc, 0.0f, kElements);
            if (hasDCOff_ != 0) {
                BeginLoad();
                CopyInGroupMatrix(float0, dCOffGm_, batch, chunk, group);
                EndLoad();
                AscendC::Muls(dCAcc, float0, 1.0f, kElements);
            } else {
                AscendC::Duplicate(dCAcc, 0.0f, kElements);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (int64_t headBlockBegin = 0;
             headBlockBegin < headsPerGroup_;
             headBlockBegin += headBlockSize_) {
            const int64_t remaining = headsPerGroup_ - headBlockBegin;
            const int64_t blockHeads = remaining < headBlockSize_
                ? remaining : headBlockSize_;
            const int64_t firstHead =
                group * headsPerGroup_ + headBlockBegin;
            const int64_t firstHeadTask =
                (batch * heads_ + firstHead) * chunks_ + chunk;
            const int64_t firstMatrixOffset = firstHeadTask * kElements;
            const int64_t firstVectorOffset = firstHeadTask * kTile;

            // Load up to four public-layout heads per tensor with one strided
            // DMA.  H/G=1 is the exact single-head fallback; the final block
            // naturally handles H/G not divisible by four.
            BeginLoad();
            CopyInHeadBlock(
                half0, dxDiagGm_, firstMatrixOffset, kElements,
                blockHeads);
            if (groupedDR_ != 0) {
                CopyInGroupedDR(
                    half1, batch, chunk, group, headBlockBegin,
                    blockHeads);
            } else {
                CopyInHeadBlock(
                    half1, dRGm_, firstMatrixOffset, kElements,
                    blockHeads);
            }
            CopyInHeadBlock(
                dA, dAGm_, firstVectorOffset, kTile, blockHeads);
            EndLoad();

            for (int64_t headOffset = 0; headOffset < blockHeads;
                 ++headOffset) {
                const int64_t localMatrixOffset = headOffset * kElements;
                const int64_t localVectorOffset = headOffset * kTile;
                const int64_t matrixOffset = firstMatrixOffset +
                    headOffset * chunks_ * kElements;
                auto dAHead = dA[localVectorOffset];
                const float lastDA = dAHead.GetValue(kTile - 1);
                AscendC::Muls(decay, dAHead, -1.0f, kTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(decay, decay, lastDA, kTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(decay, decay, kTile);
                AscendC::PipeBarrier<PIPE_V>();
                const uint32_t rowShape[2] = {kTile, 1U};
                const uint32_t matrixShape[2] = {kTile, kTile};
                auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
                AscendC::Broadcast<float, 2, 1>(
                    product, decay, matrixShape, rowShape,
                    broadcastTmp);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(
                    float0, half0[localMatrixOffset],
                    AscendC::RoundMode::CAST_NONE, kElements);
                AscendC::Cast(
                    float1, half1[localMatrixOffset],
                    AscendC::RoundMode::CAST_NONE, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(float1, float1, product, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(float0, float0, float1, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(
                    half0[localMatrixOffset], float0,
                    AscendC::RoundMode::CAST_RINT, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                BeginStore();
                CopyOut(
                    dxGm_, half0[localMatrixOffset],
                    matrixOffset, kElements);
                FinishStoreForVectorReuse();
            }

            // Retain all block-local dR row sums before half0/half1 are
            // reused for dW/W.  The increasing-head order is unchanged.
            BeginLoad();
            CopyInHeadBlock(
                half0, rGm_, firstMatrixOffset, kElements, blockHeads);
            EndLoad();
            for (int64_t headOffset = 0; headOffset < blockHeads;
                 ++headOffset) {
                const int64_t localMatrixOffset = headOffset * kElements;
                auto stateRowsHead = stateRows[headOffset * kTile];
                AscendC::Cast(
                    float0, half1[localMatrixOffset],
                    AscendC::RoundMode::CAST_NONE, kElements);
                AscendC::Cast(
                    float1, half0[localMatrixOffset],
                    AscendC::RoundMode::CAST_NONE, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, float0, float1, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float, true>(
                    stateRowsHead, product, kTile, kTile, 1, 1,
                    kTile * sizeof(float) /
                        AscendC::DEFAULT_C0_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
            }

            BeginLoad();
            CopyInHeadBlock(
                half0, dWGm_, firstMatrixOffset, kElements, blockHeads);
            CopyInHeadBlock(
                half1, wGm_, firstMatrixOffset, kElements, blockHeads);
            EndLoad();
            for (int64_t headOffset = 0; headOffset < blockHeads;
                 ++headOffset) {
                const int64_t localMatrixOffset = headOffset * kElements;
                auto stateRowsHead = stateRows[headOffset * kTile];
                auto gHead = g[headOffset * kTile];
                AscendC::Cast(
                    float0, half0[localMatrixOffset],
                    AscendC::RoundMode::CAST_NONE, kElements);
                AscendC::Cast(
                    float1, half1[localMatrixOffset],
                    AscendC::RoundMode::CAST_NONE, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, float0, float1, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(float0, product, 1.0f, kElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float, true>(
                    diagRows, float0, kTile, kTile, 1, 1,
                    kTile * sizeof(float) /
                        AscendC::DEFAULT_C0_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                for (int64_t rows = kTile / 2; rows >= 1;
                     rows >>= 1) {
                    AscendC::Add(
                        product, product, product[rows * kTile],
                        rows * kTile);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Sub(diagRows, diagRows, product, kTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(gHead, diagRows, stateRowsHead, kTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float, true>(
                    decay, stateRowsHead, kTile, 1, 1, 1,
                    kTile * sizeof(float) /
                        AscendC::DEFAULT_C0_SIZE);
                AscendC::PipeBarrier<PIPE_ALL>();
                gHead.SetValue(
                    kTile - 1,
                    gHead.GetValue(kTile - 1) + decay.GetValue(0));
            }
            BeginStore();
            CopyOutHeadBlockVector(
                gCsGm_, g, firstVectorOffset, blockHeads);
            FinishStoreForVectorReuse();

            BeginLoad();
            if (groupedDiag_ != 0) {
                CopyInHeadBlock(
                    half0, dBStateGm_, firstMatrixOffset, kElements,
                    blockHeads);
            } else {
                CopyInHeadBlock(
                    half0, dBDiagGm_, firstMatrixOffset, kElements,
                    blockHeads);
                CopyInHeadBlock(
                    half1, dBStateGm_, firstMatrixOffset, kElements,
                    blockHeads);
                CopyInHeadBlock(
                    half2, dCDiagGm_, firstMatrixOffset, kElements,
                    blockHeads);
            }
            EndLoad();
            for (int64_t headOffset = 0; headOffset < blockHeads;
                 ++headOffset) {
                const int64_t localMatrixOffset = headOffset * kElements;
                if (groupedDiag_ != 0) {
                    AscendC::Cast(
                        float0, half0[localMatrixOffset],
                        AscendC::RoundMode::CAST_NONE, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(dBAcc, dBAcc, float0, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    AscendC::Cast(
                        float0, half0[localMatrixOffset],
                        AscendC::RoundMode::CAST_NONE, kElements);
                    AscendC::Cast(
                        float1, half1[localMatrixOffset],
                        AscendC::RoundMode::CAST_NONE, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(float0, float0, float1, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(dBAcc, dBAcc, float0, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(
                        float0, half2[localMatrixOffset],
                        AscendC::RoundMode::CAST_NONE, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(dCAcc, dCAcc, float0, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }

        BeginStore();
        CopyOutGroupMatrix(dBGroupGm_, dBAcc, batch, chunk, group);
        CopyOutGroupMatrix(dCGroupGm_, dCAcc, batch, chunk, group);
        // The next strided group task starts by clearing dBAcc/dCAcc with
        // Vector Duplicate.  MTE3_MTE2 would only protect a following DMA
        // write and allowed that Vector overwrite to race the two stores.
        FinishStoreForVectorReuse();
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfBuf0_, halfBuf1_, halfBuf2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuf0_, floatBuf1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dBAccBuf_, dCAccBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dABuf_, decayBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diagRowsBuf_, stateRowsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gBuf_, broadcastTmp_;
    AscendC::GlobalTensor<half> dxDiagGm_, dRGm_, dBDiagGm_, dBStateGm_;
    AscendC::GlobalTensor<half> dCDiagGm_, dWGm_, wGm_, rGm_;
    AscendC::GlobalTensor<float> dCOffGm_, dAGm_, dBGroupGm_;
    AscendC::GlobalTensor<half> dxGm_;
    AscendC::GlobalTensor<float> dCGroupGm_, gCsGm_;
    event_t vToMte2Event_, mte2ToVEvent_, vToMte3Event_;
    event_t mte3ToVEvent_;
    int64_t batch_, heads_, chunks_, groups_, headsPerGroup_;
    int64_t headBlockSize_, groupedDiag_, groupedDR_, hasDCOff_, usedCoreNum_;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_bwd_diag_finalize(
    GM_ADDR dxDiag, GM_ADDR dR, GM_ADDR dBDiag, GM_ADDR dBState,
    GM_ADDR dCDiag, GM_ADDR dCOff, GM_ADDR dW, GM_ADDR w, GM_ADDR r,
    GM_ADDR dACumsum, GM_ADDR dx, GM_ADDR dBGroup,
    GM_ADDR dCGroup, GM_ADDR gCs,
    int64_t batch, int64_t heads, int64_t chunks, int64_t groups,
    int64_t headsPerGroup, int64_t headBlockSize,
    int64_t groupedDiag, int64_t groupedDR, int64_t hasDCOff,
    int64_t usedCoreNum)
{
    KernelMamba2SsdBwdDiagFinalize op;
    op.Init(dxDiag, dR, dBDiag, dBState, dCDiag, dCOff, dW, w, r,
            dACumsum, dx, dBGroup, dCGroup, gCs,
            batch, heads, chunks, groups, headsPerGroup, headBlockSize,
            groupedDiag, groupedDR, hasDCOff, usedCoreNum);
    op.Process();
}
