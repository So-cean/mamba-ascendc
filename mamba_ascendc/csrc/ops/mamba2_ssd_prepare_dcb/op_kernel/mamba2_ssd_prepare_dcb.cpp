// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"
#include "lib/pad/broadcast.h"

namespace {
constexpr int64_t kTile = 64;
constexpr int64_t kElements = kTile * kTile;

class KernelMamba2SsdPrepareDcb {
public:
    __aicore__ inline void Init(
        GM_ADDR dW, GM_ADDR dA, GM_ADDR dCbGroup,
        int64_t batch, int64_t heads, int64_t chunks, int64_t groups,
        int64_t headsPerGroup, int64_t headBlockSize,
        int64_t usedCoreNum, AscendC::TPipe *pipe)
    {
        dWGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dW));
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA));
        dCbGroupGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(dCbGroup));
        batch_ = batch;
        heads_ = heads;
        chunks_ = chunks;
        groups_ = groups;
        headsPerGroup_ = headsPerGroup;
        headBlockSize_ = headBlockSize;
        usedCoreNum_ = usedCoreNum;
        pipe->InitBuffer(
            dWBuf_, headBlockSize_ * kElements * sizeof(half));
        pipe->InitBuffer(
            dABuf_, headBlockSize_ * kTile * sizeof(float));
        pipe->InitBuffer(dCbBuf_, kElements * sizeof(half));
        pipe->InitBuffer(tmp0Buf_, kElements * sizeof(float));
        pipe->InitBuffer(tmp1Buf_, kElements * sizeof(float));
        pipe->InitBuffer(groupAccBuf_, kElements * sizeof(float));
        pipe->InitBuffer(causalMaskBuf_, kElements * sizeof(float));
        pipe->InitBuffer(broadcastTmp_, 2 * kElements);
        mte2ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * groups_ * chunks_;
        const int64_t block = AscendC::GetBlockIdx();
        InitCausalMask();
        for (int64_t task = block; task < taskCount;
             task += usedCoreNum_) {
            const int64_t chunk = task % chunks_;
            const int64_t group = (task / chunks_) % groups_;
            const int64_t batch = task / (groups_ * chunks_);
            auto dW = dWBuf_.Get<half>();
            auto dA = dABuf_.Get<float>();
            auto dCb = dCbBuf_.Get<half>();
            auto tmp0 = tmp0Buf_.Get<float>();
            auto tmp1 = tmp1Buf_.Get<float>();
            auto groupAcc = groupAccBuf_.Get<float>();
            auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
            AscendC::Duplicate(groupAcc, 0.0f, kElements);
            AscendC::PipeBarrier<PIPE_V>();

            for (int64_t headBlockBegin = 0;
                 headBlockBegin < headsPerGroup_;
                 headBlockBegin += headBlockSize_) {
                const int64_t remaining = headsPerGroup_ - headBlockBegin;
                const int64_t blockHeads = remaining < headBlockSize_
                    ? remaining : headBlockSize_;
                const int64_t firstHead =
                    group * headsPerGroup_ + headBlockBegin;
                const int64_t firstTask =
                    (batch * heads_ + firstHead) * chunks_ + chunk;
                CopyInHeadBlock(
                    dW, dWGm_, firstTask * kElements,
                    kElements, blockHeads);
                CopyInHeadBlock(
                    dA, dAGm_, firstTask * kTile,
                    kTile, blockHeads);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);

                for (int64_t headOffset = 0; headOffset < blockHeads;
                     ++headOffset) {
                    auto dAHead = dA[headOffset * kTile];
                    const uint32_t colShape[2] = {1U, 64U};
                    const uint32_t rowShape[2] = {64U, 1U};
                    const uint32_t matrixShape[2] = {64U, 64U};
                    AscendC::Broadcast<float, 2, 0>(
                        tmp0, dAHead, matrixShape, colShape,
                        broadcastTmp);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Broadcast<float, 2, 1>(
                        tmp1, dAHead, matrixShape, rowShape,
                        broadcastTmp);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(tmp0, tmp1, tmp0, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Exp(tmp0, tmp0, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(
                        tmp1, dW[headOffset * kElements],
                        AscendC::RoundMode::CAST_NONE, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(tmp1, tmp1, tmp0, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    // Reuse the invariant per-core lower-triangular mask.
                    // This replaces 64 row clears and 64 partial casts for
                    // every head with two full-vector instructions.
                    AscendC::Mul(
                        tmp1, tmp1, causalMaskBuf_.Get<float>(), kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(
                        dCb, tmp1, AscendC::RoundMode::CAST_RINT,
                        kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    // Preserve the old per-head FP16 rounding before the
                    // group sum; only the location of the reduction changes.
                    AscendC::Cast(
                        tmp0, dCb, AscendC::RoundMode::CAST_NONE,
                        kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(
                        groupAcc, groupAcc, tmp0, kElements);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            AscendC::Cast(
                dCb, groupAcc, AscendC::RoundMode::CAST_RINT, kElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            const int64_t outputOffset =
                ((batch * groups_ + group) * chunks_ + chunk) * kElements;
            CopyOut(dCbGroupGm_, dCb, outputOffset, kElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void InitCausalMask()
    {
        auto mask = causalMaskBuf_.Get<float>();
        AscendC::Duplicate(mask, 0.0f, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t row = 0; row < kTile; ++row) {
            AscendC::Duplicate(mask[row * kTile], 1.0f, row + 1);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    template <typename T>
    __aicore__ inline void CopyInHeadBlock(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::GlobalTensor<T> &src,
        int64_t firstOffset, int64_t elementsPerHead,
        int64_t headCount)
    {
        AscendC::DataCopyExtParams copy{
            static_cast<uint16_t>(headCount),
            static_cast<uint32_t>(elementsPerHead * sizeof(T)),
            static_cast<uint32_t>(
                (chunks_ - 1) * elementsPerHead * sizeof(T)),
            0, 0};
        AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(dst, src[firstOffset], copy, pad);
    }

    template <typename T>
    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<T> &dst,
        const AscendC::LocalTensor<T> &src,
        int64_t offset, int64_t elements)
    {
        AscendC::DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(dst[offset], src, copy);
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> dWBuf_, dABuf_, dCbBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp0Buf_, tmp1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> groupAccBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> causalMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastTmp_;
    AscendC::GlobalTensor<half> dWGm_, dCbGroupGm_;
    AscendC::GlobalTensor<float> dAGm_;
    int64_t batch_, heads_, chunks_, groups_, headsPerGroup_;
    int64_t headBlockSize_, usedCoreNum_;
    event_t mte2ToV_, vToMte3_, mte3ToV_;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_prepare_dcb(
    GM_ADDR d_w, GM_ADDR d_a, GM_ADDR d_cb_group,
    int64_t batch, int64_t heads, int64_t chunks, int64_t groups,
    int64_t heads_per_group, int64_t head_block_size,
    int64_t used_core_num)
{
    KernelMamba2SsdPrepareDcb op;
    AscendC::TPipe pipe;
    op.Init(d_w, d_a, d_cb_group, batch, heads, chunks, groups,
            heads_per_group, head_block_size, used_core_num, &pipe);
    op.Process();
}
