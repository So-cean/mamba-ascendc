// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"
#include "lib/pad/broadcast.h"

namespace {
constexpr int64_t kTile = 64;
constexpr int64_t kElements = kTile * kTile;

class KernelMamba2SsdBwdOffPrepare {
public:
    __aicore__ inline void Init(
        GM_ADDR gy, GM_ADDR statesStart, GM_ADDR dACumsum,
        GM_ADDR qHalf, GM_ADDR stateHalf,
        int64_t batch, int64_t nheads, int64_t nchunks,
        int64_t chunksPerTask, int64_t taskCount, int64_t usedCoreNum,
        int64_t stateIsHalf, int64_t groups, int64_t outputGrouped,
        AscendC::TPipe *pipe)
    {
        gyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gy));
        statesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(statesStart));
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum));
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qHalf));
        stateHalfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(stateHalf));
        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        chunksPerTask_ = chunksPerTask;
        taskCount_ = taskCount;
        usedCoreNum_ = usedCoreNum;
        stateIsHalf_ = stateIsHalf;
        groups_ = groups;
        headsPerGroup_ = nheads / groups;
        outputGrouped_ = outputGrouped;

        const int64_t tileElements = chunksPerTask_ * kElements;
        pipe->InitBuffer(gyHalfBuf_, tileElements * sizeof(half));
        pipe->InitBuffer(gyBuf_, tileElements * sizeof(float));
        if (stateIsHalf_ == 0) {
            pipe->InitBuffer(statesBuf_, tileElements * sizeof(float));
            pipe->InitBuffer(stateHalfBuf_, tileElements * sizeof(half));
        }
        pipe->InitBuffer(
            dABuf_, chunksPerTask_ * kTile * sizeof(float));
        pipe->InitBuffer(decayBuf_, tileElements * sizeof(float));
        pipe->InitBuffer(qBuf_, tileElements * sizeof(half));
        pipe->InitBuffer(broadcastTmp_, 2 * tileElements);
        mte2ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount_;
             task += usedCoreNum_) {
            ProcessTask(task);
        }
    }

private:
    template <typename T>
    __aicore__ inline void CopyIn(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::GlobalTensor<T> &src,
        int64_t offset, int64_t elements)
    {
        AscendC::DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(dst, src[offset], copy, pad);
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

    __aicore__ inline void ProcessTask(int64_t task)
    {
        const int64_t chunkTiles =
            (nchunks_ + chunksPerTask_ - 1) / chunksPerTask_;
        const int64_t chunkTile = task % chunkTiles;
        const int64_t batchHead = task / chunkTiles;
        const int64_t chunkBegin = chunkTile * chunksPerTask_;
        const int64_t remainingChunks = nchunks_ - chunkBegin;
        const int64_t activeChunks =
            remainingChunks < chunksPerTask_ ?
                remainingChunks : chunksPerTask_;
        const int64_t activeRows = activeChunks * kTile;
        const int64_t activeElements = activeChunks * kElements;
        const int64_t matrixOffset =
            (batchHead * nchunks_ + chunkBegin) * kElements;
        const int64_t vectorOffset =
            (batchHead * nchunks_ + chunkBegin) * kTile;
        auto gy = gyBuf_.Get<float>();
        auto gyHalf = gyHalfBuf_.Get<half>();
        AscendC::LocalTensor<float> states;
        auto dA = dABuf_.Get<float>();
        auto decay = decayBuf_.Get<float>();
        auto q = qBuf_.Get<half>();
        AscendC::LocalTensor<half> stateHalf;
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();

        CopyIn(gyHalf, gyGm_, matrixOffset, activeElements);
        if (stateIsHalf_ == 0) {
            states = statesBuf_.Get<float>();
            stateHalf = stateHalfBuf_.Get<half>();
            CopyIn(states, statesGm_, matrixOffset, activeElements);
        }
        CopyIn(dA, dAGm_, vectorOffset, activeRows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);

        AscendC::Cast(
            gy, gyHalf, AscendC::RoundMode::CAST_NONE, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(dA, dA, activeRows);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t srcShape[2] = {
            static_cast<uint32_t>(activeRows), 1U};
        const uint32_t dstShape[2] = {
            static_cast<uint32_t>(activeRows), static_cast<uint32_t>(kTile)};
        AscendC::Broadcast<float, 2, 1>(
            decay, dA, dstShape, srcShape, broadcastTmp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gy, gy, decay, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            q, gy, AscendC::RoundMode::CAST_RINT, activeElements);
        if (stateIsHalf_ == 0) {
            AscendC::Cast(
                stateHalf, states, AscendC::RoundMode::CAST_RINT,
                activeElements);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        if (outputGrouped_ != 0) {
            const int64_t batch = batchHead / nheads_;
            const int64_t head = batchHead % nheads_;
            const int64_t group = head / headsPerGroup_;
            const int64_t headInGroup = head % headsPerGroup_;
            for (int64_t localChunk = 0; localChunk < activeChunks;
                 ++localChunk) {
                const int64_t chunk = chunkBegin + localChunk;
                const int64_t groupTask =
                    (batch * nchunks_ + chunk) * groups_ + group;
                const int64_t destination =
                    (groupTask * kTile * headsPerGroup_ + headInGroup) *
                    kTile;
                AscendC::DataCopyExtParams groupedCopy{
                    static_cast<uint16_t>(kTile),
                    static_cast<uint32_t>(kTile * sizeof(half)),
                    0,
                    static_cast<uint32_t>(
                        (headsPerGroup_ - 1) * kTile * sizeof(half)),
                    0};
                AscendC::DataCopyPad(
                    qGm_[destination],
                    q[localChunk * kElements], groupedCopy);
            }
        } else {
            CopyOut(qGm_, q, matrixOffset, activeElements);
        }
        if (stateIsHalf_ == 0) {
            CopyOut(
                stateHalfGm_, stateHalf, matrixOffset, activeElements);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> gyHalfBuf_, gyBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> statesBuf_, dABuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> decayBuf_, qBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateHalfBuf_, broadcastTmp_;
    AscendC::GlobalTensor<half> gyGm_;
    AscendC::GlobalTensor<float> statesGm_, dAGm_;
    AscendC::GlobalTensor<half> qGm_, stateHalfGm_;
    int64_t batch_, nheads_, nchunks_, chunksPerTask_;
    int64_t taskCount_, usedCoreNum_, stateIsHalf_;
    int64_t groups_, headsPerGroup_, outputGrouped_;
    event_t mte2ToV_, vToMte3_, mte3ToV_;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_bwd_off_prepare(
    GM_ADDR gy, GM_ADDR states_start, GM_ADDR d_a_cumsum,
    GM_ADDR q_half, GM_ADDR state_half,
    int64_t batch, int64_t nheads, int64_t nchunks,
    int64_t chunks_per_task, int64_t task_count,
    int64_t used_core_num, int64_t state_is_half,
    int64_t groups, int64_t output_grouped)
{
    KernelMamba2SsdBwdOffPrepare op;
    AscendC::TPipe pipe;
    op.Init(gy, states_start, d_a_cumsum, q_half, state_half,
            batch, nheads, nchunks, chunks_per_task,
            task_count, used_core_num, state_is_half, groups,
            output_grouped, &pipe);
    op.Process();
}
