// Copyright (c) 2026, mamba-ascendc authors.

#include "kernel_operator.h"
#include "lib/pad/broadcast.h"

namespace {

constexpr uint32_t kTile = 64;
constexpr uint32_t kElements = kTile * kTile;

template <typename T>
__aicore__ inline void CopyIn(
    const AscendC::LocalTensor<T> &dst,
    const AscendC::GlobalTensor<T> &src,
    int64_t offset,
    int64_t elements)
{
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> pad;
    pad.isPad = false;
    AscendC::DataCopyPad(dst, src[offset], params, pad);
}

template <typename T>
__aicore__ inline void CopyOut(
    const AscendC::GlobalTensor<T> &dst,
    const AscendC::LocalTensor<T> &src,
    int64_t offset,
    int64_t elements)
{
    AscendC::DataCopyExtParams params{
        1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst[offset], src, params);
}

class KernelMamba2SsdPrepareW {
public:
    __aicore__ inline void Init(
        GM_ADDR cb, GM_ADDR dA, GM_ADDR w,
        int64_t batch, int64_t nheads, int64_t nchunks, int64_t ngroups,
        int64_t usedCoreNum, AscendC::TPipe *pipe)
    {
        cbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cb));
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(w));
        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        ngroups_ = ngroups;
        usedCoreNum_ = usedCoreNum;
        pipe->InitBuffer(cbBuf_, kElements * sizeof(half));
        pipe->InitBuffer(dABuf_, kTile * sizeof(float));
        pipe->InitBuffer(wBuf_, kElements * sizeof(half));
        pipe->InitBuffer(tmp0Buf_, kElements * sizeof(float));
        pipe->InitBuffer(tmp1Buf_, kElements * sizeof(float));
        pipe->InitBuffer(maskBuf_, kElements * sizeof(float));
        pipe->InitBuffer(broadcastTmpBuf_, 2 * kElements);
        mte2ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        InitMask();
        // The short W-only task can reach the next GM copy before the
        // per-core mask initialization has retired on both 910B and C310.
        // The original combined kernel hid this dependency behind the R
        // epilogue; make it explicit now that W is an independent producer.
        AscendC::PipeBarrier<PIPE_ALL>();
        const int64_t taskCount = batch_ * nheads_ * nchunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            ProcessTask(task);
        }
    }

private:
    __aicore__ inline void InitMask()
    {
        auto mask = maskBuf_.Get<float>();
        AscendC::Duplicate(mask, 0.0f, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < kTile; ++row) {
            AscendC::Duplicate(mask[row * kTile], 1.0f, row + 1);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessTask(int64_t task)
    {
        const int64_t chunk = task % nchunks_;
        const int64_t head = (task / nchunks_) % nheads_;
        const int64_t batch = task / (nchunks_ * nheads_);
        const int64_t group = head / (nheads_ / ngroups_);
        const int64_t cbOffset =
            ((batch * nchunks_ + chunk) * ngroups_ + group) * kElements;
        const int64_t headChunk = (batch * nheads_ + head) * nchunks_ + chunk;
        const int64_t dAOffset = headChunk * kTile;
        const int64_t wOffset = headChunk * kElements;

        auto cb = cbBuf_.Get<half>();
        auto dA = dABuf_.Get<float>();
        auto w = wBuf_.Get<half>();
        auto tmp0 = tmp0Buf_.Get<float>();
        auto tmp1 = tmp1Buf_.Get<float>();
        auto broadcastTmp = broadcastTmpBuf_.Get<uint8_t>();

        CopyIn(cb, cbGm_, cbOffset, kElements);
        CopyIn(dA, dAGm_, dAOffset, kTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);

        const uint32_t colShape[2] = {1U, kTile};
        const uint32_t rowShape[2] = {kTile, 1U};
        const uint32_t matrixShape[2] = {kTile, kTile};
        AscendC::Broadcast<float, 2, 0>(
            tmp0, dA, matrixShape, colShape, broadcastTmp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Broadcast<float, 2, 1>(
            tmp1, dA, matrixShape, rowShape, broadcastTmp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmp0, tmp1, tmp0, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(tmp0, tmp0, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(tmp1, cb, AscendC::RoundMode::CAST_NONE, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp1, tmp1, tmp0, kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp1, tmp1, maskBuf_.Get<float>(), kElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(w, tmp1, AscendC::RoundMode::CAST_RINT, kElements);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        CopyOut(wGm_, w, wOffset, kElements);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        // W/output and cb/tmp buffers are reused by the next task on this
        // core.  Keep the cross-task lifetime conservative until Level2
        // confirms a narrower event is sufficient.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> cbBuf_, dABuf_, wBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp0Buf_, tmp1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_, broadcastTmpBuf_;
    AscendC::GlobalTensor<half> cbGm_, wGm_;
    AscendC::GlobalTensor<float> dAGm_;
    int64_t batch_, nheads_, nchunks_, ngroups_, usedCoreNum_;
    event_t mte2ToV_, vToMte3_, mte3ToV_;
};

class KernelMamba2SsdPrepareR {
public:
    __aicore__ inline void Init(
        GM_ADDR dA, GM_ADDR x, GM_ADDR weightedX,
        int64_t batch, int64_t nheads, int64_t nchunks,
        int64_t usedCoreNum, AscendC::TPipe *pipe)
    {
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
        weightedXGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(weightedX));
        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        usedCoreNum_ = usedCoreNum;
        pipe->InitBuffer(dABuf_, kTile * sizeof(float));
        pipe->InitBuffer(xBuf_, kElements * sizeof(half));
        pipe->InitBuffer(weightedXBuf_, kElements * sizeof(half));
        pipe->InitBuffer(tmp0Buf_, kElements * sizeof(float));
        pipe->InitBuffer(tmp1Buf_, kElements * sizeof(float));
        pipe->InitBuffer(broadcastTmpBuf_, 2 * kElements);
        mte2ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * nheads_ * nchunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            const int64_t dAOffset = task * kTile;
            const int64_t xOffset = task * kElements;
            auto dA = dABuf_.Get<float>();
            auto x = xBuf_.Get<half>();
            auto weightedX = weightedXBuf_.Get<half>();
            auto tmp0 = tmp0Buf_.Get<float>();
            auto tmp1 = tmp1Buf_.Get<float>();
            auto broadcastTmp = broadcastTmpBuf_.Get<uint8_t>();

            CopyIn(dA, dAGm_, dAOffset, kTile);
            CopyIn(x, xGm_, xOffset, kElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);

            AscendC::Muls(tmp0, dA, -1.0f, kTile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmp0, tmp0, dA.GetValue(kTile - 1), kTile);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tmp0, tmp0, kTile);
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t rowShape[2] = {kTile, 1U};
            const uint32_t matrixShape[2] = {kTile, kTile};
            AscendC::Broadcast<float, 2, 1>(
                tmp1, tmp0, matrixShape, rowShape, broadcastTmp);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                tmp0, x, AscendC::RoundMode::CAST_NONE, kElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tmp0, tmp0, tmp1, kElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                weightedX, tmp0, AscendC::RoundMode::CAST_RINT, kElements);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
            CopyOut(weightedXGm_, weightedX, xOffset, kElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
            // The next logical task reuses weightedX/tmp immediately.  The
            // event pair alone is not a sufficient cross-task retirement
            // point for this short producer on either supported architecture.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> dABuf_, xBuf_, weightedXBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp0Buf_, tmp1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastTmpBuf_;
    AscendC::GlobalTensor<float> dAGm_;
    AscendC::GlobalTensor<half> xGm_, weightedXGm_;
    int64_t batch_, nheads_, nchunks_, usedCoreNum_;
    event_t mte2ToV_, vToMte3_, mte3ToV_;
};

}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_prepare_w(
    GM_ADDR cb, GM_ADDR dA, GM_ADDR w,
    int64_t batch, int64_t nheads, int64_t nchunks, int64_t ngroups,
    int64_t usedCoreNum)
{
    KernelMamba2SsdPrepareW op;
    AscendC::TPipe pipe;
    op.Init(cb, dA, w, batch, nheads, nchunks, ngroups, usedCoreNum, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void mamba2_ssd_prepare_r(
    GM_ADDR dA, GM_ADDR x, GM_ADDR weightedX,
    int64_t batch, int64_t nheads, int64_t nchunks, int64_t usedCoreNum)
{
    KernelMamba2SsdPrepareR op;
    AscendC::TPipe pipe;
    op.Init(dA, x, weightedX, batch, nheads, nchunks, usedCoreNum, &pipe);
    op.Process();
}
