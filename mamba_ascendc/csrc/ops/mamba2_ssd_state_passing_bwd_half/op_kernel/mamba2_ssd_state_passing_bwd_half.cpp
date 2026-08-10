// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"

namespace {
class KernelMamba2SsdStatePassingBwdHalf {
public:
    __aicore__ inline void Init(
        GM_ADDR statesStart, GM_ADDR dStatesStart, GM_ADDR dACumsum,
        GM_ADDR dfinalState, GM_ADDR dChunkStatesHalf,
        GM_ADDR dinitialState, GM_ADDR dAChunkLast,
        int64_t batch, int64_t nheads, int64_t nchunks,
        int64_t headdim, int64_t dstate, int64_t chunkSize,
        int64_t hasDfinal, int64_t usedCoreNum, int64_t stateIsHalf,
        int64_t dStateIsHalf, int64_t groups, int64_t headsPerGroup,
        int64_t inputGrouped,
        AscendC::TPipe *pipe)
    {
        statesStartGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(statesStart));
        statesStartHalfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(statesStart));
        dStatesStartGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dStatesStart));
        dStatesStartHalfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(dStatesStart));
        dACumsumGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dACumsum));
        dfinalStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dfinalState));
        dChunkStatesHalfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(dChunkStatesHalf));
        dinitialStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dinitialState));
        dAChunkLastGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dAChunkLast));

        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        headdim_ = headdim;
        dstate_ = dstate;
        chunkSize_ = chunkSize;
        hasDfinal_ = hasDfinal;
        usedCoreNum_ = usedCoreNum;
        stateIsHalf_ = stateIsHalf;
        dStateIsHalf_ = dStateIsHalf;
        groups_ = groups;
        headsPerGroup_ = headsPerGroup;
        inputGrouped_ = inputGrouped;
        stateElements_ = headdim_ * dstate_;
        pipelined_ = stateIsHalf_ != 0 && dstate_ == 64;

        pipe->InitBuffer(gstateBuf_, stateElements_ * sizeof(float));
        pipe->InitBuffer(productBuf_, stateElements_ * sizeof(float));
        pipe->InitBuffer(reduceWorkBuf_, stateElements_ * sizeof(float));
        pipe->InitBuffer(scalarBuf_, 64 * sizeof(float));
        pipe->InitBuffer(inQueue_, 1, stateElements_ * sizeof(float));
        pipe->InitBuffer(outQueue_, 1, stateElements_ * sizeof(float));
        pipe->InitBuffer(halfOutBuf_, stateElements_ * sizeof(half));
        if (pipelined_) {
            pipe->InitBuffer(stateHalfPrefetch_, 2,
                             stateElements_ * sizeof(half));
            if (dStateIsHalf_ != 0) {
                pipe->InitBuffer(dStateHalfPrefetch_, 2,
                                 stateElements_ * sizeof(half));
            } else {
                pipe->InitBuffer(dStatePrefetch_, 2,
                                 stateElements_ * sizeof(float));
            }
        }
        if (inputGrouped_ != 0) {
            pipe->InitBuffer(
                transposeOffsetsBuf_, stateElements_ * sizeof(uint32_t));
            auto offsets = transposeOffsetsBuf_.Get<uint32_t>();
            for (int64_t row = 0; row < dstate_; ++row) {
                for (int64_t col = 0; col < headdim_; ++col) {
                    offsets.SetValue(
                        row * headdim_ + col,
                        static_cast<uint32_t>(
                            (col * dstate_ + row) * sizeof(float)));
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        vToMte3_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE3_V));
        mte2ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE2_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * nheads_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
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
        AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(dst, src[offset], params, pad);
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

    __aicore__ inline void CopyInGroupedHalf(
        const AscendC::LocalTensor<half> &dst,
        const AscendC::GlobalTensor<half> &src,
        int64_t task, int64_t chunk)
    {
        const int64_t batch = task / nheads_;
        const int64_t head = task % nheads_;
        const int64_t group = head / headsPerGroup_;
        const int64_t headInGroup = head % headsPerGroup_;
        const int64_t groupTask =
            (batch * nchunks_ + chunk) * groups_ + group;
        const int64_t offset =
            (groupTask * dstate_ * headsPerGroup_ + headInGroup) *
            headdim_;
        AscendC::DataCopyExtParams copy{
            static_cast<uint16_t>(dstate_),
            static_cast<uint32_t>(headdim_ * sizeof(half)),
            static_cast<uint32_t>(
                (headsPerGroup_ - 1) * headdim_ * sizeof(half)),
            0, 0};
        AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src[offset], copy, pad);
    }

    __aicore__ inline void StoreGroupedHalf(
        const AscendC::LocalTensor<half> &src,
        int64_t task, int64_t chunk)
    {
        // The recurrence state is already [N,P] in UB.  Preserve that
        // producer-native order in GM as [B,K,G,N,R,P], writing one head's
        // P-vector per N row with an R-head destination stride.  This removes
        // the previous per-chunk 64x64 Vector transpose.
        const int64_t batch = task / nheads_;
        const int64_t head = task % nheads_;
        const int64_t group = head / headsPerGroup_;
        const int64_t headInGroup = head % headsPerGroup_;
        const int64_t groupTask =
            (batch * nchunks_ + chunk) * groups_ + group;
        const int64_t offset =
            (groupTask * dstate_ * headsPerGroup_ + headInGroup) *
            headdim_;
        AscendC::DataCopyExtParams store{
            static_cast<uint16_t>(dstate_),
            static_cast<uint32_t>(headdim_ * sizeof(half)),
            0,
            static_cast<uint32_t>(
                (headsPerGroup_ - 1) * headdim_ * sizeof(half)),
            0};
        AscendC::DataCopyPad(dChunkStatesHalfGm_[offset], src, store);
    }

    __aicore__ inline float Pow2FromExponent(int32_t exponent)
    {
        if (exponent < -126) return 0.0f;
        if (exponent > 127) exponent = 127;
        union FloatBits { uint32_t bits; float value; } result;
        result.bits = static_cast<uint32_t>(exponent + 127) << 23;
        return result.value;
    }

    __aicore__ inline float ScalarExp(float value)
    {
        if (value <= -87.0f) return 0.0f;
        if (value >= 88.0f) value = 88.0f;
        constexpr float invLn2 = 1.4426950408889634f;
        constexpr float ln2Hi = 0.6931457519531250f;
        constexpr float ln2Lo = 1.4286067653301870e-6f;
        const float scaled = value * invLn2;
        const int32_t exponent = static_cast<int32_t>(
            scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        const float r = (value - static_cast<float>(exponent) * ln2Hi) -
                        static_cast<float>(exponent) * ln2Lo;
        const float r2 = r * r;
        const float polynomial = 1.0f + r + r2 *
            (0.5f + r * (0.1666666716f + r * (0.0416666679f +
            r * (0.0083333338f + r * 0.0013888889f))));
        return polynomial * Pow2FromExponent(exponent);
    }

    __aicore__ inline void StoreHalfState(
        int64_t task, int64_t chunk, int64_t stateOffset,
        const AscendC::LocalTensor<float> &gstate)
    {
        auto halfOut = halfOutBuf_.Get<half>();
        AscendC::Cast(
            halfOut, gstate, AscendC::RoundMode::CAST_RINT, stateElements_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        if (inputGrouped_ != 0) {
            StoreGroupedHalf(halfOut, task, chunk);
        } else {
            CopyOut(dChunkStatesHalfGm_, halfOut, stateOffset,
                    stateElements_);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void ProcessTask(int64_t task)
    {
        if (pipelined_) {
            ProcessTaskPipelined(task);
        } else {
            ProcessTaskSerial(task);
        }
    }

    __aicore__ inline void PrefetchChunk(int64_t task, int64_t chunk)
    {
        const int64_t stateOffset =
            (task * nchunks_ + chunk) * stateElements_;
        auto stateHalf = stateHalfPrefetch_.AllocTensor<half>();
        if (inputGrouped_ != 0) {
            CopyInGroupedHalf(stateHalf, statesStartHalfGm_, task, chunk);
        } else {
            CopyIn(stateHalf, statesStartHalfGm_, stateOffset,
                   stateElements_);
        }
        stateHalfPrefetch_.EnQue(stateHalf);

        if (dStateIsHalf_ != 0) {
            auto dStateHalf = dStateHalfPrefetch_.AllocTensor<half>();
            if (inputGrouped_ != 0) {
                CopyInGroupedHalf(
                    dStateHalf, dStatesStartHalfGm_, task, chunk);
            } else {
                CopyIn(dStateHalf, dStatesStartHalfGm_, stateOffset,
                       stateElements_);
            }
            dStateHalfPrefetch_.EnQue(dStateHalf);
        } else {
            auto dState = dStatePrefetch_.AllocTensor<float>();
            CopyIn(dState, dStatesStartGm_, stateOffset, stateElements_);
            dStatePrefetch_.EnQue(dState);
        }
    }

    __aicore__ inline void ProcessTaskPipelined(int64_t task)
    {
        auto gstate = gstateBuf_.Get<float>();
        auto product = productBuf_.Get<float>();
        auto reduceWork = reduceWorkBuf_.Get<float>();
        auto scalar = scalarBuf_.Get<float>();

        if (hasDfinal_ != 0) {
            auto input = inQueue_.AllocTensor<float>();
            CopyIn(input, dfinalStateGm_, task * stateElements_,
                   stateElements_);
            inQueue_.EnQue(input);
            input = inQueue_.DeQue<float>();
            if (inputGrouped_ != 0) {
                AscendC::Gather(
                    gstate, input,
                    transposeOffsetsBuf_.Get<uint32_t>(), 0U,
                    static_cast<uint32_t>(stateElements_));
            } else {
                AscendC::Adds(gstate, input, 0.0f, stateElements_);
            }
            inQueue_.FreeTensor(input);
        } else {
            AscendC::Duplicate(gstate, 0.0f, stateElements_);
        }

        // Queue the last chunk before entering the recurrence.  At each
        // iteration the following chunk is submitted into the second slot
        // before current Vector work starts, allowing MTE2 to run ahead by
        // one chunk while preserving the sequential gstate dependency.
        PrefetchChunk(task, nchunks_ - 1);
        for (int64_t chunk = nchunks_ - 1; chunk >= 0; --chunk) {
            if (chunk > 0) {
                PrefetchChunk(task, chunk - 1);
            }
            const int64_t stateOffset =
                (task * nchunks_ + chunk) * stateElements_;
            const int64_t decayOffset =
                (task * nchunks_ + chunk) * chunkSize_ + chunkSize_ - 1;

            StoreHalfState(task, chunk, stateOffset, gstate);

            auto stateHalf = stateHalfPrefetch_.DeQue<half>();
            AscendC::Cast(product, stateHalf,
                          AscendC::RoundMode::CAST_NONE,
                          stateElements_);
            stateHalfPrefetch_.FreeTensor(stateHalf);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, gstate, product, stateElements_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float, true>(
                scalar[8], product, reduceWork, stateElements_);
            AscendC::PipeBarrier<PIPE_V>();

            CopyIn(scalar, dACumsumGm_, decayOffset, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float alpha = ScalarExp(scalar.GetValue(0));

            auto output = outQueue_.AllocTensor<float>();
            AscendC::Muls(output, scalar[8], alpha, 1);
            outQueue_.EnQue(output);
            output = outQueue_.DeQue<float>();
            CopyOut(dAChunkLastGm_, output,
                    task * nchunks_ + chunk, 1);
            outQueue_.FreeTensor(output);

            AscendC::Muls(gstate, gstate, alpha, stateElements_);
            AscendC::PipeBarrier<PIPE_V>();
            if (dStateIsHalf_ != 0) {
                auto dStateHalf = dStateHalfPrefetch_.DeQue<half>();
                AscendC::Cast(product, dStateHalf,
                              AscendC::RoundMode::CAST_NONE,
                              stateElements_);
                dStateHalfPrefetch_.FreeTensor(dStateHalf);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(gstate, gstate, product, stateElements_);
            } else {
                auto dState = dStatePrefetch_.DeQue<float>();
                AscendC::Add(gstate, gstate, dState, stateElements_);
                dStatePrefetch_.FreeTensor(dState);
            }
        }

        auto output = outQueue_.AllocTensor<float>();
        if (inputGrouped_ != 0) {
            AscendC::Gather(
                output, gstate,
                transposeOffsetsBuf_.Get<uint32_t>(), 0U,
                static_cast<uint32_t>(stateElements_));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Adds(output, gstate, 0.0f, stateElements_);
        }
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<float>();
        CopyOut(dinitialStateGm_, output,
                task * stateElements_, stateElements_);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline void ProcessTaskSerial(int64_t task)
    {
        auto gstate = gstateBuf_.Get<float>();
        auto product = productBuf_.Get<float>();
        auto reduceWork = reduceWorkBuf_.Get<float>();
        auto scalar = scalarBuf_.Get<float>();

        if (hasDfinal_ != 0) {
            auto input = inQueue_.AllocTensor<float>();
            CopyIn(input, dfinalStateGm_, task * stateElements_,
                   stateElements_);
            inQueue_.EnQue(input);
            input = inQueue_.DeQue<float>();
            AscendC::Adds(gstate, input, 0.0f, stateElements_);
            inQueue_.FreeTensor(input);
        } else {
            AscendC::Duplicate(gstate, 0.0f, stateElements_);
        }

        for (int64_t chunk = nchunks_ - 1; chunk >= 0; --chunk) {
            const int64_t stateOffset =
                (task * nchunks_ + chunk) * stateElements_;
            const int64_t decayOffset =
                (task * nchunks_ + chunk) * chunkSize_ + chunkSize_ - 1;

            StoreHalfState(task, chunk, stateOffset, gstate);

            AscendC::LocalTensor<float> input;
            if (stateIsHalf_ != 0) {
                auto stateHalf = halfOutBuf_.Get<half>();
                CopyIn(stateHalf, statesStartHalfGm_, stateOffset,
                       stateElements_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::Cast(product, stateHalf,
                              AscendC::RoundMode::CAST_NONE,
                              stateElements_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, gstate, product, stateElements_);
            } else {
                input = inQueue_.AllocTensor<float>();
                CopyIn(input, statesStartGm_, stateOffset, stateElements_);
                inQueue_.EnQue(input);
                input = inQueue_.DeQue<float>();
                AscendC::Mul(product, gstate, input, stateElements_);
                inQueue_.FreeTensor(input);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float, true>(
                scalar[8], product, reduceWork, stateElements_);
            AscendC::PipeBarrier<PIPE_V>();

            CopyIn(scalar, dACumsumGm_, decayOffset, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float alpha = ScalarExp(scalar.GetValue(0));

            auto output = outQueue_.AllocTensor<float>();
            AscendC::Muls(output, scalar[8], alpha, 1);
            outQueue_.EnQue(output);
            output = outQueue_.DeQue<float>();
            CopyOut(dAChunkLastGm_, output,
                    task * nchunks_ + chunk, 1);
            outQueue_.FreeTensor(output);

            AscendC::Muls(gstate, gstate, alpha, stateElements_);
            AscendC::PipeBarrier<PIPE_V>();
            if (dStateIsHalf_ != 0) {
                auto dStateHalf = halfOutBuf_.Get<half>();
                CopyIn(dStateHalf, dStatesStartHalfGm_, stateOffset,
                       stateElements_);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
                AscendC::Cast(product, dStateHalf,
                              AscendC::RoundMode::CAST_NONE,
                              stateElements_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(gstate, gstate, product, stateElements_);
            } else {
                input = inQueue_.AllocTensor<float>();
                CopyIn(input, dStatesStartGm_, stateOffset, stateElements_);
                inQueue_.EnQue(input);
                input = inQueue_.DeQue<float>();
                AscendC::Add(gstate, gstate, input, stateElements_);
                inQueue_.FreeTensor(input);
            }
        }

        auto output = outQueue_.AllocTensor<float>();
        AscendC::Adds(output, gstate, 0.0f, stateElements_);
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<float>();
        CopyOut(dinitialStateGm_, output,
                task * stateElements_, stateElements_);
        outQueue_.FreeTensor(output);
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> gstateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> halfOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transposeOffsetsBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> stateHalfPrefetch_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> dStateHalfPrefetch_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> dStatePrefetch_;
    AscendC::GlobalTensor<float> statesStartGm_, dStatesStartGm_;
    AscendC::GlobalTensor<half> statesStartHalfGm_, dStatesStartHalfGm_;
    AscendC::GlobalTensor<float> dACumsumGm_, dfinalStateGm_;
    AscendC::GlobalTensor<half> dChunkStatesHalfGm_;
    AscendC::GlobalTensor<float> dinitialStateGm_, dAChunkLastGm_;
    int64_t batch_, nheads_, nchunks_, headdim_, dstate_, chunkSize_;
    int64_t hasDfinal_, usedCoreNum_, stateElements_, stateIsHalf_;
    int64_t dStateIsHalf_;
    int64_t groups_, headsPerGroup_, inputGrouped_;
    bool pipelined_;
    event_t vToMte3_, mte3ToV_, mte2ToV_;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_state_passing_bwd_half(
    GM_ADDR statesStart, GM_ADDR dStatesStart, GM_ADDR dACumsum,
    GM_ADDR dfinalState, GM_ADDR dChunkStatesHalf, GM_ADDR dinitialState,
    GM_ADDR dAChunkLast, int64_t batch, int64_t nheads, int64_t nchunks,
    int64_t headdim, int64_t dstate, int64_t chunkSize,
    int64_t hasDfinal, int64_t usedCoreNum, int64_t stateIsHalf,
    int64_t dStateIsHalf, int64_t groups,
    int64_t headsPerGroup, int64_t inputGrouped)
{
    AscendC::TPipe pipe;
    KernelMamba2SsdStatePassingBwdHalf op;
    op.Init(statesStart, dStatesStart, dACumsum, dfinalState,
            dChunkStatesHalf, dinitialState, dAChunkLast,
            batch, nheads, nchunks, headdim, dstate, chunkSize,
            hasDfinal, usedCoreNum, stateIsHalf, dStateIsHalf,
            groups, headsPerGroup, inputGrouped, &pipe);
    op.Process();
}
