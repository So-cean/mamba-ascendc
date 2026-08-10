// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"

class KernelMamba2SsdStatePassingBwd {
public:
    __aicore__ inline void Init(
        GM_ADDR statesStart, GM_ADDR dStatesStart, GM_ADDR dACumsum,
        GM_ADDR dfinalState, GM_ADDR dChunkStates, GM_ADDR dinitialState,
        GM_ADDR dAChunkLast, int64_t batch, int64_t nheads,
        int64_t nchunks, int64_t headdim, int64_t dstate,
        int64_t chunkSize, int64_t hasDfinal, int64_t usedCoreNum)
    {
        statesStartGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(statesStart));
        dStatesStartGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dStatesStart));
        dACumsumGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dACumsum));
        dfinalStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dfinalState));
        dChunkStatesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dChunkStates));
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
        stateElements_ = headdim_ * dstate_;

        pipe_.InitBuffer(gstateBuf_, stateElements_ * sizeof(float));
        pipe_.InitBuffer(productBuf_, stateElements_ * sizeof(float));
        pipe_.InitBuffer(reduceWorkBuf_, stateElements_ * sizeof(float));
        pipe_.InitBuffer(scalarBuf_, 64 * sizeof(float));
        pipe_.InitBuffer(inQueue_, 1, stateElements_ * sizeof(float));
        pipe_.InitBuffer(outQueue_, 1, stateElements_ * sizeof(float));
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
        int64_t offset,
        int64_t elements)
    {
        AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
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

    __aicore__ inline void ProcessTask(int64_t task)
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

            auto output = outQueue_.AllocTensor<float>();
            AscendC::Adds(output, gstate, 0.0f, stateElements_);
            outQueue_.EnQue(output);
            output = outQueue_.DeQue<float>();
            CopyOut(dChunkStatesGm_, output, stateOffset, stateElements_);
            outQueue_.FreeTensor(output);

            auto input = inQueue_.AllocTensor<float>();
            CopyIn(input, statesStartGm_, stateOffset, stateElements_);
            inQueue_.EnQue(input);
            input = inQueue_.DeQue<float>();
            AscendC::Mul(product, gstate, input, stateElements_);
            inQueue_.FreeTensor(input);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float, true>(
                scalar[8], product, reduceWork, stateElements_);
            AscendC::PipeBarrier<PIPE_V>();

            CopyIn(scalar, dACumsumGm_, decayOffset, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            const float alpha = ScalarExp(scalar.GetValue(0));

            output = outQueue_.AllocTensor<float>();
            AscendC::Muls(output, scalar[8], alpha, 1);
            outQueue_.EnQue(output);
            output = outQueue_.DeQue<float>();
            CopyOut(dAChunkLastGm_, output,
                    task * nchunks_ + chunk, 1);
            outQueue_.FreeTensor(output);

            input = inQueue_.AllocTensor<float>();
            CopyIn(input, dStatesStartGm_, stateOffset, stateElements_);
            inQueue_.EnQue(input);
            input = inQueue_.DeQue<float>();
            AscendC::Muls(gstate, gstate, alpha, stateElements_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(gstate, gstate, input, stateElements_);
            inQueue_.FreeTensor(input);
        }

        auto output = outQueue_.AllocTensor<float>();
        AscendC::Adds(output, gstate, 0.0f, stateElements_);
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<float>();
        CopyOut(dinitialStateGm_, output, task * stateElements_,
                stateElements_);
        outQueue_.FreeTensor(output);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gstateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> productBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceWorkBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::GlobalTensor<float> statesStartGm_, dStatesStartGm_;
    AscendC::GlobalTensor<float> dACumsumGm_, dfinalStateGm_;
    AscendC::GlobalTensor<float> dChunkStatesGm_, dinitialStateGm_;
    AscendC::GlobalTensor<float> dAChunkLastGm_;
    int64_t batch_, nheads_, nchunks_, headdim_, dstate_, chunkSize_;
    int64_t hasDfinal_, usedCoreNum_, stateElements_;
};

extern "C" __global__ __aicore__ void mamba2_ssd_state_passing_bwd(
    GM_ADDR statesStart, GM_ADDR dStatesStart, GM_ADDR dACumsum,
    GM_ADDR dfinalState, GM_ADDR dChunkStates, GM_ADDR dinitialState,
    GM_ADDR dAChunkLast, int64_t batch, int64_t nheads, int64_t nchunks,
    int64_t headdim, int64_t dstate, int64_t chunkSize,
    int64_t hasDfinal, int64_t usedCoreNum)
{
    KernelMamba2SsdStatePassingBwd op;
    op.Init(statesStart, dStatesStart, dACumsum, dfinalState,
            dChunkStates, dinitialState, dAChunkLast, batch, nheads,
            nchunks, headdim, dstate, chunkSize, hasDfinal, usedCoreNum);
    op.Process();
}
