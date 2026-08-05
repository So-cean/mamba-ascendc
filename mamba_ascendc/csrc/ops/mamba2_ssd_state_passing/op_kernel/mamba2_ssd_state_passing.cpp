// Copyright (c) 2026, mamba-ascendc authors.

#include "kernel_operator.h"

class KernelMamba2SsdStatePassing {
public:
    __aicore__ inline void Init(
        GM_ADDR chunkStates, GM_ADDR dACumsum, GM_ADDR initialStates,
        GM_ADDR statesStart, GM_ADDR finalState,
        int64_t batch, int64_t nheads, int64_t nchunks, int64_t headdim,
        int64_t dstate, int64_t chunkSize, int64_t hasInitial,
        int64_t usedCoreNum)
    {
        chunkStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(chunkStates));
        dACumsumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum));
        initialStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initialStates));
        statesStartGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(statesStart));
        finalStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(finalState));
        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        headdim_ = headdim;
        dstate_ = dstate;
        chunkSize_ = chunkSize;
        hasInitial_ = hasInitial;
        usedCoreNum_ = usedCoreNum;
        stateElements_ = headdim_ * dstate_;
        pipe_.InitBuffer(stateBuf_, stateElements_ * sizeof(float));
        pipe_.InitBuffer(inQueue_, 1, stateElements_ * sizeof(float));
        pipe_.InitBuffer(outQueue_, 1, stateElements_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * nheads_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            auto state = stateBuf_.Get<float>();
            if (hasInitial_ != 0) {
                auto initial = inQueue_.AllocTensor<float>();
                AscendC::DataCopy(
                    initial, initialStatesGm_[task * stateElements_], stateElements_);
                inQueue_.EnQue(initial);
                initial = inQueue_.DeQue<float>();
                AscendC::Adds(state, initial, 0.0f, stateElements_);
                inQueue_.FreeTensor(initial);
            } else {
                AscendC::Duplicate(state, 0.0f, stateElements_);
            }

            for (int64_t chunk = 0; chunk < nchunks_; ++chunk) {
                const int64_t stateOffset =
                    (task * nchunks_ + chunk) * stateElements_;
                StoreState(statesStartGm_, stateOffset, state);

                const int64_t decayOffset =
                    (task * nchunks_ + chunk) * chunkSize_ +
                    chunkSize_ - 1;
                const float decay = ScalarExp(dACumsumGm_.GetValue(decayOffset));
                auto contribution = inQueue_.AllocTensor<float>();
                AscendC::DataCopy(
                    contribution, chunkStatesGm_[stateOffset], stateElements_);
                inQueue_.EnQue(contribution);
                contribution = inQueue_.DeQue<float>();
                AscendC::Muls(state, state, decay, stateElements_);
                AscendC::Add(state, state, contribution, stateElements_);
                inQueue_.FreeTensor(contribution);
            }
            StoreState(finalStateGm_, task * stateElements_, state);
        }
    }

private:
    __aicore__ inline void StoreState(
        AscendC::GlobalTensor<float> &dst, int64_t offset,
        const AscendC::LocalTensor<float> &state)
    {
        auto out = outQueue_.AllocTensor<float>();
        AscendC::Adds(out, state, 0.0f, stateElements_);
        outQueue_.EnQue(out);
        out = outQueue_.DeQue<float>();
        AscendC::DataCopy(dst[offset], out, stateElements_);
        outQueue_.FreeTensor(out);
    }

    __aicore__ inline float Pow2FromExponent(int32_t exponent)
    {
        if (exponent < -126) return 0.0f;
        if (exponent > 127) exponent = 127;
        union FloatBits { uint32_t bits; float value; } result;
        result.bits = static_cast<uint32_t>(exponent + 127) << 23;
        return result.value;
    }

    __aicore__ inline float ScalarExp(float x)
    {
        if (x <= -87.0f) return 0.0f;
        if (x >= 88.0f) x = 88.0f;
        constexpr float invLn2 = 1.4426950408889634f;
        constexpr float ln2Hi = 0.6931457519531250f;
        constexpr float ln2Lo = 1.4286067653301870e-6f;
        const float scaled = x * invLn2;
        const int32_t exponent = static_cast<int32_t>(
            scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        const float r = (x - static_cast<float>(exponent) * ln2Hi) -
                        static_cast<float>(exponent) * ln2Lo;
        const float r2 = r * r;
        const float polynomial = 1.0f + r + r2 *
            (0.5f + r * (0.1666666716f + r * (0.0416666679f +
            r * (0.0083333338f + r * 0.0013888889f))));
        return polynomial * Pow2FromExponent(exponent);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::GlobalTensor<float> chunkStatesGm_, dACumsumGm_, initialStatesGm_;
    AscendC::GlobalTensor<float> statesStartGm_, finalStateGm_;
    int64_t batch_, nheads_, nchunks_, headdim_, dstate_, chunkSize_;
    int64_t hasInitial_, usedCoreNum_, stateElements_;
};

extern "C" __global__ __aicore__ void mamba2_ssd_state_passing(
    GM_ADDR chunkStates, GM_ADDR dACumsum, GM_ADDR initialStates,
    GM_ADDR statesStart, GM_ADDR finalState,
    int64_t batch, int64_t nheads, int64_t nchunks, int64_t headdim,
    int64_t dstate, int64_t chunkSize, int64_t hasInitial,
    int64_t usedCoreNum)
{
    KernelMamba2SsdStatePassing op;
    op.Init(chunkStates, dACumsum, initialStates, statesStart, finalState,
            batch, nheads, nchunks, headdim, dstate, chunkSize,
            hasInitial, usedCoreNum);
    op.Process();
}
