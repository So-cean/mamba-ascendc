// Copyright (c) 2026, mamba-ascendc authors.

#include "kernel_operator.h"

class KernelMamba2SsdFwd {
public:
    __aicore__ inline KernelMamba2SsdFwd() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR dt, GM_ADDR a, GM_ADDR b, GM_ADDR c,
        GM_ADDR d, GM_ADDR z, GM_ADDR dtBias, GM_ADDR initialStates,
        GM_ADDR out, GM_ADDR finalState,
        int64_t batch, int64_t seqlen, int64_t nheads, int64_t headdim,
        int64_t dstate, int64_t ngroups, int64_t hasD, int64_t dHasHdim,
        int64_t hasZ, int64_t hasDtBias, int64_t hasInitialState,
        int64_t dtSoftplus, float dtLimitMin, float dtLimitMax,
        int64_t usedCoreNum)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x);
        dtGm.SetGlobalBuffer((__gm__ float *)dt);
        aGm.SetGlobalBuffer((__gm__ float *)a);
        bGm.SetGlobalBuffer((__gm__ float *)b);
        cGm.SetGlobalBuffer((__gm__ float *)c);
        dGm.SetGlobalBuffer((__gm__ float *)d);
        zGm.SetGlobalBuffer((__gm__ float *)z);
        dtBiasGm.SetGlobalBuffer((__gm__ float *)dtBias);
        initialGm.SetGlobalBuffer((__gm__ float *)initialStates);
        outGm.SetGlobalBuffer((__gm__ float *)out);
        finalGm.SetGlobalBuffer((__gm__ float *)finalState);

        batch_ = batch;
        seqlen_ = seqlen;
        nheads_ = nheads;
        headdim_ = headdim;
        dstate_ = dstate;
        ngroups_ = ngroups;
        hasD_ = hasD;
        dHasHdim_ = dHasHdim;
        hasZ_ = hasZ;
        hasDtBias_ = hasDtBias;
        hasInitialState_ = hasInitialState;
        dtSoftplus_ = dtSoftplus;
        dtLimitMin_ = dtLimitMin;
        dtLimitMax_ = dtLimitMax;
        usedCoreNum_ = usedCoreNum;

        const int64_t stateBytes = headdim_ * dstate_ * sizeof(float);
        const int64_t alignedStateBytes = (stateBytes + 31) / 32 * 32;
        const int64_t outBytes = headdim_ * sizeof(float);
        const int64_t alignedOutBytes = (outBytes + 31) / 32 * 32;
        pipe.InitBuffer(stateBuf, alignedStateBytes);
        pipe.InitBuffer(outBuf, alignedOutBytes);
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * nheads_;
        const int64_t blockIdx = AscendC::GetBlockIdx();
        for (int64_t task = blockIdx; task < taskCount; task += usedCoreNum_) {
            ProcessTask(task / nheads_, task % nheads_);
        }
    }

private:
    __aicore__ inline float Pow2FromExponent(int32_t exponent)
    {
        if (exponent < -126) {
            return 0.0f;
        }
        if (exponent > 127) {
            exponent = 127;
        }
        union FloatBits {
            uint32_t bits;
            float value;
        } result;
        result.bits = static_cast<uint32_t>(exponent + 127) << 23;
        return result.value;
    }

    // Range-reduced degree-6 approximation. Relative error is below 2e-6 over
    // the recurrence's clamped finite range and is substantially more accurate
    // than the bit-only approximation used by the discarded ops-nn prototype.
    __aicore__ inline float ScalarExp(float x)
    {
        if (x <= -87.0f) {
            return 0.0f;
        }
        if (x >= 88.0f) {
            x = 88.0f;
        }
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

    __aicore__ inline float ScalarLog(float x)
    {
        union FloatBits {
            uint32_t bits;
            float value;
        } value;
        value.value = x;
        const int32_t exponent = static_cast<int32_t>((value.bits >> 23) & 0xff) - 127;
        value.bits = (value.bits & 0x007fffffU) | 0x3f800000U;
        const float y = (value.value - 1.0f) / (value.value + 1.0f);
        const float y2 = y * y;
        float term = y;
        float series = term;
        term *= y2;
        series += term / 3.0f;
        term *= y2;
        series += term / 5.0f;
        term *= y2;
        series += term / 7.0f;
        term *= y2;
        series += term / 9.0f;
        term *= y2;
        series += term / 11.0f;
        return 2.0f * series + static_cast<float>(exponent) * 0.6931471805599453f;
    }

    __aicore__ inline float Softplus(float x)
    {
        if (x > 20.0f) {
            return x;
        }
        // For sufficiently negative x, log(1 + exp(x)) loses exp(x) when the
        // addition rounds to exactly 1 in FP32. The asymptotic exp(x) branch
        // keeps the relative error below the FP32 precision gate.
        if (x < -9.0f) {
            return ScalarExp(x);
        }
        if (x >= 0.0f) {
            return x + ScalarLog(1.0f + ScalarExp(-x));
        }
        return ScalarLog(1.0f + ScalarExp(x));
    }

    __aicore__ inline float ClampDt(float value)
    {
        if (value < dtLimitMin_) {
            value = dtLimitMin_;
        }
        if (value > dtLimitMax_) {
            value = dtLimitMax_;
        }
        return value;
    }

    __aicore__ inline void ProcessTask(int64_t batchIdx, int64_t headIdx)
    {
        AscendC::LocalTensor<float> state = stateBuf.Get<float>();
        AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();
        const int64_t group = headIdx / (nheads_ / ngroups_);

        for (int64_t n = 0; n < dstate_; ++n) {
            for (int64_t p = 0; p < headdim_; ++p) {
                const int64_t stateOffset = n * headdim_ + p;
                float value = 0.0f;
                if (hasInitialState_ != 0) {
                    const int64_t initialOffset =
                        ((batchIdx * nheads_ + headIdx) * headdim_ + p) * dstate_ + n;
                    value = initialGm.GetValue(initialOffset);
                }
                state.SetValue(stateOffset, value);
            }
        }

        const float aValue = aGm.GetValue(headIdx);
        const float biasValue = hasDtBias_ != 0 ? dtBiasGm.GetValue(headIdx) : 0.0f;

        for (int64_t t = 0; t < seqlen_; ++t) {
            const int64_t dtOffset = (batchIdx * seqlen_ + t) * nheads_ + headIdx;
            float dtValue = dtGm.GetValue(dtOffset) + biasValue;
            if (dtSoftplus_ != 0) {
                dtValue = Softplus(dtValue);
            }
            dtValue = ClampDt(dtValue);
            const float decay = ScalarExp(dtValue * aValue);
            const int64_t xBase = ((batchIdx * seqlen_ + t) * nheads_ + headIdx) * headdim_;
            const int64_t bcBase = ((batchIdx * seqlen_ + t) * ngroups_ + group) * dstate_;

            for (int64_t n = 0; n < dstate_; ++n) {
                const float bValue = bGm.GetValue(bcBase + n);
                for (int64_t p = 0; p < headdim_; ++p) {
                    const int64_t stateOffset = n * headdim_ + p;
                    const float oldState = state.GetValue(stateOffset);
                    const float xValue = xGm.GetValue(xBase + p);
                    state.SetValue(stateOffset,
                                   decay * oldState + bValue * dtValue * xValue);
                }
            }

            for (int64_t p = 0; p < headdim_; ++p) {
                float yValue = 0.0f;
                for (int64_t n = 0; n < dstate_; ++n) {
                    yValue += cGm.GetValue(bcBase + n) *
                              state.GetValue(n * headdim_ + p);
                }
                const float xValue = xGm.GetValue(xBase + p);
                if (hasD_ != 0) {
                    const int64_t dOffset = dHasHdim_ != 0 ?
                        headIdx * headdim_ + p : headIdx;
                    yValue += dGm.GetValue(dOffset) * xValue;
                }
                if (hasZ_ != 0) {
                    const float zValue = zGm.GetValue(xBase + p);
                    const float silu = zValue / (1.0f + ScalarExp(-zValue));
                    yValue *= silu;
                }
                outLocal.SetValue(p, yValue);
            }
            AscendC::DataCopyExtParams outCopyParams{
                1, static_cast<uint32_t>(headdim_ * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(outGm[xBase], outLocal, outCopyParams);
        }

        for (int64_t p = 0; p < headdim_; ++p) {
            for (int64_t n = 0; n < dstate_; ++n) {
                const int64_t finalOffset =
                    ((batchIdx * nheads_ + headIdx) * headdim_ + p) * dstate_ + n;
                finalGm.SetValue(finalOffset, state.GetValue(n * headdim_ + p));
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> dtGm;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    AscendC::GlobalTensor<float> dGm;
    AscendC::GlobalTensor<float> zGm;
    AscendC::GlobalTensor<float> dtBiasGm;
    AscendC::GlobalTensor<float> initialGm;
    AscendC::GlobalTensor<float> outGm;
    AscendC::GlobalTensor<float> finalGm;

    int64_t batch_;
    int64_t seqlen_;
    int64_t nheads_;
    int64_t headdim_;
    int64_t dstate_;
    int64_t ngroups_;
    int64_t hasD_;
    int64_t dHasHdim_;
    int64_t hasZ_;
    int64_t hasDtBias_;
    int64_t hasInitialState_;
    int64_t dtSoftplus_;
    int64_t usedCoreNum_;
    float dtLimitMin_;
    float dtLimitMax_;
};

extern "C" __global__ __aicore__ void mamba2_ssd_fwd(
    GM_ADDR x, GM_ADDR dt, GM_ADDR a, GM_ADDR b, GM_ADDR c,
    GM_ADDR d, GM_ADDR z, GM_ADDR dtBias, GM_ADDR initialStates,
    GM_ADDR out, GM_ADDR finalState,
    int64_t batch, int64_t seqlen, int64_t nheads, int64_t headdim,
    int64_t dstate, int64_t ngroups, int64_t hasD, int64_t dHasHdim,
    int64_t hasZ, int64_t hasDtBias, int64_t hasInitialState,
    int64_t dtSoftplus, float dtLimitMin, float dtLimitMax,
    int64_t usedCoreNum)
{
    KernelMamba2SsdFwd op;
    op.Init(x, dt, a, b, c, d, z, dtBias, initialStates, out, finalState,
            batch, seqlen, nheads, headdim, dstate, ngroups,
            hasD, dHasHdim, hasZ, hasDtBias, hasInitialState,
            dtSoftplus, dtLimitMin, dtLimitMax, usedCoreNum);
    op.Process();
}
