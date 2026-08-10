// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"

namespace {

constexpr int32_t BUFFER_NUM = 2;

class KernelMamba2SsdBwd {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR dt, GM_ADDR A, GM_ADDR B, GM_ADDR C,
        GM_ADDR dout, GM_ADDR finalState, GM_ADDR statesWork,
        GM_ADDR dfinalState,
        GM_ADDR dx, GM_ADDR ddt, GM_ADDR dAPartial, GM_ADDR dB, GM_ADDR dC,
        int64_t batch, int64_t seqlen, int64_t nheads, int64_t headdim,
        int64_t dstate, int64_t ngroups, int64_t hasDfinal,
        int64_t usedCoreNum)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        dtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dt));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(B));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(C));
        doutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dout));
        finalStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(finalState));
        statesWorkGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(statesWork));
        dfinalStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dfinalState));
        dxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dx));
        ddtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ddt));
        dAPartialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dAPartial));
        dBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dB));
        dCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dC));

        batch_ = batch;
        seqlen_ = seqlen;
        nheads_ = nheads;
        headdim_ = headdim;
        dstate_ = dstate;
        ngroups_ = ngroups;
        hasDfinal_ = hasDfinal;
        usedCoreNum_ = usedCoreNum;
        headsPerGroup_ = nheads_ / ngroups_;
        stateElements_ = headdim_ * dstate_;

        pipe_.InitBuffer(stateBuf_, stateElements_ * sizeof(float));
        pipe_.InitBuffer(gstateBuf_, stateElements_ * sizeof(float));
        pipe_.InitBuffer(statePrevBuf_, stateElements_ * sizeof(float));
        pipe_.InitBuffer(xBuf_, headdim_ * sizeof(float));
        pipe_.InitBuffer(doutBuf_, headdim_ * sizeof(float));
        pipe_.InitBuffer(dvBuf_, headdim_ * sizeof(float));
        pipe_.InitBuffer(dxBuf_, headdim_ * sizeof(float));
        pipe_.InitBuffer(bBuf_, dstate_ * sizeof(float));
        pipe_.InitBuffer(cBuf_, dstate_ * sizeof(float));
        pipe_.InitBuffer(dBBuf_, dstate_ * sizeof(float));
        pipe_.InitBuffer(dCBuf_, dstate_ * sizeof(float));
        pipe_.InitBuffer(scalarBuf_, 32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * ngroups_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            ProcessGroup(task);
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

    __aicore__ inline void ProcessGroup(int64_t task)
    {
        const int64_t batch = task / ngroups_;
        const int64_t group = task % ngroups_;
        const int64_t firstHead = group * headsPerGroup_;
        const int64_t groupBase =
            ((batch * seqlen_) * ngroups_ + group) * dstate_;

        for (int64_t headInGroup = 0; headInGroup < headsPerGroup_;
             ++headInGroup) {
            const int64_t head = firstHead + headInGroup;
            ProcessHead(batch, group, head, groupBase, headInGroup == 0);
        }
    }

    __aicore__ inline void ProcessHead(
        int64_t batch,
        int64_t group,
        int64_t head,
        int64_t groupBase,
        bool firstHead)
    {
        auto state = stateBuf_.Get<float>();
        auto gstate = gstateBuf_.Get<float>();
        auto statePrev = statePrevBuf_.Get<float>();
        auto xLocal = xBuf_.Get<float>();
        auto doutLocal = doutBuf_.Get<float>();
        auto dvLocal = dvBuf_.Get<float>();
        auto dxLocal = dxBuf_.Get<float>();
        auto bLocal = bBuf_.Get<float>();
        auto cLocal = cBuf_.Get<float>();
        auto dBLocal = dBBuf_.Get<float>();
        auto dCLocal = dCBuf_.Get<float>();
        auto scalar = scalarBuf_.Get<float>();

        const int64_t stateOffset =
            (batch * nheads_ + head) * stateElements_;
        const int64_t workOffset =
            AscendC::GetBlockIdx() * seqlen_ * stateElements_;
        const int64_t aOffset = head;
        CopyIn(scalar, aGm_, aOffset, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        const float aParam = scalar.GetValue(0);

        // M0 correctness path: recompute token states in FP32 and save them in
        // a per-core reusable workspace.  Reconstructing history by dividing
        // backward from final_state amplifies its forward rounding error in
        // dA/dC, even when the other gradients remain accurate.
        AscendC::Duplicate(state, 0.0f, stateElements_);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t t = 0; t < seqlen_; ++t) {
            const int64_t xOffset =
                ((batch * seqlen_ + t) * nheads_ + head) * headdim_;
            const int64_t dtOffset =
                (batch * seqlen_ + t) * nheads_ + head;
            const int64_t bcOffset = groupBase + t * ngroups_ * dstate_;
            CopyIn(xLocal, xGm_, xOffset, headdim_);
            CopyIn(bLocal, bGm_, bcOffset, dstate_);
            CopyIn(scalar[8], dtGm_, dtOffset, 1);
            AscendC::PipeBarrier<PIPE_ALL>();

            const float q = scalar.GetValue(8);
            const float decay = ScalarExp(aParam * q);
            AscendC::Muls(state, state, decay, stateElements_);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int64_t p = 0; p < headdim_; ++p) {
                const float xq = xLocal.GetValue(p) * q;
                for (int64_t n = 0; n < dstate_; ++n) {
                    const int64_t index = p * dstate_ + n;
                    state.SetValue(
                        index,
                        state.GetValue(index) + xq * bLocal.GetValue(n));
                }
            }
            CopyOut(
                statesWorkGm_, state,
                workOffset + t * stateElements_, stateElements_);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        if (hasDfinal_ != 0) {
            CopyIn(gstate, dfinalStateGm_, stateOffset, stateElements_);
        } else {
            AscendC::Duplicate(gstate, 0.0f, stateElements_);
        }
        float dAAccum = 0.0f;

        for (int64_t t = seqlen_ - 1; t >= 0; --t) {
            const int64_t xOffset =
                ((batch * seqlen_ + t) * nheads_ + head) * headdim_;
            const int64_t dtOffset =
                (batch * seqlen_ + t) * nheads_ + head;
            const int64_t bcOffset = groupBase + t * ngroups_ * dstate_;

            CopyIn(xLocal, xGm_, xOffset, headdim_);
            CopyIn(doutLocal, doutGm_, xOffset, headdim_);
            CopyIn(bLocal, bGm_, bcOffset, dstate_);
            CopyIn(cLocal, cGm_, bcOffset, dstate_);
            CopyIn(scalar[8], dtGm_, dtOffset, 1);
            CopyIn(
                state, statesWorkGm_,
                workOffset + t * stateElements_, stateElements_);
            if (t > 0) {
                CopyIn(
                    statePrev, statesWorkGm_,
                    workOffset + (t - 1) * stateElements_, stateElements_);
            } else {
                AscendC::Duplicate(statePrev, 0.0f, stateElements_);
            }
            if (firstHead) {
                AscendC::Duplicate(dBLocal, 0.0f, dstate_);
                AscendC::Duplicate(dCLocal, 0.0f, dstate_);
            } else {
                CopyIn(dBLocal, dBGm_, bcOffset, dstate_);
                CopyIn(dCLocal, dCGm_, bcOffset, dstate_);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            const float q = scalar.GetValue(8);
            const float decay = ScalarExp(aParam * q);
            AscendC::Duplicate(dvLocal, 0.0f, headdim_);
            float da = 0.0f;

            for (int64_t p = 0; p < headdim_; ++p) {
                const float xValue = xLocal.GetValue(p);
                const float gy = doutLocal.GetValue(p);
                float dv = 0.0f;
                for (int64_t n = 0; n < dstate_; ++n) {
                    const int64_t index = p * dstate_ + n;
                    const float bValue = bLocal.GetValue(n);
                    const float cValue = cLocal.GetValue(n);
                    const float stateValue = state.GetValue(index);
                    const float previous = statePrev.GetValue(index);

                    const float gs = gstate.GetValue(index) + gy * cValue;
                    gstate.SetValue(index, gs);
                    dCLocal.SetValue(
                        n, dCLocal.GetValue(n) + gy * stateValue);
                    dBLocal.SetValue(
                        n, dBLocal.GetValue(n) + gs * xValue * q);
                    dv += gs * bValue;
                    da += gs * previous;
                }
                dvLocal.SetValue(p, dv);
                dxLocal.SetValue(p, q * dv);
            }

            float ddtValue = da * decay * aParam;
            for (int64_t p = 0; p < headdim_; ++p) {
                ddtValue += dvLocal.GetValue(p) * xLocal.GetValue(p);
            }
            dAAccum += da * decay * q;
            scalar.SetValue(16, ddtValue);

            AscendC::Muls(gstate, gstate, decay, stateElements_);
            AscendC::PipeBarrier<PIPE_ALL>();

            CopyOut(dxGm_, dxLocal, xOffset, headdim_);
            CopyOut(ddtGm_, scalar[16], dtOffset, 1);
            CopyOut(dBGm_, dBLocal, bcOffset, dstate_);
            CopyOut(dCGm_, dCLocal, bcOffset, dstate_);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        scalar.SetValue(24, dAAccum);
        CopyOut(
            dAPartialGm_, scalar[24], batch * nheads_ + head, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf_, gstateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> statePrevBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_, doutBuf_, dvBuf_, dxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf_, cBuf_, dBBuf_, dCBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf_;
    AscendC::GlobalTensor<float> xGm_, dtGm_, aGm_, bGm_, cGm_, doutGm_;
    AscendC::GlobalTensor<float> finalStateGm_, statesWorkGm_, dfinalStateGm_;
    AscendC::GlobalTensor<float> dxGm_, ddtGm_, dAPartialGm_, dBGm_, dCGm_;
    int64_t batch_, seqlen_, nheads_, headdim_, dstate_, ngroups_;
    int64_t headsPerGroup_, stateElements_, hasDfinal_, usedCoreNum_;
};

}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_bwd(
    GM_ADDR x, GM_ADDR dt, GM_ADDR A, GM_ADDR B, GM_ADDR C,
    GM_ADDR dout, GM_ADDR finalState, GM_ADDR statesWork,
    GM_ADDR dfinalState,
    GM_ADDR dx, GM_ADDR ddt, GM_ADDR dAPartial, GM_ADDR dB, GM_ADDR dC,
    int64_t batch, int64_t seqlen, int64_t nheads, int64_t headdim,
    int64_t dstate, int64_t ngroups, int64_t hasDfinal,
    int64_t usedCoreNum)
{
    KernelMamba2SsdBwd op;
    op.Init(x, dt, A, B, C, dout, finalState, statesWork, dfinalState,
            dx, ddt, dAPartial, dB, dC,
            batch, seqlen, nheads, headdim, dstate, ngroups,
            hasDfinal, usedCoreNum);
    op.Process();
}
