// Copyright (c) 2026, mamba-ascendc authors.

#include "kernel_operator.h"

class KernelMamba2SsdPrepare {
public:
    __aicore__ inline void Init(
        GM_ADDR cb, GM_ADDR dACumsum, GM_ADDR xCube,
        GM_ADDR wCube, GM_ADDR weightedXCube,
        int64_t batch, int64_t nheads, int64_t nchunks, int64_t ngroups,
        int64_t chunkSize, int64_t headdim, int64_t usedCoreNum,
        AscendC::TPipe *pipe)
    {
        cbGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cb));
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(xCube));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wCube));
        weightedXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weightedXCube));
        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        ngroups_ = ngroups;
        chunkSize_ = chunkSize;
        headdim_ = headdim;
        usedCoreNum_ = usedCoreNum;
        matrixElements_ = chunkSize_ * chunkSize_;
        xElements_ = chunkSize_ * headdim_;

        pipe->InitBuffer(cbBuf_, matrixElements_ * sizeof(half));
        pipe->InitBuffer(xBuf_, xElements_ * sizeof(half));
        pipe->InitBuffer(dABuf_, chunkSize_ * sizeof(float));
        pipe->InitBuffer(wBuf_, matrixElements_ * sizeof(half));
        pipe->InitBuffer(weightedXBuf_, xElements_ * sizeof(half));
        pipe->InitBuffer(floatBuf0_, chunkSize_ * sizeof(float));
        pipe->InitBuffer(floatBuf1_, chunkSize_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * nheads_ * nchunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            ProcessTask(task);
        }
    }

private:
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

    template <typename T>
    __aicore__ inline void CopyIn(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::GlobalTensor<T> &src,
        int64_t offset,
        int64_t elements)
    {
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        AscendC::DataCopyPad(dst, src[offset], copyParams, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<T> &dst,
        const AscendC::LocalTensor<T> &src,
        int64_t offset,
        int64_t elements)
    {
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(dst[offset], src, copyParams);
    }

    __aicore__ inline void ProcessTask(int64_t task)
    {
        const int64_t chunk = task % nchunks_;
        const int64_t head = (task / nchunks_) % nheads_;
        const int64_t batch = task / (nchunks_ * nheads_);
        const int64_t headsPerGroup = nheads_ / ngroups_;
        const int64_t group = head / headsPerGroup;
        const int64_t cbOffset =
            ((batch * nchunks_ + chunk) * ngroups_ + group) * matrixElements_;
        const int64_t headChunk = (batch * nheads_ + head) * nchunks_ + chunk;
        const int64_t dAOffset = headChunk * chunkSize_;
        const int64_t xOffset = headChunk * xElements_;

        auto cb = cbBuf_.Get<half>();
        auto x = xBuf_.Get<half>();
        auto dA = dABuf_.Get<float>();
        auto w = wBuf_.Get<half>();
        auto weightedX = weightedXBuf_.Get<half>();
        auto tmp0 = floatBuf0_.Get<float>();
        auto tmp1 = floatBuf1_.Get<float>();

        CopyIn(cb, cbGm_, cbOffset, matrixElements_);
        CopyIn(x, xGm_, xOffset, xElements_);
        CopyIn(dA, dAGm_, dAOffset, chunkSize_);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Causal lower triangle: cb[i,j] * exp(dA[i] - dA[j]).
        // The complete cb/x blocks stay in UB; only the final FP16 matrices
        // are written to GM for the following Cube matmuls.
        for (int64_t row = 0; row < chunkSize_; ++row) {
            const int64_t valid = row + 1;
            const int64_t rowOffset = row * chunkSize_;
            AscendC::Duplicate(w[rowOffset], static_cast<half>(0.0f), chunkSize_);
            AscendC::Cast(tmp0, cb[rowOffset], AscendC::RoundMode::CAST_NONE, valid);
            AscendC::Muls(tmp1, dA, -1.0f, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmp1, tmp1, dA.GetValue(row), valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tmp1, tmp1, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(tmp0, tmp0, tmp1, valid);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(w[rowOffset], tmp0, AscendC::RoundMode::CAST_NONE, valid);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Do not consume a Vector result through scalar GetValue: that path is
        // not synchronized by PIPE_V for short vectors on 910B.  Read the
        // already synchronized dA buffer and form one scalar per x row instead.
        const float dALast = dA.GetValue(chunkSize_ - 1);
        for (int64_t row = 0; row < chunkSize_; ++row) {
            const int64_t rowOffset = row * headdim_;
            const float decay = ScalarExp(dALast - dA.GetValue(row));
            AscendC::Cast(tmp0, x[rowOffset], AscendC::RoundMode::CAST_NONE, headdim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmp0, tmp0, decay, headdim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(weightedX[rowOffset], tmp0,
                          AscendC::RoundMode::CAST_NONE, headdim_);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CopyOut(wGm_, w, headChunk * matrixElements_, matrixElements_);
        CopyOut(weightedXGm_, weightedX, xOffset, xElements_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> cbBuf_, xBuf_, dABuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf_, weightedXBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuf0_, floatBuf1_;
    AscendC::GlobalTensor<half> cbGm_, xGm_, wGm_, weightedXGm_;
    AscendC::GlobalTensor<float> dAGm_;
    int64_t batch_, nheads_, nchunks_, ngroups_, chunkSize_, headdim_;
    int64_t usedCoreNum_, matrixElements_, xElements_;
};

extern "C" __global__ __aicore__ void mamba2_ssd_prepare(
    GM_ADDR cb, GM_ADDR dACumsum, GM_ADDR xCube,
    GM_ADDR wCube, GM_ADDR weightedXCube,
    int64_t batch, int64_t nheads, int64_t nchunks, int64_t ngroups,
    int64_t chunkSize, int64_t headdim, int64_t usedCoreNum)
{
    KernelMamba2SsdPrepare op;
    AscendC::TPipe pipe;
    op.Init(cb, dACumsum, xCube, wCube, weightedXCube,
            batch, nheads, nchunks, ngroups, chunkSize, headdim,
            usedCoreNum, &pipe);
    op.Process();
}
