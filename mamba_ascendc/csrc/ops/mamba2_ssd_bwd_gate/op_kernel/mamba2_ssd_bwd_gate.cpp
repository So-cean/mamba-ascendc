// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"

namespace {
constexpr int64_t kChunk = 64;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kTileRows = 32;
constexpr int64_t kTileElements = kTileRows * kHeadDim;
constexpr int64_t kHeadBlock = 8;
constexpr int64_t kHeadBlockTileRows = 8;
constexpr int64_t kHeadBlockRowElements = kHeadBlock * kHeadDim;
constexpr int64_t kHeadBlockTileElements =
    kHeadBlockTileRows * kHeadBlockRowElements;

class KernelMamba2SsdBwdGateHeadBlock {
public:
    __aicore__ inline void Init(
        GM_ADDR dout, GM_ADDR z, GM_ADDR yPre, GM_ADDR x,
        GM_ADDR gyHead, GM_ADDR dz, GM_ADDR dDPartial,
        int64_t batch, int64_t seqlen, int64_t nheads,
        int64_t nchunks, int64_t usedCoreNum, int64_t computeD,
        int64_t yPreIsHalf, AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        doutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dout));
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z));
        yPreFloatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yPre));
        yPreHalfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yPre));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        gyHeadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gyHead));
        dzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dz));
        dDPartialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dDPartial));
        batch_ = batch;
        seqlen_ = seqlen;
        nheads_ = nheads;
        nchunks_ = nchunks;
        usedCoreNum_ = usedCoreNum;
        computeD_ = computeD;
        yPreIsHalf_ = yPreIsHalf;

        pipe_->InitBuffer(doutBuf_, kHeadBlockTileElements * sizeof(float));
        pipe_->InitBuffer(zBuf_, kHeadBlockTileElements * sizeof(float));
        pipe_->InitBuffer(yPreBuf_, kHeadBlockTileElements * sizeof(float));
        pipe_->InitBuffer(
            xBuf_, computeD_ != 0
                ? kHeadBlockTileElements * sizeof(float) : 32);
        pipe_->InitBuffer(sigmoidBuf_, kHeadBlockTileElements * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, kHeadBlockTileElements * sizeof(float));
        pipe_->InitBuffer(scratchBuf_, kHeadBlockTileElements * sizeof(float));
        pipe_->InitBuffer(
            dDAccBuf_, computeD_ != 0
                ? kHeadBlockRowElements * sizeof(float) : 32);
        pipe_->InitBuffer(
            gyHalfBuf_, kHeadBlockTileElements * sizeof(half));
        mte2ToVEvent_ = static_cast<event_t>(
            pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3Event_ = static_cast<event_t>(
            pipe_->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToMte2Event_ = static_cast<event_t>(
            pipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        mte3ToVEvent_ = static_cast<event_t>(
            pipe_->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t nHeadBlocks =
            (nheads_ + kHeadBlock - 1) / kHeadBlock;
        const int64_t taskCount = batch_ * nHeadBlocks * nchunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            const int64_t chunk = task % nchunks_;
            const int64_t headBlock = (task / nchunks_) % nHeadBlocks;
            const int64_t batch = task / (nchunks_ * nHeadBlocks);
            ProcessChunk(batch, headBlock, chunk);
        }
    }

private:
    __aicore__ inline void ProcessChunk(
        int64_t batch, int64_t headBlock, int64_t chunk)
    {
        const int64_t firstToken = chunk * kChunk;
        const int64_t headBegin = headBlock * kHeadBlock;
        const int64_t remainingHeads = nheads_ - headBegin;
        const int64_t headCount =
            remainingHeads < kHeadBlock ? remainingHeads : kHeadBlock;
        const int64_t rowElements = headCount * kHeadDim;
        const int64_t activeElements = kHeadBlockTileRows * rowElements;

        auto dout = doutBuf_.Get<float>();
        auto z = zBuf_.Get<float>();
        auto yPre = yPreBuf_.Get<float>();
        auto x = xBuf_.Get<float>();
        auto sigmoid = sigmoidBuf_.Get<float>();
        auto tmp = tmpBuf_.Get<float>();
        auto scratch = scratchBuf_.Get<float>();
        auto dDAcc = dDAccBuf_.Get<float>();
        auto gyHalf = gyHalfBuf_.Get<half>();

        for (int64_t tile = 0;
             tile < kChunk / kHeadBlockTileRows; ++tile) {
        const int64_t rowBegin = tile * kHeadBlockTileRows;
        const int64_t publicOffset =
            ((batch * seqlen_ + firstToken + rowBegin) * nheads_ + headBegin) *
            kHeadDim;

        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(kHeadBlockTileRows),
            static_cast<uint32_t>(rowElements * sizeof(float)),
            static_cast<uint32_t>((nheads_ - headCount) *
                                  kHeadDim * sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(dout, doutGm_[publicOffset], inputParams, noPad);
        AscendC::DataCopyPad(z, zGm_[publicOffset], inputParams, noPad);
        if (yPreIsHalf_ != 0) {
            AscendC::DataCopyExtParams yPreHalfParams{
                static_cast<uint16_t>(kHeadBlockTileRows),
                static_cast<uint32_t>(rowElements * sizeof(half)),
                static_cast<uint32_t>((nheads_ - headCount) *
                                      kHeadDim * sizeof(half)),
                0, 0};
            AscendC::DataCopyPadExtParams<half> noPadHalf{
                false, 0, 0, static_cast<half>(0)};
            AscendC::DataCopyPad(
                gyHalf, yPreHalfGm_[publicOffset],
                yPreHalfParams, noPadHalf);
        } else {
            AscendC::DataCopyPad(
                yPre, yPreFloatGm_[publicOffset], inputParams, noPad);
        }
        if (computeD_ != 0) {
            AscendC::DataCopyPad(x, xGm_[publicOffset], inputParams, noPad);
        }
        // PipeBarrier orders instructions within a pipeline but does not
        // make four asynchronous MTE2 copies visible to Vector.  The race is
        // intermittent at small task counts and deterministic to reproduce
        // once a core loops over many chunks, so use the required hard
        // producer/consumer event before the first vector read.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_);
        if (yPreIsHalf_ != 0) {
            AscendC::Cast(
                yPre, gyHalf, AscendC::RoundMode::CAST_NONE,
                activeElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // sigmoid(z) = 1 / (1 + exp(-z)).  Keep a copy of the denominator
        // and refine the reciprocal estimate twice.  Clamp only the exponent
        // input; this preserves the saturated 0/1 result for extreme z.
        AscendC::Muls(sigmoid, z, -1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(sigmoid, sigmoid, 80.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(sigmoid, sigmoid, -80.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(sigmoid, sigmoid, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(sigmoid, sigmoid, 1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scratch, sigmoid, 1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Reciprocal(sigmoid, sigmoid, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        for (int iteration = 0; iteration < 2; ++iteration) {
            AscendC::Mul(tmp, scratch, sigmoid, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmp, tmp, -1.0f, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmp, tmp, 2.0f, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(sigmoid, sigmoid, tmp, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // dz = dout * y_pre * sigmoid(z) * (1 + z * (1 - sigmoid(z))).
        AscendC::Muls(tmp, sigmoid, -1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tmp, tmp, 1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp, tmp, z, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tmp, tmp, 1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yPre, yPre, dout, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yPre, yPre, sigmoid, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yPre, yPre, tmp, activeElements);
        AscendC::PipeBarrier<PIPE_V>();

        // gy = dout * silu(z). The dz factor in tmp is dead now, so reuse
        // the tile. The separate scratch tile keeps x intact and removes the
        // baseline's second GM->UB x transfer.
        AscendC::Mul(tmp, z, sigmoid, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp, tmp, dout, activeElements);
        AscendC::PipeBarrier<PIPE_V>();

        if (computeD_ != 0) {
            // x remains intact because Newton refinement uses scratch.
            // Reduce all heads in the block across eight time rows together.
            AscendC::Mul(scratch, x, tmp, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t rows = kHeadBlockTileRows / 2;
                 rows >= 1; rows >>= 1) {
                AscendC::Add(
                    scratch, scratch, scratch[rows * rowElements],
                    rows * rowElements);
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (tile == 0) {
                AscendC::Muls(dDAcc, scratch, 1.0f, rowElements);
            } else {
                AscendC::Add(dDAcc, dDAcc, scratch, rowElements);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        // gy_head is an internal workspace consumed by FP16 Cube GEMMs.
        // Preserve FP32 gate/dD arithmetic and cast only at the write boundary
        // to halve both this store and all downstream gy reads.
        AscendC::Cast(
            gyHalf, tmp, AscendC::RoundMode::CAST_RINT, activeElements);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
        AscendC::DataCopyExtParams gyParams{
            static_cast<uint16_t>(kHeadBlockTileRows),
            static_cast<uint32_t>(kHeadDim * sizeof(half)),
            static_cast<uint32_t>((headCount - 1) * kHeadDim *
                                  sizeof(half) / 32),
            0, 0};
        for (int64_t localHead = 0; localHead < headCount; ++localHead) {
            const int64_t head = headBegin + localHead;
            const int64_t headOffset =
                ((((batch * nheads_ + head) * nchunks_ + chunk) *
                  kChunk + rowBegin) * kHeadDim);
            AscendC::DataCopyPad(
                gyHeadGm_[headOffset],
                gyHalf[localHead * kHeadDim], gyParams);
        }
        AscendC::DataCopyExtParams dzParams{
            static_cast<uint16_t>(kHeadBlockTileRows),
            static_cast<uint32_t>(rowElements * sizeof(float)), 0,
            static_cast<uint32_t>((nheads_ - headCount) *
                                  kHeadDim * sizeof(float)),
            0};
        AscendC::DataCopyPad(dzGm_[publicOffset], yPre, dzParams);
        AscendC::DataCopyExtParams partialParams{
            1, static_cast<uint32_t>(kHeadDim * sizeof(float)), 0, 0, 0};
        if (computeD_ != 0 &&
            tile == kChunk / kHeadBlockTileRows - 1) {
            for (int64_t localHead = 0; localHead < headCount; ++localHead) {
                const int64_t head = headBegin + localHead;
                const int64_t partialOffset =
                    ((head * batch_ + batch) * nchunks_ + chunk) * kHeadDim;
                AscendC::DataCopyPad(
                    dDPartialGm_[partialOffset],
                    dDAcc[localHead * kHeadDim], partialParams);
            }
        }
        // A core loops over many chunks and reuses the same UB tiles.  The
        // next iteration starts with MTE2 writes, so protect those writes
        // until all three MTE3 consumers above have finished reading UB.
        // MTE3_V is insufficient here: it orders a later Vector instruction,
        // not the next iteration's MTE2 overwrite.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
        }
    }

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> doutBuf_, zBuf_, yPreBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_, sigmoidBuf_, tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dDAccBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gyHalfBuf_;
    AscendC::GlobalTensor<float> doutGm_, zGm_, yPreFloatGm_, xGm_;
    AscendC::GlobalTensor<half> yPreHalfGm_, gyHeadGm_;
    AscendC::GlobalTensor<float> dzGm_, dDPartialGm_;
    event_t mte2ToVEvent_;
    event_t vToMte3Event_;
    event_t mte3ToMte2Event_;
    event_t mte3ToVEvent_;
    int64_t batch_, seqlen_, nheads_, nchunks_, usedCoreNum_, computeD_;
    int64_t yPreIsHalf_;
};

// H256 training dispatches the no-dD path and stores y_pre in FP16.  Its
// single-buffer implementation serializes MTE2, Vector and MTE3 for all eight
// token tiles.  This specialization uses two queue slots for every external
// tensor while keeping the three FP32 compute workspaces single-buffered.
// The resulting 176 KiB UB footprint fits Ascend 910B3's 192 KiB UB.
class KernelMamba2SsdBwdGateNoDdPipeline {
public:
    __aicore__ inline void Init(
        GM_ADDR dout, GM_ADDR z, GM_ADDR yPre, GM_ADDR gyHead, GM_ADDR dz,
        int64_t batch, int64_t seqlen, int64_t nheads,
        int64_t nchunks, int64_t usedCoreNum, AscendC::TPipe *pipe)
    {
        doutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dout));
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z));
        yPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(yPre));
        gyHeadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gyHead));
        dzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dz));
        batch_ = batch;
        seqlen_ = seqlen;
        nheads_ = nheads;
        nchunks_ = nchunks;
        usedCoreNum_ = usedCoreNum;

        pipe->InitBuffer(doutIn_, 2, kHeadBlockTileElements * sizeof(float));
        pipe->InitBuffer(zIn_, 2, kHeadBlockTileElements * sizeof(float));
        pipe->InitBuffer(yPreIn_, 2, kHeadBlockTileElements * sizeof(half));
        pipe->InitBuffer(dzOut_, 2, kHeadBlockTileElements * sizeof(float));
        pipe->InitBuffer(gyOut_, 2, kHeadBlockTileElements * sizeof(half));
        pipe->InitBuffer(sigmoidBuf_, kHeadBlockTileElements * sizeof(float));
        pipe->InitBuffer(tmpBuf_, kHeadBlockTileElements * sizeof(float));
        pipe->InitBuffer(scratchBuf_, kHeadBlockTileElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t nHeadBlocks =
            (nheads_ + kHeadBlock - 1) / kHeadBlock;
        const int64_t taskCount = batch_ * nHeadBlocks * nchunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            const int64_t chunk = task % nchunks_;
            const int64_t headBlock = (task / nchunks_) % nHeadBlocks;
            const int64_t batch = task / (nchunks_ * nHeadBlocks);
            ProcessChunk(batch, headBlock, chunk);
        }
    }

private:
    __aicore__ inline void ProcessChunk(
        int64_t batch, int64_t headBlock, int64_t chunk)
    {
        const int64_t headBegin = headBlock * kHeadBlock;
        const int64_t remainingHeads = nheads_ - headBegin;
        const int64_t headCount =
            remainingHeads < kHeadBlock ? remainingHeads : kHeadBlock;
        for (int64_t tile = 0;
             tile < kChunk / kHeadBlockTileRows; ++tile) {
            CopyIn(batch, headBegin, headCount, chunk, tile);
            Compute(headCount);
            CopyOut(batch, headBegin, headCount, chunk, tile);
        }
    }

    __aicore__ inline void CopyIn(
        int64_t batch, int64_t headBegin, int64_t headCount,
        int64_t chunk, int64_t tile)
    {
        const int64_t rowBegin = tile * kHeadBlockTileRows;
        const int64_t firstToken = chunk * kChunk;
        const int64_t publicOffset =
            ((batch * seqlen_ + firstToken + rowBegin) * nheads_ +
             headBegin) * kHeadDim;
        const int64_t rowElements = headCount * kHeadDim;
        AscendC::DataCopyExtParams fp32Params{
            static_cast<uint16_t>(kHeadBlockTileRows),
            static_cast<uint32_t>(rowElements * sizeof(float)),
            static_cast<uint32_t>((nheads_ - headCount) * kHeadDim *
                                  sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> noPadFloat{
            false, 0, 0, 0.0f};
        auto dout = doutIn_.AllocTensor<float>();
        auto z = zIn_.AllocTensor<float>();
        AscendC::DataCopyPad(
            dout, doutGm_[publicOffset], fp32Params, noPadFloat);
        AscendC::DataCopyPad(z, zGm_[publicOffset], fp32Params, noPadFloat);
        doutIn_.EnQue(dout);
        zIn_.EnQue(z);

        AscendC::DataCopyExtParams fp16Params{
            static_cast<uint16_t>(kHeadBlockTileRows),
            static_cast<uint32_t>(rowElements * sizeof(half)),
            static_cast<uint32_t>((nheads_ - headCount) * kHeadDim *
                                  sizeof(half)),
            0, 0};
        AscendC::DataCopyPadExtParams<half> noPadHalf{
            false, 0, 0, static_cast<half>(0)};
        auto yPre = yPreIn_.AllocTensor<half>();
        AscendC::DataCopyPad(
            yPre, yPreGm_[publicOffset], fp16Params, noPadHalf);
        yPreIn_.EnQue(yPre);
    }

    __aicore__ inline void Compute(int64_t headCount)
    {
        const int64_t activeElements =
            kHeadBlockTileRows * headCount * kHeadDim;
        auto dout = doutIn_.DeQue<float>();
        auto z = zIn_.DeQue<float>();
        auto yPreHalf = yPreIn_.DeQue<half>();
        auto dz = dzOut_.AllocTensor<float>();
        auto gy = gyOut_.AllocTensor<half>();
        auto sigmoid = sigmoidBuf_.Get<float>();
        auto tmp = tmpBuf_.Get<float>();
        auto scratch = scratchBuf_.Get<float>();

        AscendC::Cast(
            dz, yPreHalf, AscendC::RoundMode::CAST_NONE, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(sigmoid, z, -1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(sigmoid, sigmoid, 80.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(sigmoid, sigmoid, -80.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(sigmoid, sigmoid, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(sigmoid, sigmoid, 1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scratch, sigmoid, 1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Reciprocal(sigmoid, sigmoid, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        for (int iteration = 0; iteration < 2; ++iteration) {
            AscendC::Mul(tmp, scratch, sigmoid, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(tmp, tmp, -1.0f, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmp, tmp, 2.0f, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(sigmoid, sigmoid, tmp, activeElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Muls(tmp, sigmoid, -1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tmp, tmp, 1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp, tmp, z, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tmp, tmp, 1.0f, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dz, dz, dout, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dz, dz, sigmoid, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dz, dz, tmp, activeElements);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(tmp, z, sigmoid, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp, tmp, dout, activeElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            gy, tmp, AscendC::RoundMode::CAST_RINT, activeElements);

        doutIn_.FreeTensor(dout);
        zIn_.FreeTensor(z);
        yPreIn_.FreeTensor(yPreHalf);
        dzOut_.EnQue(dz);
        gyOut_.EnQue(gy);
    }

    __aicore__ inline void CopyOut(
        int64_t batch, int64_t headBegin, int64_t headCount,
        int64_t chunk, int64_t tile)
    {
        const int64_t rowBegin = tile * kHeadBlockTileRows;
        const int64_t firstToken = chunk * kChunk;
        const int64_t publicOffset =
            ((batch * seqlen_ + firstToken + rowBegin) * nheads_ +
             headBegin) * kHeadDim;
        const int64_t rowElements = headCount * kHeadDim;
        auto dz = dzOut_.DeQue<float>();
        auto gy = gyOut_.DeQue<half>();
        AscendC::DataCopyExtParams dzParams{
            static_cast<uint16_t>(kHeadBlockTileRows),
            static_cast<uint32_t>(rowElements * sizeof(float)), 0,
            static_cast<uint32_t>((nheads_ - headCount) * kHeadDim *
                                  sizeof(float)),
            0};
        AscendC::DataCopyPad(dzGm_[publicOffset], dz, dzParams);

        AscendC::DataCopyExtParams gyParams{
            static_cast<uint16_t>(kHeadBlockTileRows),
            static_cast<uint32_t>(kHeadDim * sizeof(half)),
            static_cast<uint32_t>((headCount - 1) * kHeadDim *
                                  sizeof(half) / 32),
            0, 0};
        for (int64_t localHead = 0; localHead < headCount; ++localHead) {
            const int64_t head = headBegin + localHead;
            const int64_t headOffset =
                ((((batch * nheads_ + head) * nchunks_ + chunk) * kChunk +
                  rowBegin) * kHeadDim);
            AscendC::DataCopyPad(
                gyHeadGm_[headOffset], gy[localHead * kHeadDim], gyParams);
        }
        dzOut_.FreeTensor(dz);
        gyOut_.FreeTensor(gy);
    }

    AscendC::TQue<AscendC::TPosition::VECIN, 2> doutIn_, zIn_, yPreIn_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> dzOut_, gyOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sigmoidBuf_, tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuf_;
    AscendC::GlobalTensor<float> doutGm_, zGm_, dzGm_;
    AscendC::GlobalTensor<half> yPreGm_, gyHeadGm_;
    int64_t batch_, seqlen_, nheads_, nchunks_, usedCoreNum_;
};

class KernelMamba2SsdBwdGateReduce {
public:
    __aicore__ inline void Init(
        GM_ADDR dDPartial, GM_ADDR dD, int64_t batch, int64_t nheads,
        int64_t nchunks, int64_t usedCoreNum)
    {
        dDPartialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dDPartial));
        dDGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dD));
        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        usedCoreNum_ = usedCoreNum;
        pipe_.InitBuffer(partialBuf_, kTileElements * sizeof(float));
        pipe_.InitBuffer(sumBuf_, kHeadDim * sizeof(float));
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
        const int64_t rows = batch_ * nchunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t head = block; head < nheads_; head += usedCoreNum_) {
            auto partial = partialBuf_.Get<float>();
            auto sum = sumBuf_.Get<float>();
            AscendC::Duplicate(sum, 0.0f, kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t rowBegin = 0; rowBegin < rows;
                 rowBegin += kTileRows) {
                const int64_t tileRows =
                    (rows - rowBegin < kTileRows) ? rows - rowBegin : kTileRows;
                if (tileRows < kTileRows) {
                    AscendC::Duplicate(partial, 0.0f, kTileElements);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Event_);
                AscendC::DataCopyExtParams inputParams{
                    1,
                    static_cast<uint32_t>(tileRows * kHeadDim * sizeof(float)),
                    0, 0, 0};
                AscendC::DataCopyPadExtParams<float> noPad{
                    false, 0, 0, 0.0f};
                AscendC::DataCopyPad(
                    partial,
                    dDPartialGm_[(head * rows + rowBegin) * kHeadDim],
                    inputParams, noPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToVEvent_);

                // A short final tile is zero padded above, so the same fixed
                // 32-row tree also covers arbitrary B*K values.
                for (int64_t active = kTileRows / 2; active >= 1; active >>= 1) {
                    AscendC::Add(
                        partial, partial, partial[active * kHeadDim],
                        active * kHeadDim);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Add(sum, sum, partial, kHeadDim);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3Event_);
            AscendC::DataCopyExtParams outputParams{
                1, static_cast<uint32_t>(kHeadDim * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPad(dDGm_[head * kHeadDim], sum, outputParams);
            // When H exceeds the AIV count, one core immediately reuses
            // ``sum`` for a later head.  V_MTE3 only orders the producer
            // before the store starts; without the reverse dependency the
            // following Duplicate can overwrite UB while MTE3 is still
            // reading it, corrupting dD in an allocation/timing-dependent
            // way.  This was hidden at H<=AIV because each core owned only
            // one head.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVEvent_);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuf_, sumBuf_;
    AscendC::GlobalTensor<float> dDPartialGm_, dDGm_;
    event_t vToMte2Event_, mte2ToVEvent_, vToMte3Event_, mte3ToVEvent_;
    int64_t batch_, nheads_, nchunks_, usedCoreNum_;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_bwd_gate(
    GM_ADDR dout, GM_ADDR z, GM_ADDR yPre, GM_ADDR x,
    GM_ADDR gyHead, GM_ADDR dz, GM_ADDR dDPartial,
    int64_t batch, int64_t seqlen, int64_t nheads,
    int64_t nchunks, int64_t usedCoreNum, int64_t computeD,
    int64_t yPreIsHalf)
{
    AscendC::TPipe pipe;
    if (computeD == 0 && yPreIsHalf != 0) {
        KernelMamba2SsdBwdGateNoDdPipeline op;
        op.Init(dout, z, yPre, gyHead, dz, batch, seqlen, nheads,
                nchunks, usedCoreNum, &pipe);
        op.Process();
        return;
    }
    KernelMamba2SsdBwdGateHeadBlock op;
    op.Init(dout, z, yPre, x, gyHead, dz, dDPartial,
            batch, seqlen, nheads, nchunks, usedCoreNum, computeD,
            yPreIsHalf, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void mamba2_ssd_bwd_gate_reduce(
    GM_ADDR dDPartial, GM_ADDR dD, int64_t batch, int64_t nheads,
    int64_t nchunks, int64_t usedCoreNum)
{
    KernelMamba2SsdBwdGateReduce op;
    op.Init(dDPartial, dD, batch, nheads, nchunks, usedCoreNum);
    op.Process();
}
