// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"
#include "lib/pad/broadcast.h"

class KernelMamba2SsdDtBwd {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR dXdt, GM_ADDR gCs, GM_ADDR gCsOff,
        GM_ADDR gCsLast, GM_ADDR dt, GM_ADDR A,
        GM_ADDR dtBias, GM_ADDR gyHead, GM_ADDR dMatrix,
        GM_ADDR dx, GM_ADDR ddt, GM_ADDR dAPartial,
        GM_ADDR dBiasPartial, GM_ADDR dDPartial,
        int64_t batch, int64_t seqlen, int64_t nheads, int64_t nchunks,
        int64_t chunkSize, int64_t headdim, int64_t hasBias,
        int64_t dtSoftplus, int64_t hasD,
        float dtLimitMin, float dtLimitMax,
        int64_t usedCoreNum, int64_t headBlock, int64_t computeDD,
        int64_t ngroups, int64_t headsPerGroup, int64_t groupedGcs,
        int64_t dXdtIsHalf)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        dXdtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dXdt));
        dXdtHalfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dXdt));
        gCsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gCs));
        gCsOffGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(gCsOff));
        gCsLastGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(gCsLast));
        dtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dt));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias));
        gyHeadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gyHead));
        dMatrixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dMatrix));
        dxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dx));
        ddtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ddt));
        dAPartialGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAPartial));
        dBiasPartialGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dBiasPartial));
        dDPartialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dDPartial));
        batch_ = batch;
        seqlen_ = seqlen;
        nheads_ = nheads;
        nchunks_ = nchunks;
        chunkSize_ = chunkSize;
        headdim_ = headdim;
        hasBias_ = hasBias;
        dtSoftplus_ = dtSoftplus;
        hasD_ = hasD;
        dtLimitMin_ = dtLimitMin;
        dtLimitMax_ = dtLimitMax;
        usedCoreNum_ = usedCoreNum;
        headBlock_ = headBlock;
        computeDD_ = computeDD;
        ngroups_ = ngroups;
        headsPerGroup_ = headsPerGroup;
        groupedGcs_ = groupedGcs;
        dXdtIsHalf_ = dXdtIsHalf;
        const int64_t matrixElements = chunkSize_ * headdim_;
        pipe_.InitBuffer(xBuf_, matrixElements * sizeof(float));
        pipe_.InitBuffer(dXdtBuf_, matrixElements * sizeof(float));
        pipe_.InitBuffer(gCsBuf_, chunkSize_ * sizeof(float));
        pipe_.InitBuffer(
            gCsOffBuf_,
            groupedGcs_ == 1
                ? chunkSize_ * headsPerGroup_ * sizeof(float)
                : (groupedGcs_ == 2
                       ? chunkSize_ * sizeof(float) : 32));
        pipe_.InitBuffer(
            gCsOffOffsetsBuf_,
            groupedGcs_ == 1 ? chunkSize_ * sizeof(uint32_t) : 32);
        pipe_.InitBuffer(gCsLastBuf_, 32);
        pipe_.InitBuffer(dtPaddedBuf_, chunkSize_ * 8 * sizeof(float));
        pipe_.InitBuffer(qCompactBuf_, chunkSize_ * sizeof(float));
        pipe_.InitBuffer(dtGatherOffsetsBuf_,
                         chunkSize_ * sizeof(uint32_t));
        pipe_.InitBuffer(rowGatherOffsetsBuf_,
                         chunkSize_ * sizeof(uint32_t));
        pipe_.InitBuffer(reverseScanBuf_, chunkSize_ * sizeof(float));
        pipe_.InitBuffer(reverseScanOffsetsBuf_,
                         6 * 64 * sizeof(uint32_t));
        // CompareScalar writes one predicate bit per input element.  Reserve
        // one aligned data block for both supported chunk sizes (<=128).
        pipe_.InitBuffer(compareMaskBuf_, 32);
        pipe_.InitBuffer(broadcastTmpBuf_,
                         2 * matrixElements * sizeof(uint8_t));
        const int64_t compactElements =
            chunkSize_ > headdim_ ? chunkSize_ : headdim_;
        pipe_.InitBuffer(scalarBuf_, compactElements * sizeof(float));
        pipe_.InitBuffer(dBuf_, compactElements * sizeof(float));
        pipe_.InitBuffer(dxQueue_, 1, matrixElements * sizeof(float));
        pipe_.InitBuffer(ddtQueue_, 1, chunkSize_ * 8 * sizeof(float));
        const int64_t headBlockMatrixBytes = headBlock_ > 1
            ? headBlock_ * matrixElements * sizeof(float)
            : 32;
        const int64_t headBlockDtBytes = headBlock_ > 1
            ? chunkSize_ * 8 * sizeof(float)
            : 32;
        pipe_.InitBuffer(xHeadBlockBuf_, headBlockMatrixBytes);
        pipe_.InitBuffer(dxHeadBlockBuf_, headBlockMatrixBytes);
        pipe_.InitBuffer(dtHeadBlockBuf_, headBlockDtBytes);
        pipe_.InitBuffer(headScalarBuf_, 64);
        pipe_.InitBuffer(dHeadBlockBuf_,
                         headBlock_ > 1
                             ? headBlock_ * headdim_ * sizeof(float)
                             : 32);
        pipe_.InitBuffer(
            gyHalfBuf_,
            (hasD_ != 0 || dXdtIsHalf_ != 0)
                ? matrixElements * sizeof(half) : 32);
        pipe_.InitBuffer(
            dDPartialQueue_, 1,
            computeDD_ != 0 ? headdim_ * sizeof(float) : 32);

        // DataCopyPad stores each strided dt scalar in one 32-byte UB block.
        // Build the byte offsets once per core so the per-task hot path can
        // compact all 64 values with one vector Gather instead of 64 scalar
        // GetValue operations.
        auto gatherOffsets = dtGatherOffsetsBuf_.Get<uint32_t>();
        auto rowGatherOffsets = rowGatherOffsetsBuf_.Get<uint32_t>();
        auto gCsOffOffsets = gCsOffOffsetsBuf_.Get<uint32_t>();
        for (int64_t token = 0; token < chunkSize_; ++token) {
            gatherOffsets.SetValue(
                token, static_cast<uint32_t>(token * 32));
            rowGatherOffsets.SetValue(
                token,
                static_cast<uint32_t>(token * headdim_ * sizeof(float)));
            if (groupedGcs_ == 1) {
                gCsOffOffsets.SetValue(
                    token,
                    static_cast<uint32_t>(
                        token * headsPerGroup_ * sizeof(float)));
            }
        }
        if (chunkSize_ == 64) {
            auto reverseScanOffsets =
                reverseScanOffsetsBuf_.Get<uint32_t>();
            int64_t stage = 0;
            for (int64_t offset = 1; offset < 64;
                 offset <<= 1, ++stage) {
                for (int64_t token = 0; token < 64 - offset; ++token) {
                    reverseScanOffsets.SetValue(
                        stage * 64 + token,
                        static_cast<uint32_t>(
                            (token + offset) * sizeof(float)));
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        if (hasD_ != 0 && headBlock_ == 2) {
            ProcessHeadBlocks();
        } else {
            ProcessChunks();
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

    __aicore__ inline float ScalarLog(float value)
    {
        union FloatBits { uint32_t bits; float value; } bits;
        bits.value = value;
        const int32_t exponent =
            static_cast<int32_t>((bits.bits >> 23) & 0xff) - 127;
        bits.bits = (bits.bits & 0x007fffffU) | 0x3f800000U;
        const float y = (bits.value - 1.0f) / (bits.value + 1.0f);
        const float y2 = y * y;
        float term = y;
        float series = term;
        term *= y2; series += term / 3.0f;
        term *= y2; series += term / 5.0f;
        term *= y2; series += term / 7.0f;
        term *= y2; series += term / 9.0f;
        term *= y2; series += term / 11.0f;
        return 2.0f * series + static_cast<float>(exponent) * 0.6931471805599453f;
    }

    __aicore__ inline float Softplus(float value)
    {
        if (value > 20.0f) return value;
        if (value < -9.0f) return ScalarExp(value);
        if (value >= 0.0f) return value + ScalarLog(1.0f + ScalarExp(-value));
        return ScalarLog(1.0f + ScalarExp(value));
    }

    __aicore__ inline float Sigmoid(float value)
    {
        if (value >= 0.0f) {
            const float expNeg = ScalarExp(-value);
            return 1.0f / (1.0f + expNeg);
        }
        const float expPos = ScalarExp(value);
        return expPos / (1.0f + expPos);
    }

    __aicore__ inline void StoreChunkScalar(
        const AscendC::GlobalTensor<float> &dst, int64_t offset, float value)
    {
        auto output = ddtQueue_.AllocTensor<float>();
        output.SetValue(0, value);
        ddtQueue_.EnQue(output);
        output = ddtQueue_.DeQue<float>();
        AscendC::DataCopyExtParams params{1, sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(dst[offset], output, params);
        ddtQueue_.FreeTensor(output);
    }

    __aicore__ inline void ProcessChunks()
    {
        const int64_t taskCount = batch_ * nheads_ * nchunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            const int64_t chunk = task % nchunks_;
            const int64_t head = (task / nchunks_) % nheads_;
            const int64_t batch = task / (nchunks_ * nheads_);
            ProcessChunk(task, batch, head, chunk);
        }
    }

    __aicore__ inline void ProcessHeadBlocks()
    {
        constexpr int64_t kHeadBlock = 2;
        const int64_t headBlockCount =
            (nheads_ + kHeadBlock - 1) / kHeadBlock;
        const int64_t taskCount = batch_ * headBlockCount;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            const int64_t batch = task / headBlockCount;
            const int64_t headBase = (task % headBlockCount) * kHeadBlock;
            const int64_t validHeads =
                headBase + kHeadBlock <= nheads_
                    ? kHeadBlock : nheads_ - headBase;
            LoadHeadConstants(headBase, validHeads);
            for (int64_t chunk = 0; chunk < nchunks_; ++chunk) {
                LoadHeadBlockChunk(batch, headBase, chunk, validHeads);
                for (int64_t localHead = 0;
                     localHead < validHeads; ++localHead) {
                    const int64_t head = headBase + localHead;
                    const int64_t chunkTask =
                        (batch * nheads_ + head) * nchunks_ + chunk;
                    ProcessChunk(
                        chunkTask, batch, head, chunk,
                        localHead, validHeads);
                }
                StoreHeadBlockDx(batch, headBase, chunk, validHeads);
            }
        }
    }

    __aicore__ inline void LoadHeadConstants(
        int64_t headBase, int64_t validHeads)
    {
        auto headScalars = headScalarBuf_.Get<float>();
        auto dBlock = dHeadBlockBuf_.Get<float>();
        AscendC::DataCopyExtParams scalarParams{
            1, static_cast<uint32_t>(validHeads * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> scalarPad{
            true, 0, static_cast<uint8_t>(8 - validHeads), 0.0f};
        AscendC::DataCopyPad(
            headScalars, aGm_[headBase], scalarParams, scalarPad);
        if (hasBias_ != 0) {
            AscendC::DataCopyPad(
                headScalars[8], biasGm_[headBase], scalarParams, scalarPad);
        }
        AscendC::DataCopyExtParams dParams{
            1,
            static_cast<uint32_t>(
                validHeads * headdim_ * sizeof(float)),
            0, 0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(
            dBlock, dMatrixGm_[headBase * headdim_], dParams, noPad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadHeadBlockChunk(
        int64_t batch, int64_t headBase, int64_t chunk,
        int64_t validHeads)
    {
        const int64_t firstToken = chunk * chunkSize_;
        const int64_t xOffset =
            ((batch * seqlen_ + firstToken) * nheads_ + headBase) * headdim_;
        const int64_t dtOffset =
            (batch * seqlen_ + firstToken) * nheads_ + headBase;
        auto xBlock = xHeadBlockBuf_.Get<float>();
        auto dtBlock = dtHeadBlockBuf_.Get<float>();
        AscendC::DataCopyExtParams xParams{
            static_cast<uint16_t>(chunkSize_),
            static_cast<uint32_t>(
                validHeads * headdim_ * sizeof(float)),
            static_cast<uint32_t>(
                (nheads_ - validHeads) * headdim_ * sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(xBlock, xGm_[xOffset], xParams, noPad);
        AscendC::DataCopyExtParams dtParams{
            static_cast<uint16_t>(chunkSize_),
            static_cast<uint32_t>(validHeads * sizeof(float)),
            static_cast<uint32_t>(
                (nheads_ - validHeads) * sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> dtPad{
            true, 0, static_cast<uint8_t>(8 - validHeads), 0.0f};
        AscendC::DataCopyPad(dtBlock, dtGm_[dtOffset], dtParams, dtPad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreHeadBlockDx(
        int64_t batch, int64_t headBase, int64_t chunk,
        int64_t validHeads)
    {
        const int64_t firstToken = chunk * chunkSize_;
        const int64_t xOffset =
            ((batch * seqlen_ + firstToken) * nheads_ + headBase) * headdim_;
        auto dxBlock = dxHeadBlockBuf_.Get<float>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams dxParams{
            static_cast<uint16_t>(chunkSize_),
            static_cast<uint32_t>(
                validHeads * headdim_ * sizeof(float)),
            0,
            static_cast<uint32_t>(
                (nheads_ - validHeads) * headdim_ * sizeof(float)),
            0};
        AscendC::DataCopyPad(dxGm_[xOffset], dxBlock, dxParams);
    }

    __aicore__ inline void ProcessChunk(
        int64_t task, int64_t batch, int64_t head, int64_t chunk,
        int64_t localHead = -1, int64_t validHeads = 1)
    {
        const int64_t matrixElements = chunkSize_ * headdim_;
        const int64_t firstToken = chunk * chunkSize_;
        const int64_t xOffset =
            ((batch * seqlen_ + firstToken) * nheads_ + head) * headdim_;
        const int64_t dtOffset = (batch * seqlen_ + firstToken) * nheads_ + head;
        const int64_t chunkOffset = task * matrixElements;

        auto xLocal = xBuf_.Get<float>();
        auto dXdtLocal = dXdtBuf_.Get<float>();
        auto gCsLocal = gCsBuf_.Get<float>();
        auto dtPadded = dtPaddedBuf_.Get<float>();
        auto qCompact = qCompactBuf_.Get<float>();
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        AscendC::DataCopyExtParams xParams{
            static_cast<uint16_t>(chunkSize_),
            static_cast<uint32_t>(headdim_ * sizeof(float)),
            static_cast<uint32_t>((nheads_ - 1) * headdim_ * sizeof(float)), 0, 0};
        if (localHead >= 0) {
            auto xBlock = xHeadBlockBuf_.Get<float>();
            AscendC::DataCopyParams unpackParams{
                static_cast<uint16_t>(chunkSize_),
                static_cast<uint16_t>(
                    headdim_ * sizeof(float) /
                    AscendC::DEFAULT_C0_SIZE),
                static_cast<uint16_t>(
                    (validHeads - 1) * headdim_ * sizeof(float) /
                    AscendC::DEFAULT_C0_SIZE),
                0};
            AscendC::DataCopy(
                xLocal, xBlock[localHead * headdim_], unpackParams);
        } else {
            AscendC::DataCopyPad(xLocal, xGm_[xOffset], xParams, noPad);
        }
        AscendC::DataCopyExtParams matrixParams{
            1, static_cast<uint32_t>(matrixElements * sizeof(float)), 0, 0, 0};
        if (dXdtIsHalf_ != 0) {
            auto dXdtHalf = gyHalfBuf_.Get<half>();
            AscendC::DataCopyExtParams halfMatrixParams{
                1,
                static_cast<uint32_t>(matrixElements * sizeof(half)),
                0, 0, 0};
            AscendC::DataCopyPadExtParams<half> noHalfPad{
                false, 0, 0, static_cast<half>(0)};
            AscendC::DataCopyPad(
                dXdtHalf, dXdtHalfGm_[chunkOffset],
                halfMatrixParams, noHalfPad);
        } else {
            AscendC::DataCopyPad(
                dXdtLocal, dXdtGm_[chunkOffset], matrixParams, noPad);
        }
        AscendC::DataCopyExtParams vectorParams{
            1, static_cast<uint32_t>(chunkSize_ * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(gCsLocal, gCsGm_[task * chunkSize_], vectorParams, noPad);
        if (groupedGcs_ == 1) {
            const int64_t group = head / headsPerGroup_;
            const int64_t headInGroup = head - group * headsPerGroup_;
            const int64_t groupedOffset =
                (((batch * nchunks_ + chunk) * ngroups_ + group) *
                 chunkSize_) * headsPerGroup_;
            auto gCsOffTile = gCsOffBuf_.Get<float>();
            AscendC::DataCopyExtParams groupedParams{
                1,
                static_cast<uint32_t>(
                    chunkSize_ * headsPerGroup_ * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyPad(
                gCsOffTile, gCsOffGm_[groupedOffset], groupedParams, noPad);
            auto gCsLast = gCsLastBuf_.Get<float>();
            AscendC::DataCopyExtParams lastParams{
                1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(
                gCsLast, gCsLastGm_[task], lastParams, noPad);
            AscendC::PipeBarrier<PIPE_ALL>();
            // Off produces [T,R].  Extract this head's column in UB and add
            // it to the head-major Diag contribution.  The chunk-state term
            // contributes only to the last token, matching the old Cat path.
            auto gCsOffCompact = reverseScanBuf_.Get<float>();
            AscendC::Gather(
                gCsOffCompact, gCsOffTile[headInGroup],
                gCsOffOffsetsBuf_.Get<uint32_t>(), 0U,
                static_cast<uint32_t>(chunkSize_));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(
                gCsLocal, gCsLocal, gCsOffCompact, chunkSize_);
            AscendC::PipeBarrier<PIPE_ALL>();
            gCsLocal.SetValue(
                chunkSize_ - 1,
                gCsLocal.GetValue(chunkSize_ - 1) + gCsLast.GetValue(0));
        } else if (groupedGcs_ == 2) {
            auto gCsOffTile = gCsOffBuf_.Get<float>();
            auto gCsLast = gCsLastBuf_.Get<float>();
            AscendC::DataCopyExtParams offParams{
                1, static_cast<uint32_t>(chunkSize_ * sizeof(float)),
                0, 0, 0};
            AscendC::DataCopyExtParams lastParams{
                1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(
                gCsOffTile, gCsOffGm_[task * chunkSize_],
                offParams, noPad);
            AscendC::DataCopyPad(
                gCsLast, gCsLastGm_[task], lastParams, noPad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(
                gCsLocal, gCsLocal, gCsOffTile, chunkSize_);
            AscendC::PipeBarrier<PIPE_ALL>();
            gCsLocal.SetValue(
                chunkSize_ - 1,
                gCsLocal.GetValue(chunkSize_ - 1) + gCsLast.GetValue(0));
        } else if (groupedGcs_ == 3) {
            // gCs already contains Diag + Off.  Pull only the state-passing
            // scalar needed by this chunk; unlike mode 2 this does not read
            // another full T-vector through MTE2.
            auto gCsLast = gCsLastBuf_.Get<float>();
            AscendC::DataCopyExtParams lastParams{
                1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(
                gCsLast, gCsLastGm_[task], lastParams, noPad);
            AscendC::PipeBarrier<PIPE_ALL>();
            gCsLocal.SetValue(
                chunkSize_ - 1,
                gCsLocal.GetValue(chunkSize_ - 1) + gCsLast.GetValue(0));
        }
        AscendC::DataCopyExtParams dtParams{
            static_cast<uint16_t>(chunkSize_), sizeof(float),
            static_cast<uint32_t>((nheads_ - 1) * sizeof(float)), 0, 0};
        AscendC::DataCopyPadExtParams<float> dtPad{true, 0, 7, 0.0f};
        if (localHead < 0) {
            AscendC::DataCopyPad(
                dtPadded, dtGm_[dtOffset], dtParams, dtPad);
        }

        auto scalar = scalarBuf_.Get<float>();
        AscendC::DataCopyExtParams scalarParams{1, sizeof(float), 0, 0, 0};
        if (localHead < 0) {
            AscendC::DataCopyPad(scalar, aGm_[head], scalarParams, noPad);
            if (hasBias_ != 0) {
                AscendC::DataCopyPad(
                    scalar[8], biasGm_[head], scalarParams, noPad);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        auto headScalars = headScalarBuf_.Get<float>();
        const float aValue = localHead >= 0
            ? headScalars.GetValue(localHead)
            : scalar.GetValue(0);
        const float bias = hasBias_ != 0
            ? (localHead >= 0
                   ? headScalars.GetValue(8 + localHead)
                   : scalar.GetValue(8))
            : 0.0f;
        if (dXdtIsHalf_ != 0) {
            AscendC::Cast(
                dXdtLocal, gyHalfBuf_.Get<half>(),
                AscendC::RoundMode::CAST_NONE, matrixElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // gy is independent of the reverse scan, softplus derivative and
        // direct dx branch below.  Once an FP16 dXdt producer has been cast
        // out of the shared half buffer, start the next MTE2 transfer so it
        // overlaps that Vector work instead of extending the D epilogue.
        if (hasD_ != 0) {
            if (dXdtIsHalf_ != 0) {
                event_t vToGyLoad = static_cast<event_t>(
                    pipe_.FetchEventID(AscendC::HardEvent::V_MTE2));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vToGyLoad);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vToGyLoad);
            }
            AscendC::DataCopyExtParams gyParams{
                1,
                static_cast<uint32_t>(matrixElements * sizeof(half)),
                0, 0, 0};
            AscendC::DataCopyPadExtParams<half> noHalfPad{
                false, 0, 0, static_cast<half>(0)};
            AscendC::DataCopyPad(
                gyHalfBuf_.Get<half>(), gyHeadGm_[chunkOffset],
                gyParams, noHalfPad);
        }

        auto dxLocal = dxQueue_.AllocTensor<float>();
        auto ddtLocal = ddtQueue_.AllocTensor<float>();

        // Vectorize the O(T * P) direct-gradient path.  Reuse xLocal as the
        // product buffer: after reducing each row, x is no longer needed by
        // this task.  WholeReduceSum handles one 64-element FP32 repeat, so
        // P=128 is reduced as two independent halves and combined below.
        AscendC::Mul(xLocal, dXdtLocal, xLocal, matrixElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(
            xLocal, xLocal, 64, static_cast<int32_t>(chunkSize_),
            static_cast<int32_t>(headdim_), 1,
            static_cast<int32_t>(headdim_ / 8));
        if (headdim_ == 128) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(
                xLocal[64], xLocal[64], 64,
                static_cast<int32_t>(chunkSize_),
                static_cast<int32_t>(headdim_), 1,
                static_cast<int32_t>(headdim_ / 8));
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        float running = 0.0f;
        float dAPartial = 0.0f;
        float dBiasPartial = 0.0f;

        // Compute q=clamp(softplus(dt+bias)) and the softplus derivative as
        // vectors.  The previous implementation evaluated polynomial scalar
        // exp/log/sigmoid inside the 64-step reverse scan, which occupied the
        // AIV scalar pipe for most of this kernel on large shapes.
        auto qPreCompact = scalarBuf_.Get<float>();
        auto sigmoidCompact = dBuf_.Get<float>();
        auto dtGatherSource = localHead >= 0
            ? dtHeadBlockBuf_.Get<float>()[localHead]
            : dtPadded;
        AscendC::Gather(
            qPreCompact, dtGatherSource,
            dtGatherOffsetsBuf_.Get<uint32_t>(),
            0U, static_cast<uint32_t>(chunkSize_));
        AscendC::PipeBarrier<PIPE_V>();
        if (hasBias_ != 0) {
            AscendC::Adds(qPreCompact, qPreCompact, bias, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (dtSoftplus_ != 0) {
            // sigmoid(u) = 1 / (1 + exp(-u)).  Bound the exponent and apply
            // two Newton refinements because Reciprocal is an ISA estimate.
            auto denominator = dtPadded[chunkSize_];
            AscendC::Muls(sigmoidCompact, qPreCompact, -1.0f, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(sigmoidCompact, sigmoidCompact, 80.0f, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(sigmoidCompact, sigmoidCompact, -80.0f, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(sigmoidCompact, sigmoidCompact, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(sigmoidCompact, sigmoidCompact, 1.0f, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(denominator, sigmoidCompact, 1.0f, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Reciprocal(
                sigmoidCompact, sigmoidCompact, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            for (int iteration = 0; iteration < 2; ++iteration) {
                AscendC::Mul(qCompact, denominator, sigmoidCompact,
                             chunkSize_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(qCompact, qCompact, -1.0f, chunkSize_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(qCompact, qCompact, 2.0f, chunkSize_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(sigmoidCompact, sigmoidCompact, qCompact,
                             chunkSize_);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // Stable softplus(u) = max(u, 0) + log(1 + exp(-abs(u))).
            auto softplusTmp = dtPadded;
            AscendC::Abs(softplusTmp, qPreCompact, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(softplusTmp, softplusTmp, -1.0f, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(softplusTmp, softplusTmp, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(softplusTmp, softplusTmp, 1.0f, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Ln(softplusTmp, softplusTmp, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(qCompact, qPreCompact, 0.0f, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(qPreCompact, qCompact, softplusTmp, chunkSize_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Maxs(qCompact, qPreCompact, dtLimitMin_, chunkSize_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(qCompact, qCompact, dtLimitMax_, chunkSize_);
        AscendC::PipeBarrier<PIPE_ALL>();

        if (chunkSize_ == 64) {
            // T64 hot path: replace the 64-step Scalar loop with a
            // deterministic Hillis-Steele reverse inclusive scan.  Ping-pong
            // buffers avoid shifted in-place Add aliasing within a Vector
            // instruction.  Six stages leave the final result in gCsLocal.
            auto reverseTmp = reverseScanBuf_.Get<float>();
            auto reverseScanOffsets =
                reverseScanOffsetsBuf_.Get<uint32_t>();
            bool sourceIsGcs = true;
            int64_t scanStage = 0;
            for (int64_t offset = 1; offset < 64;
                 offset <<= 1, ++scanStage) {
                if (sourceIsGcs) {
                    AscendC::Muls(reverseTmp, gCsLocal, 1.0f, 64);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Gather(
                        reverseTmp, gCsLocal,
                        reverseScanOffsets[scanStage * 64],
                        0U, static_cast<uint32_t>(64 - offset));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(
                        reverseTmp, reverseTmp, gCsLocal, 64 - offset);
                } else {
                    AscendC::Muls(gCsLocal, reverseTmp, 1.0f, 64);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Gather(
                        gCsLocal, reverseTmp,
                        reverseScanOffsets[scanStage * 64],
                        0U, static_cast<uint32_t>(64 - offset));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(
                        gCsLocal, gCsLocal, reverseTmp, 64 - offset);
                }
                AscendC::PipeBarrier<PIPE_V>();
                sourceIsGcs = !sourceIsGcs;
            }
            if (!sourceIsGcs) {
                AscendC::Muls(gCsLocal, reverseTmp, 1.0f, 64);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // Compact the sparse row reductions left in xLocal.  Each row
            // result is 256 bytes apart for P=64.
            auto dqDirectCompact = dtPadded[2 * chunkSize_];
            AscendC::Gather(
                dqDirectCompact, xLocal,
                rowGatherOffsetsBuf_.Get<uint32_t>(), 0U, 64U);
            AscendC::PipeBarrier<PIPE_V>();

            // chain = clamp_grad(q_pre) * softplus_grad(u).  CompareScalar
            // preserves the inclusive clamp boundary semantics of PyTorch.
            auto chainCompact = dBuf_.Get<float>();
            if (dtSoftplus_ == 0) {
                AscendC::Duplicate(chainCompact, 1.0f, 64);
                AscendC::PipeBarrier<PIPE_V>();
            }
            auto compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::CompareScalar<float, uint8_t>(
                compareMask, qPreCompact, dtLimitMin_,
                AscendC::CMPMODE::GE, 64U);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select<float, uint8_t>(
                chainCompact, compareMask, chainCompact, 0.0f,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64U);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar<float, uint8_t>(
                compareMask, qPreCompact, dtLimitMax_,
                AscendC::CMPMODE::LE, 64U);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select<float, uint8_t>(
                chainCompact, compareMask, chainCompact, 0.0f,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64U);
            AscendC::PipeBarrier<PIPE_V>();

            auto ddtCompact = dtPadded[3 * chunkSize_];
            AscendC::Muls(ddtCompact, gCsLocal, aValue, 64);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ddtCompact, ddtCompact, dqDirectCompact, 64);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(ddtCompact, ddtCompact, chainCompact, 64);
            AscendC::PipeBarrier<PIPE_V>();
            // C220 does not implement Vector Scatter.  Broadcast each compact
            // scalar to one aligned 32-byte row; the existing strided MTE3
            // copy consumes only the first float from every row.
            const uint32_t ddtSrcShape[2] = {64U, 1U};
            const uint32_t ddtDstShape[2] = {64U, 8U};
            auto ddtBroadcastTmp = broadcastTmpBuf_.Get<uint8_t>();
            AscendC::Broadcast<float, 2, 1>(
                ddtLocal, ddtCompact, ddtDstShape, ddtSrcShape,
                ddtBroadcastTmp);
            AscendC::PipeBarrier<PIPE_V>();

            // Vector products plus one WholeReduceSum per partial replace
            // the two Scalar accumulators.  The two results occupy separate
            // 32-byte blocks in qPreCompact before the final scalar stores.
            AscendC::Mul(reverseTmp, gCsLocal, qCompact, 64);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(
                qPreCompact, reverseTmp, 64, 1, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(
                qPreCompact[8], ddtCompact, 64, 1, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            dAPartial = qPreCompact.GetValue(0);
            dBiasPartial = qPreCompact.GetValue(8);
        } else {
            // Keep the verified T128 implementation unchanged in this
            // single-variable optimization round.
            for (int64_t token = chunkSize_ - 1; token >= 0; --token) {
                running += gCsLocal.GetValue(token);
                const float qPre = qPreCompact.GetValue(token);
                const float q = qCompact.GetValue(token);
                const int64_t row = token * headdim_;
                float dqDirect = xLocal.GetValue(row);
                if (headdim_ == 128) dqDirect += xLocal.GetValue(row + 64);
                float chain =
                    (qPre >= dtLimitMin_ && qPre <= dtLimitMax_)
                    ? 1.0f : 0.0f;
                if (dtSoftplus_ != 0) {
                    chain *= sigmoidCompact.GetValue(token);
                }
                const float ddtValue =
                    (dqDirect + running * aValue) * chain;
                ddtLocal.SetValue(token * 8, ddtValue);
                dAPartial += running * q;
                dBiasPartial += ddtValue;
            }
        }

        // Broadcast q[T, 1] to [T, P] and compute dx in one vector multiply.
        // xLocal is deliberately reused: its reduction results have already
        // been consumed by the scalar reverse scan above.
        AscendC::PipeBarrier<PIPE_ALL>();
        const uint32_t qSrcShape[2] = {
            static_cast<uint32_t>(chunkSize_), 1U};
        const uint32_t qDstShape[2] = {
            static_cast<uint32_t>(chunkSize_),
            static_cast<uint32_t>(headdim_)};
        auto broadcastTmp = broadcastTmpBuf_.Get<uint8_t>();
        AscendC::Broadcast<float, 2, 1>(
            xLocal, qCompact, qDstShape, qSrcShape,
            broadcastTmp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(dxLocal, dXdtLocal, xLocal, matrixElements);
        AscendC::PipeBarrier<PIPE_V>();

        // Optional D branch for the fused gate epilogue.  Reuse dXdtLocal
        // after the direct path has consumed it, and reuse xLocal after the
        // q broadcast.  This avoids materializing dx_D or launching a public
        // Add over the complete [B,L,H,P] tensor.
        if (hasD_ != 0) {
            auto gyHalfLocal = gyHalfBuf_.Get<half>();
            auto dLocal = dBuf_.Get<float>();
            if (localHead >= 0) {
                AscendC::DataCopyParams dLocalParams{
                    1,
                    static_cast<uint16_t>(
                        headdim_ * sizeof(float) /
                        AscendC::DEFAULT_C0_SIZE),
                    0, 0};
                AscendC::DataCopy(
                    dLocal,
                    dHeadBlockBuf_.Get<float>()[localHead * headdim_],
                    dLocalParams);
                AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                AscendC::DataCopyExtParams dParams{
                    1, static_cast<uint32_t>(
                           headdim_ * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPad(
                    dLocal, dMatrixGm_[head * headdim_], dParams, noPad);
            }
            event_t mte2ToV = static_cast<event_t>(
                pipe_.FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
            AscendC::Cast(
                dXdtLocal, gyHalfLocal,
                AscendC::RoundMode::CAST_NONE, matrixElements);
            AscendC::PipeBarrier<PIPE_V>();
            if (computeDD_ != 0) {
                // xHeadBlockBuf still owns the original public-layout x tile.
                // Repack it inside UB and reduce x*gy across the 64 token
                // rows, avoiding the Gate kernel's second full HBM x read.
                auto xBlock = xHeadBlockBuf_.Get<float>();
                AscendC::DataCopyParams xLocalParams{
                    static_cast<uint16_t>(chunkSize_),
                    static_cast<uint16_t>(
                        headdim_ * sizeof(float) /
                        AscendC::DEFAULT_C0_SIZE),
                    static_cast<uint16_t>(
                        (validHeads - 1) * headdim_ * sizeof(float) /
                        AscendC::DEFAULT_C0_SIZE),
                    0};
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopy(
                    xLocal, xBlock[localHead * headdim_], xLocalParams);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(
                    xLocal, xLocal, dXdtLocal, matrixElements);
                AscendC::PipeBarrier<PIPE_V>();
                for (int64_t rows = chunkSize_ / 2;
                     rows >= 1; rows >>= 1) {
                    AscendC::Add(
                        xLocal, xLocal,
                        xLocal[rows * headdim_],
                        rows * headdim_);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                auto dDOut = dDPartialQueue_.AllocTensor<float>();
                AscendC::Adds(dDOut, xLocal, 0.0f, headdim_);
                dDPartialQueue_.EnQue(dDOut);
                dDOut = dDPartialQueue_.DeQue<float>();
                const int64_t dDOffset =
                    ((head * batch_ + batch) * nchunks_ + chunk) * headdim_;
                AscendC::DataCopyExtParams dDParams{
                    1, static_cast<uint32_t>(headdim_ * sizeof(float)),
                    0, 0, 0};
                AscendC::DataCopyPad(
                    dDPartialGm_[dDOffset], dDOut, dDParams);
                dDPartialQueue_.FreeTensor(dDOut);
            }
            const uint32_t dSrcShape[2] = {
                1U, static_cast<uint32_t>(headdim_)};
            const uint32_t dDstShape[2] = {
                static_cast<uint32_t>(chunkSize_),
                static_cast<uint32_t>(headdim_)};
            AscendC::Broadcast<float, 2, 0>(
                xLocal, dLocal, dDstShape, dSrcShape, broadcastTmp);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(dXdtLocal, dXdtLocal, xLocal, matrixElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(dxLocal, dxLocal, dXdtLocal, matrixElements);
            AscendC::PipeBarrier<PIPE_V>();
        }

        dxQueue_.EnQue(dxLocal);
        dxLocal = dxQueue_.DeQue<float>();
        if (localHead >= 0) {
            auto dxBlock = dxHeadBlockBuf_.Get<float>();
            AscendC::DataCopyParams packParams{
                static_cast<uint16_t>(chunkSize_),
                static_cast<uint16_t>(
                    headdim_ * sizeof(float) /
                    AscendC::DEFAULT_C0_SIZE),
                0,
                static_cast<uint16_t>(
                    (validHeads - 1) * headdim_ * sizeof(float) /
                    AscendC::DEFAULT_C0_SIZE)};
            AscendC::DataCopy(
                dxBlock[localHead * headdim_], dxLocal, packParams);
        } else {
            AscendC::DataCopyExtParams dxParams{
                static_cast<uint16_t>(chunkSize_),
                static_cast<uint32_t>(headdim_ * sizeof(float)), 0,
                static_cast<uint32_t>(
                    (nheads_ - 1) * headdim_ * sizeof(float)), 0};
            AscendC::DataCopyPad(dxGm_[xOffset], dxLocal, dxParams);
        }
        dxQueue_.FreeTensor(dxLocal);

        ddtQueue_.EnQue(ddtLocal);
        ddtLocal = ddtQueue_.DeQue<float>();
        AscendC::DataCopyExtParams ddtParams{
            static_cast<uint16_t>(chunkSize_), sizeof(float), 0,
            static_cast<uint32_t>((nheads_ - 1) * sizeof(float)), 0};
        AscendC::DataCopyPad(ddtGm_[dtOffset], ddtLocal, ddtParams);
        ddtQueue_.FreeTensor(ddtLocal);

        const int64_t partialOffset = (head * batch_ + batch) * nchunks_ + chunk;
        StoreChunkScalar(dAPartialGm_, partialOffset, dAPartial);
        StoreChunkScalar(dBiasPartialGm_, partialOffset, dBiasPartial);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_, dXdtBuf_, gCsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gCsOffBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gCsOffOffsetsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gCsLastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dtPaddedBuf_, qCompactBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dtGatherOffsetsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowGatherOffsetsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reverseScanBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reverseScanOffsetsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> compareMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastTmpBuf_, scalarBuf_, dBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xHeadBlockBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dxHeadBlockBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dtHeadBlockBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> headScalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dHeadBlockBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gyHalfBuf_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> dxQueue_, ddtQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> dDPartialQueue_;
    AscendC::GlobalTensor<float> xGm_, dXdtGm_, gCsGm_, gCsOffGm_;
    AscendC::GlobalTensor<half> dXdtHalfGm_;
    AscendC::GlobalTensor<float> gCsLastGm_, dtGm_;
    AscendC::GlobalTensor<float> aGm_, biasGm_, dxGm_, ddtGm_;
    AscendC::GlobalTensor<half> gyHeadGm_;
    AscendC::GlobalTensor<float> dMatrixGm_;
    AscendC::GlobalTensor<float> dAPartialGm_, dBiasPartialGm_;
    AscendC::GlobalTensor<float> dDPartialGm_;
    int64_t batch_, seqlen_, nheads_, nchunks_, chunkSize_, headdim_;
    int64_t hasBias_, dtSoftplus_, hasD_, usedCoreNum_, headBlock_, computeDD_;
    int64_t ngroups_, headsPerGroup_, groupedGcs_, dXdtIsHalf_;
    float dtLimitMin_, dtLimitMax_;
};

extern "C" __global__ __aicore__ void mamba2_ssd_dt_bwd(
    GM_ADDR x, GM_ADDR dXdt, GM_ADDR gCs, GM_ADDR dt, GM_ADDR A,
    GM_ADDR dtBias, GM_ADDR dx, GM_ADDR ddt, GM_ADDR dAPartial,
    GM_ADDR dBiasPartial,
    int64_t batch, int64_t seqlen, int64_t nheads, int64_t nchunks,
    int64_t chunkSize, int64_t headdim, int64_t hasBias,
    int64_t dtSoftplus, float dtLimitMin, float dtLimitMax,
    int64_t usedCoreNum)
{
    KernelMamba2SsdDtBwd op;
    op.Init(x, dXdt, gCs, gCs, gCs, dt, A, dtBias, dXdt, A,
            dx, ddt, dAPartial,
            dBiasPartial, dBiasPartial,
            batch, seqlen, nheads, nchunks,
            chunkSize, headdim, hasBias, dtSoftplus, 0, dtLimitMin,
            dtLimitMax, usedCoreNum, 1, 0, 1, 1, 0, 0);
    op.Process();
}

extern "C" __global__ __aicore__ void mamba2_ssd_dt_bwd_d(
    GM_ADDR x, GM_ADDR dXdt, GM_ADDR gCs, GM_ADDR dt, GM_ADDR A,
    GM_ADDR dtBias, GM_ADDR gyHead, GM_ADDR dMatrix,
    GM_ADDR dx, GM_ADDR ddt, GM_ADDR dAPartial, GM_ADDR dBiasPartial,
    GM_ADDR dDPartial,
    int64_t batch, int64_t seqlen, int64_t nheads, int64_t nchunks,
    int64_t chunkSize, int64_t headdim, int64_t hasBias,
    int64_t dtSoftplus, float dtLimitMin, float dtLimitMax,
    int64_t usedCoreNum, int64_t headBlock, int64_t computeDD,
    int64_t dXdtIsHalf)
{
    KernelMamba2SsdDtBwd op;
    op.Init(x, dXdt, gCs, gCs, gCs, dt, A, dtBias, gyHead, dMatrix,
            dx, ddt, dAPartial, dBiasPartial, dDPartial,
            batch, seqlen, nheads, nchunks, chunkSize, headdim,
            hasBias, dtSoftplus, 1, dtLimitMin, dtLimitMax,
            usedCoreNum, headBlock, computeDD, 1, 1, 0, dXdtIsHalf);
    op.Process();
}

extern "C" __global__ __aicore__ void mamba2_ssd_dt_bwd_grouped_d(
    GM_ADDR x, GM_ADDR dXdt, GM_ADDR gCsDiag, GM_ADDR gCsOff,
    GM_ADDR gCsLast, GM_ADDR dt, GM_ADDR A, GM_ADDR dtBias,
    GM_ADDR gyHead, GM_ADDR dMatrix,
    GM_ADDR dx, GM_ADDR ddt, GM_ADDR dAPartial, GM_ADDR dBiasPartial,
    GM_ADDR dDPartial,
    int64_t batch, int64_t seqlen, int64_t nheads, int64_t nchunks,
    int64_t chunkSize, int64_t headdim, int64_t hasBias,
    int64_t dtSoftplus, float dtLimitMin, float dtLimitMax,
    int64_t usedCoreNum, int64_t headBlock, int64_t computeDD,
    int64_t ngroups, int64_t headsPerGroup, int64_t gCsMode,
    int64_t dXdtIsHalf)
{
    KernelMamba2SsdDtBwd op;
    op.Init(x, dXdt, gCsDiag, gCsOff, gCsLast, dt, A, dtBias,
            gyHead, dMatrix, dx, ddt, dAPartial, dBiasPartial, dDPartial,
            batch, seqlen, nheads, nchunks, chunkSize, headdim,
            hasBias, dtSoftplus, 1, dtLimitMin, dtLimitMax,
            usedCoreNum, headBlock, computeDD,
            ngroups, headsPerGroup, gCsMode, dXdtIsHalf);
    op.Process();
}

class KernelMamba2SsdDtBwdReduce {
public:
    __aicore__ inline void Init(
        GM_ADDR dAPartial, GM_ADDR dBiasPartial, GM_ADDR dA, GM_ADDR dBias,
        int64_t batch, int64_t nheads, int64_t nchunks,
        int64_t usedCoreNum)
    {
        dAPartialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dAPartial));
        dBiasPartialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dBiasPartial));
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dA));
        dBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dBias));
        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        usedCoreNum_ = usedCoreNum;
        pipe_.InitBuffer(outQueue_, 1, 64 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t block = AscendC::GetBlockIdx();
        const int64_t perHead = batch_ * nchunks_;
        for (int64_t head = block; head < nheads_; head += usedCoreNum_) {
            float dAValue = 0.0f;
            float dBiasValue = 0.0f;
            const int64_t base = head * perHead;
            for (int64_t i = 0; i < perHead; ++i) {
                dAValue += dAPartialGm_.GetValue(base + i);
                dBiasValue += dBiasPartialGm_.GetValue(base + i);
            }
            Store(dAGm_, head, dAValue);
            Store(dBiasGm_, head, dBiasValue);
        }
    }

private:
    __aicore__ inline void Store(
        const AscendC::GlobalTensor<float> &dst, int64_t offset, float value)
    {
        auto output = outQueue_.AllocTensor<float>();
        output.SetValue(0, value);
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<float>();
        AscendC::DataCopyExtParams params{1, sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(dst[offset], output, params);
        outQueue_.FreeTensor(output);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::GlobalTensor<float> dAPartialGm_, dBiasPartialGm_;
    AscendC::GlobalTensor<float> dAGm_, dBiasGm_;
    int64_t batch_, nheads_, nchunks_, usedCoreNum_;
};

extern "C" __global__ __aicore__ void mamba2_ssd_dt_bwd_reduce(
    GM_ADDR dAPartial, GM_ADDR dBiasPartial, GM_ADDR dA, GM_ADDR dBias,
    int64_t batch, int64_t nheads, int64_t nchunks,
    int64_t usedCoreNum)
{
    KernelMamba2SsdDtBwdReduce op;
    op.Init(dAPartial, dBiasPartial, dA, dBias,
            batch, nheads, nchunks, usedCoreNum);
    op.Process();
}
