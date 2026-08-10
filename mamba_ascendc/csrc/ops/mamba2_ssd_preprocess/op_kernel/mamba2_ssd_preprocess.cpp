// Copyright (c) 2026, mamba-ascendc authors.

#include "kernel_operator.h"
#include "lib/pad/broadcast.h"

namespace {
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements = kTransposeTile * kTransposeTile;
constexpr uint32_t kBulkTransposeBytes = 8 * 1024;
constexpr int64_t kHeadBlock = 4;
}

template <bool GroupedX = false>
class KernelMamba2SsdPreprocess {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR dt, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR dtBias,
        GM_ADDR xCube, GM_ADDR dACumsum, GM_ADDR bCube, GM_ADDR cCube,
        int64_t batch, int64_t seqlen, int64_t nheads, int64_t headdim,
        int64_t dstate, int64_t ngroups, int64_t chunkSize,
        int64_t hasDtBias, int64_t dtSoftplus, float dtLimitMin,
        float dtLimitMax, int64_t usedCoreNum)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        dtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dt));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c));
        dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias));
        xCubeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(xCube));
        dACumsumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum));
        bCubeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bCube));
        cCubeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube));
        batch_ = batch;
        seqlen_ = seqlen;
        nheads_ = nheads;
        headdim_ = headdim;
        dstate_ = dstate;
        ngroups_ = ngroups;
        chunkSize_ = chunkSize;
        nchunks_ = seqlen / chunkSize;
        headBlock_ = GroupedX
            ? kHeadBlock
            : (chunkSize_ == 64 && headdim_ == 64 && nheads_ >= 32 &&
               nheads_ % kHeadBlock == 0 ? kHeadBlock : 1);
        useDtGather_ = batch_ * nheads_ * nchunks_ >= 512;
        scanLevels_ = 0;
        for (int64_t stride = 1; stride < chunkSize_; stride <<= 1) {
            ++scanLevels_;
        }
        hasDtBias_ = hasDtBias;
        dtSoftplus_ = dtSoftplus;
        dtLimitMin_ = dtLimitMin;
        dtLimitMax_ = dtLimitMax;
        usedCoreNum_ = usedCoreNum;
        const int64_t rowElements = headdim_ > dstate_ ? headdim_ : dstate_;
        const int64_t tileElements = chunkSize_ * rowElements;
        const int64_t bcElements = chunkSize_ * dstate_;
        useBulkTransposeStore_ =
            bcElements * static_cast<int64_t>(sizeof(half)) >=
            kBulkTransposeBytes;
        pipe_.InitBuffer(inQueue_, 1, tileElements * sizeof(float));
        pipe_.InitBuffer(outQueue_, 1, tileElements * sizeof(half));
        if (headBlock_ > 1) {
            // Public x is [B,L,H,P].  Load four adjacent heads in one
            // 1024-byte GM burst per token, then select each head in UB.
            // This keeps the consumer-facing [B,H,K,T,P] output unchanged.
            pipe_.InitBuffer(
                xHeadBlockBuf_,
                headBlock_ * chunkSize_ * headdim_ * sizeof(float));
            // Accumulate four head-major FP16 results in UB and issue one
            // strided MTE3 descriptor.  Each block is 8 KB, while the full
            // descriptor transfers 32 KB and avoids four independent stores.
            pipe_.InitBuffer(
                xHeadBlockOutBuf_,
                headBlock_ * chunkSize_ * headdim_ * sizeof(half));
        }
        pipe_.InitBuffer(transposeInQueue_, 1,
                         kTransposeElements * sizeof(half));
        pipe_.InitBuffer(transposeOutQueue_, 1,
                         kTransposeElements * sizeof(half));
        if (useBulkTransposeStore_) {
            // Assemble B^T contiguously in UB.  T128/N128 then replaces 64
            // separate 512-byte strided GM stores with one 32-KB store.
            pipe_.InitBuffer(bTransposeBuf_, bcElements * sizeof(half));
            bTransposeLocal_ = bTransposeBuf_.Get<half>();
            bPackToStoreEvent_ = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(
                    AscendC::HardEvent::MTE2_MTE3));
            bStoreToPackEvent_ = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(
                    AscendC::HardEvent::MTE3_MTE2));
        }
        pipe_.InitBuffer(
            dABuf_, headBlock_ * chunkSize_ * sizeof(float));
        // A one-element FP32 dt row is padded to one 32-byte UB block.
        pipe_.InitBuffer(dtBuf_, chunkSize_ * 8 * sizeof(float));
        pipe_.InitBuffer(dtCompactBuf_, chunkSize_ * sizeof(float));
        pipe_.InitBuffer(dtTmpBuf_, chunkSize_ * sizeof(float));
        pipe_.InitBuffer(dtMatrixBuf_, chunkSize_ * headdim_ * sizeof(float));
        pipe_.InitBuffer(broadcastTmpBuf_,
                         2 * chunkSize_ * headdim_ * sizeof(uint8_t));
        // DataCopyPad stores every scalar dt in its own 32-byte block.  Build
        // the byte-offset table once per core, then compact all 64 values with
        // one vector Gather per task instead of 64 scalar GetValue/SetValue
        // pairs.
        if (useDtGather_) {
            pipe_.InitBuffer(dtGatherOffsetsBuf_,
                             headBlock_ * chunkSize_ * sizeof(uint32_t));
            pipe_.InitBuffer(scanSourceBuf_,
                             (chunkSize_ + 8) * sizeof(float));
            pipe_.InitBuffer(scanOffsetsBuf_,
                             scanLevels_ * chunkSize_ * sizeof(uint32_t));
            auto gatherOffsets = dtGatherOffsetsBuf_.Get<uint32_t>();
            auto scanOffsets = scanOffsetsBuf_.Get<uint32_t>();
            for (int64_t localHead = 0; localHead < headBlock_; ++localHead) {
                for (int64_t token = 0; token < chunkSize_; ++token) {
                    gatherOffsets.SetValue(
                        localHead * chunkSize_ + token,
                        static_cast<uint32_t>(
                            token * 32 + localHead * sizeof(float)));
                }
            }
            for (uint32_t level = 0, stride = 1;
                 level < static_cast<uint32_t>(scanLevels_);
                 ++level, stride <<= 1) {
                for (uint32_t token = 0;
                     token < static_cast<uint32_t>(chunkSize_); ++token) {
                    const uint32_t source = token < stride
                        ? 0U : 8U + token - stride;
                    scanOffsets.SetValue(level * chunkSize_ + token,
                                         source * sizeof(float));
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void Process()
    {
        ProcessXAndDA();
        ProcessBC();
        WaitForPendingBTransposeStore();
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

    __aicore__ inline float ScalarLog(float x)
    {
        union FloatBits { uint32_t bits; float value; } value;
        value.value = x;
        const int32_t exponent = static_cast<int32_t>((value.bits >> 23) & 0xff) - 127;
        value.bits = (value.bits & 0x007fffffU) | 0x3f800000U;
        const float y = (value.value - 1.0f) / (value.value + 1.0f);
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

    __aicore__ inline float Softplus(float x)
    {
        if (x > 20.0f) return x;
        if (x < -9.0f) return ScalarExp(x);
        if (x >= 0.0f) return x + ScalarLog(1.0f + ScalarExp(-x));
        return ScalarLog(1.0f + ScalarExp(x));
    }

    __aicore__ inline float ProcessDtValue(float value, float bias)
    {
        if (hasDtBias_ != 0) value += bias;
        if (dtSoftplus_ != 0) value = Softplus(value);
        if (value < dtLimitMin_) value = dtLimitMin_;
        if (value > dtLimitMax_) value = dtLimitMax_;
        return value;
    }

    __aicore__ inline void ProcessXAndDA()
    {
        const int64_t headBlocks = nheads_ / headBlock_;
        const int64_t taskCount = batch_ * headBlocks * nchunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            const int64_t chunk = task % nchunks_;
            const int64_t headBlockIndex = (task / nchunks_) % headBlocks;
            const int64_t batch = task / (nchunks_ * headBlocks);
            const int64_t firstHead = headBlockIndex * headBlock_;
            const int64_t xElements = chunkSize_ * headdim_;
            const int64_t firstSeq = chunk * chunkSize_;
            const int64_t xSrcOffset =
                ((batch * seqlen_ + firstSeq) * nheads_ + firstHead) *
                headdim_;
            const int64_t dtSrcOffset =
                (batch * seqlen_ + firstSeq) * nheads_ + firstHead;

            auto xHeadBlockLocal = headBlock_ > 1
                ? xHeadBlockBuf_.Get<float>()
                : inQueue_.AllocTensor<float>();
            AscendC::DataCopyExtParams xCopyParams{
                static_cast<uint16_t>(chunkSize_),
                static_cast<uint32_t>(
                    headBlock_ * headdim_ * sizeof(float)),
                static_cast<uint32_t>(
                    (nheads_ - headBlock_) * headdim_ * sizeof(float)),
                0, 0};
            AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(
                xHeadBlockLocal, xGm_[xSrcOffset], xCopyParams, noPad);

            auto dtLocal = dtBuf_.Get<float>();
            AscendC::DataCopyExtParams dtCopyParams{
                static_cast<uint16_t>(chunkSize_),
                static_cast<uint32_t>(headBlock_ * sizeof(float)),
                static_cast<uint32_t>(
                    (nheads_ - headBlock_) * sizeof(float)),
                0, 0};
            AscendC::DataCopyPadExtParams<float> dtPad{
                true, 0, static_cast<uint8_t>(8 - headBlock_), 0.0f};
            AscendC::DataCopyPad(dtLocal, dtGm_[dtSrcOffset], dtCopyParams, dtPad);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (int64_t localHead = 0; localHead < headBlock_; ++localHead) {
                const int64_t head = firstHead + localHead;
                const int64_t outputTask =
                    (batch * nheads_ + head) * nchunks_ + chunk;
                const float bias = hasDtBias_ != 0
                    ? dtBiasGm_.GetValue(head) : 0.0f;
                const float aValue = aGm_.GetValue(head);

                auto xLocal = headBlock_ > 1
                    ? inQueue_.AllocTensor<float>() : xHeadBlockLocal;
                if (headBlock_ > 1) {
                    AscendC::DataCopyParams selectHeadParams{
                        static_cast<uint16_t>(chunkSize_),
                        static_cast<uint16_t>(
                            headdim_ * sizeof(float) /
                            AscendC::DEFAULT_C0_SIZE),
                        static_cast<uint16_t>(
                            (headBlock_ - 1) * headdim_ * sizeof(float) /
                            AscendC::DEFAULT_C0_SIZE),
                            0};
                    AscendC::DataCopy(
                        xLocal, xHeadBlockLocal[localHead * headdim_],
                        selectHeadParams);
                }
                inQueue_.EnQue(xLocal);
                xLocal = inQueue_.DeQue<float>();
                AscendC::PipeBarrier<PIPE_ALL>();

                auto dtCompact = dtCompactBuf_.Get<float>();
                if (useDtGather_) {
                    AscendC::Gather(
                        dtCompact, dtLocal,
                        dtGatherOffsetsBuf_.Get<uint32_t>()[
                            localHead * chunkSize_],
                        0U, static_cast<uint32_t>(chunkSize_));
                } else {
                    for (int64_t token = 0; token < chunkSize_; ++token) {
                        dtCompact.SetValue(
                            token, dtLocal.GetValue(token * 8 + localHead));
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (hasDtBias_ != 0) {
                    AscendC::Adds(dtCompact, dtCompact, bias, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                if (dtSoftplus_ != 0) {
                    // Stable vector softplus:
                    // max(x,0) + log(1 + exp(-abs(x))).
                    auto dtTmp = dtTmpBuf_.Get<float>();
                    AscendC::Abs(dtTmp, dtCompact, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(dtTmp, dtTmp, -1.0f, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Exp(dtTmp, dtTmp, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(dtTmp, dtTmp, 1.0f, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln(dtTmp, dtTmp, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Maxs(
                        dtCompact, dtCompact, 0.0f, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(
                        dtCompact, dtCompact, dtTmp, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Maxs(dtCompact, dtCompact, dtLimitMin_, chunkSize_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins(dtCompact, dtCompact, dtLimitMax_, chunkSize_);
                AscendC::PipeBarrier<PIPE_V>();

                auto dALocal = dABuf_.Get<float>()[
                    localHead * chunkSize_];
                if (useDtGather_) {
                AscendC::Muls(dALocal, dtCompact, aValue, chunkSize_);
                AscendC::PipeBarrier<PIPE_V>();
                auto scanSource = scanSourceBuf_.Get<float>();
                auto scanTmp = dtTmpBuf_.Get<float>();
                auto scanOffsets = scanOffsetsBuf_.Get<uint32_t>();
                for (uint32_t level = 0, stride = 1;
                     level < static_cast<uint32_t>(scanLevels_);
                     ++level, stride <<= 1) {
                    AscendC::Duplicate(scanSource, 0.0f, 8);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::DataCopy(scanSource[8], dALocal,
                                      static_cast<uint32_t>(chunkSize_));
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Gather(
                        scanTmp, scanSource,
                        scanOffsets[level * chunkSize_], 0U,
                        static_cast<uint32_t>(chunkSize_));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(dALocal, dALocal, scanTmp, chunkSize_);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                } else {
                    float prefix = 0.0f;
                    for (int64_t token = 0; token < chunkSize_; ++token) {
                        prefix += dtCompact.GetValue(token) * aValue;
                        dALocal.SetValue(token, prefix);
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                auto dtMatrix = dtMatrixBuf_.Get<float>();
                auto broadcastTmp = broadcastTmpBuf_.Get<uint8_t>();
                const uint32_t dtSrcShape[2] = {
                    static_cast<uint32_t>(chunkSize_), 1U};
                const uint32_t dtDstShape[2] = {
                    static_cast<uint32_t>(chunkSize_),
                    static_cast<uint32_t>(headdim_)};
                AscendC::Broadcast<float, 2, 1>(
                    dtMatrix, dtCompact, dtDstShape, dtSrcShape,
                    broadcastTmp);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(xLocal, xLocal, dtMatrix, xElements);
                AscendC::PipeBarrier<PIPE_V>();
                if (headBlock_ > 1) {
                    auto outLocal = xHeadBlockOutBuf_.Get<half>()[
                        localHead * xElements];
                    AscendC::Cast(
                        outLocal, xLocal, AscendC::RoundMode::CAST_NONE,
                        xElements);
                } else {
                    auto outLocal = outQueue_.AllocTensor<half>();
                    AscendC::Cast(
                        outLocal, xLocal, AscendC::RoundMode::CAST_NONE,
                        xElements);
                    outQueue_.EnQue(outLocal);
                    outLocal = outQueue_.DeQue<half>();
                    AscendC::DataCopyExtParams xOutParams{
                        1, static_cast<uint32_t>(xElements * sizeof(half)),
                        0, 0, 0};
                    AscendC::DataCopyPad(
                        xCubeGm_[outputTask * xElements], outLocal,
                        xOutParams);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyExtParams dAOutParams{
                        1,
                        static_cast<uint32_t>(chunkSize_ * sizeof(float)),
                        0, 0, 0};
                    AscendC::DataCopyPad(
                        dACumsumGm_[outputTask * chunkSize_], dALocal,
                        dAOutParams);
                    outQueue_.FreeTensor(outLocal);
                }
                inQueue_.FreeTensor(xLocal);
            }
            if (headBlock_ > 1) {
                AscendC::PipeBarrier<PIPE_ALL>();
                const int64_t firstOutputTask =
                    (batch * nheads_ + firstHead) * nchunks_ + chunk;
                if constexpr (GroupedX) {
                    // Convert the UB-only [R,T,P] work layout directly into
                    // canonical [T,R,P].  The public x load was already
                    // [T,R,P], so no intermediate GM tensor is materialized.
                    // Reuse the now-dead FP32 input buffer as a 32-KiB FP16
                    // destination and write one contiguous group task.
                    auto groupLocal = xHeadBlockBuf_.Get<half>();
                    const AscendC::DataCopyParams packHead{
                        static_cast<uint16_t>(chunkSize_),
                        static_cast<uint16_t>(
                            headdim_ * sizeof(half) /
                            AscendC::DEFAULT_C0_SIZE),
                        0,
                        static_cast<uint16_t>(
                            (headBlock_ - 1) * headdim_ * sizeof(half) /
                            AscendC::DEFAULT_C0_SIZE)};
                    for (int64_t localHead = 0; localHead < headBlock_;
                         ++localHead) {
                        AscendC::DataCopy(
                            groupLocal[localHead * headdim_],
                            xHeadBlockOutBuf_.Get<half>()[
                                localHead * xElements],
                            packHead);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                    const int64_t group = firstHead / headBlock_;
                    const int64_t groupTask =
                        (batch * nchunks_ + chunk) * ngroups_ + group;
                    AscendC::DataCopyExtParams xOutParams{
                        1,
                        static_cast<uint32_t>(
                            headBlock_ * xElements * sizeof(half)),
                        0, 0, 0};
                    AscendC::DataCopyPad(
                        xCubeGm_[groupTask * headBlock_ * xElements],
                        groupLocal, xOutParams);
                } else {
                    AscendC::DataCopyExtParams xOutParams{
                        static_cast<uint16_t>(headBlock_),
                        static_cast<uint32_t>(xElements * sizeof(half)),
                        0,
                        static_cast<uint32_t>(
                            (nchunks_ - 1) * xElements * sizeof(half)),
                        0};
                    AscendC::DataCopyPad(
                        xCubeGm_[firstOutputTask * xElements],
                        xHeadBlockOutBuf_.Get<half>(), xOutParams);
                }
                AscendC::DataCopyExtParams dAOutParams{
                    static_cast<uint16_t>(headBlock_),
                    static_cast<uint32_t>(chunkSize_ * sizeof(float)),
                    0,
                    static_cast<uint32_t>(
                        (nchunks_ - 1) * chunkSize_ * sizeof(float)),
                    0};
                AscendC::DataCopyPad(
                    dACumsumGm_[firstOutputTask * chunkSize_],
                    dABuf_.Get<float>(), dAOutParams);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessBC()
    {
        const int64_t taskCount = batch_ * nchunks_ * ngroups_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            const int64_t group = task % ngroups_;
            const int64_t chunk = (task / ngroups_) % nchunks_;
            const int64_t batch = task / (ngroups_ * nchunks_);
            const int64_t firstSeq = chunk * chunkSize_;
            const int64_t srcOffset =
                ((batch * seqlen_ + firstSeq) * ngroups_ + group) * dstate_;
            const int64_t tileElements = chunkSize_ * dstate_;
            const int64_t dstOffset = task * tileElements;
            CopyCastBTranspose(
                bGm_, bCubeGm_, srcOffset, dstOffset, tileElements);
            CopyCastTile(cGm_, cCubeGm_, srcOffset, dstOffset, tileElements);
        }
    }

    __aicore__ inline void CopyCastBTranspose(
        AscendC::GlobalTensor<float> &src, AscendC::GlobalTensor<half> &dst,
        int64_t srcOffset, int64_t dstOffset, int64_t tileElements)
    {
        auto inLocal = inQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams inParams{
            static_cast<uint16_t>(chunkSize_),
            static_cast<uint32_t>(dstate_ * sizeof(float)),
            static_cast<uint32_t>((ngroups_ - 1) * dstate_ * sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(inLocal, src[srcOffset], inParams, noPad);
        inQueue_.EnQue(inLocal);
        inLocal = inQueue_.DeQue<float>();
        auto halfLocal = outQueue_.AllocTensor<half>();
        AscendC::Cast(
            halfLocal, inLocal, AscendC::RoundMode::CAST_NONE,
            tileElements);
        outQueue_.EnQue(halfLocal);
        halfLocal = outQueue_.DeQue<half>();

        // A previous task may still be writing this shared UB buffer to GM.
        // Delay the dependency until immediately before the first overwrite,
        // so its MTE3 store overlaps C conversion and this task's load/cast.
        WaitForPendingBTransposeStore();

        // Write B directly as [N,T].  Packing one 16x16 tile in UB keeps the
        // source and destination strides out of the Transpose instruction.
        for (int64_t row = 0; row < chunkSize_; row += kTransposeTile) {
            for (int64_t col = 0; col < dstate_; col += kTransposeTile) {
                auto tileIn = transposeInQueue_.AllocTensor<half>();
                AscendC::DataCopyParams loadParams{
                    static_cast<uint16_t>(kTransposeTile),
                    static_cast<uint16_t>(
                        kTransposeTile * sizeof(half) /
                        AscendC::DEFAULT_C0_SIZE),
                    static_cast<uint16_t>(
                        (dstate_ - kTransposeTile) * sizeof(half) /
                        AscendC::DEFAULT_C0_SIZE),
                    0};
                AscendC::DataCopy(
                    tileIn, halfLocal[row * dstate_ + col], loadParams);
                transposeInQueue_.EnQue(tileIn);
                tileIn = transposeInQueue_.DeQue<half>();

                auto tileOut = transposeOutQueue_.AllocTensor<half>();
                AscendC::Transpose(tileOut, tileIn);
                transposeInQueue_.FreeTensor(tileIn);
                transposeOutQueue_.EnQue(tileOut);
                tileOut = transposeOutQueue_.DeQue<half>();
                if (useBulkTransposeStore_) {
                    AscendC::DataCopyParams packParams{
                        static_cast<uint16_t>(kTransposeTile),
                        static_cast<uint16_t>(
                            kTransposeTile * sizeof(half) /
                            AscendC::DEFAULT_C0_SIZE),
                        0,
                        static_cast<uint16_t>(
                            (chunkSize_ - kTransposeTile) * sizeof(half) /
                            AscendC::DEFAULT_C0_SIZE)};
                    AscendC::DataCopy(
                        bTransposeLocal_[col * chunkSize_ + row],
                        tileOut, packParams);
                } else {
                    AscendC::DataCopyExtParams storeParams{
                        static_cast<uint16_t>(kTransposeTile),
                        static_cast<uint32_t>(
                            kTransposeTile * sizeof(half)),
                        0,
                        static_cast<uint32_t>(
                            (chunkSize_ - kTransposeTile) * sizeof(half)),
                        0};
                    AscendC::DataCopyPad(
                        dst[dstOffset + col * chunkSize_ + row],
                        tileOut, storeParams);
                }
                transposeOutQueue_.FreeTensor(tileOut);
            }
        }
        if (useBulkTransposeStore_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(
                bPackToStoreEvent_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(
                bPackToStoreEvent_);
            AscendC::DataCopyExtParams storeParams{
                1, static_cast<uint32_t>(tileElements * sizeof(half)),
                0, 0, 0};
            AscendC::DataCopyPad(
                dst[dstOffset], bTransposeLocal_, storeParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                bStoreToPackEvent_);
            bTransposeStorePending_ = true;
        }
        inQueue_.FreeTensor(inLocal);
        outQueue_.FreeTensor(halfLocal);
    }

    __aicore__ inline void CopyCastTile(
        AscendC::GlobalTensor<float> &src, AscendC::GlobalTensor<half> &dst,
        int64_t srcOffset, int64_t dstOffset, int64_t tileElements)
    {
        auto inLocal = inQueue_.AllocTensor<float>();
        AscendC::DataCopyExtParams inParams{
            static_cast<uint16_t>(chunkSize_),
            static_cast<uint32_t>(dstate_ * sizeof(float)),
            static_cast<uint32_t>((ngroups_ - 1) * dstate_ * sizeof(float)),
            0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(inLocal, src[srcOffset], inParams, noPad);
        inQueue_.EnQue(inLocal);
        inLocal = inQueue_.DeQue<float>();
        auto outLocal = outQueue_.AllocTensor<half>();
        AscendC::Cast(
            outLocal, inLocal, AscendC::RoundMode::CAST_NONE, tileElements);
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.DeQue<half>();
        AscendC::DataCopyExtParams outParams{
            1, static_cast<uint32_t>(tileElements * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(dst[dstOffset], outLocal, outParams);
        inQueue_.FreeTensor(inLocal);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void WaitForPendingBTransposeStore()
    {
        if (useBulkTransposeStore_ && bTransposeStorePending_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                bStoreToPackEvent_);
            bTransposeStorePending_ = false;
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> transposeInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> transposeOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bTransposeBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xHeadBlockBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xHeadBlockOutBuf_;
    AscendC::LocalTensor<half> bTransposeLocal_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dABuf_, dtBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dtCompactBuf_, dtTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dtGatherOffsetsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scanSourceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scanOffsetsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dtMatrixBuf_, broadcastTmpBuf_;
    AscendC::GlobalTensor<float> xGm_, dtGm_, aGm_, bGm_, cGm_, dtBiasGm_;
    AscendC::GlobalTensor<half> xCubeGm_, bCubeGm_, cCubeGm_;
    AscendC::GlobalTensor<float> dACumsumGm_;
    int64_t batch_, seqlen_, nheads_, headdim_, dstate_, ngroups_;
    int64_t chunkSize_, nchunks_, headBlock_, hasDtBias_, dtSoftplus_;
    int64_t usedCoreNum_;
    int64_t scanLevels_ = 0;
    bool useDtGather_ = false;
    bool useBulkTransposeStore_ = false;
    bool bTransposeStorePending_ = false;
    event_t bPackToStoreEvent_;
    event_t bStoreToPackEvent_;
    float dtLimitMin_, dtLimitMax_;
};

extern "C" __global__ __aicore__ void mamba2_ssd_preprocess(
    GM_ADDR x, GM_ADDR dt, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR dtBias,
    GM_ADDR xCube, GM_ADDR dACumsum, GM_ADDR bCube, GM_ADDR cCube,
    int64_t batch, int64_t seqlen, int64_t nheads, int64_t headdim,
    int64_t dstate, int64_t ngroups, int64_t chunkSize,
    int64_t hasDtBias, int64_t dtSoftplus, float dtLimitMin,
    float dtLimitMax, int64_t usedCoreNum)
{
    KernelMamba2SsdPreprocess<false> op;
    op.Init(x, dt, a, b, c, dtBias, xCube, dACumsum, bCube, cCube,
            batch, seqlen, nheads, headdim, dstate, ngroups, chunkSize,
            hasDtBias, dtSoftplus, dtLimitMin, dtLimitMax, usedCoreNum);
    op.Process();
}

extern "C" __global__ __aicore__ void mamba2_ssd_preprocess_grouped(
    GM_ADDR x, GM_ADDR dt, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR dtBias,
    GM_ADDR xCube, GM_ADDR dACumsum, GM_ADDR bCube, GM_ADDR cCube,
    int64_t batch, int64_t seqlen, int64_t nheads, int64_t headdim,
    int64_t dstate, int64_t ngroups, int64_t chunkSize,
    int64_t hasDtBias, int64_t dtSoftplus, float dtLimitMin,
    float dtLimitMax, int64_t usedCoreNum)
{
    KernelMamba2SsdPreprocess<true> op;
    op.Init(x, dt, a, b, c, dtBias, xCube, dACumsum, bCube, cCube,
            batch, seqlen, nheads, headdim, dstate, ngroups, chunkSize,
            hasDtBias, dtSoftplus, dtLimitMin, dtLimitMax, usedCoreNum);
    op.Process();
}
