#ifndef MAMBA2_DYNAMIC_MATMUL_H
#define MAMBA2_DYNAMIC_MATMUL_H

#include "kernel_operator.h"

// A 64x64 output-tile Cube helper whose K length and all GM row strides are
// supplied per invocation.  One allocation can therefore serve CB, diagonal
// output and state GEMMs without triplicating L1/L0 storage.
template <typename InType, typename OutType, uint32_t BaseK = 16,
          uint32_t MaxN = 128>
class Mamba2DynamicMatmul {
    static constexpr uint32_t kBaseM = 64;
    static constexpr uint32_t kBaseN = 64;
    // DiagState backward packs up to four N=64 heads into a 256-column
    // projection.  64x256 FP32 occupies 64 KiB in L0C, within one Cube
    // accumulator, and avoids four independent Fixpipe operations.
    static constexpr uint32_t kMaxN = MaxN;
    // BaseK is selected per caller: ChunkMix uses a complete 64-column tile
    // to reduce control, while StateEpilogue retains the pipelined 16-column
    // tile that performs better for its chunk-stream schedule.
    static constexpr uint32_t kBaseK = BaseK;
    static constexpr uint32_t kL1Prefetch = 64 / kBaseK;
    static constexpr uint32_t kC0 = 32 / sizeof(InType);
    static constexpr uint32_t kBaseMK = kBaseM * kBaseK;
    static constexpr uint32_t kBaseKN = kBaseK * kMaxN;
    static constexpr uint32_t kBaseMN = kBaseM * kMaxN;

public:
    __aicore__ inline void Init(AscendC::TPipe &pipe)
    {
        pipe.InitBuffer(a1Queue_, 2,
                        kBaseMK * kL1Prefetch * sizeof(InType));
        pipe.InitBuffer(a2Queue_, 2, kBaseMK * sizeof(InType));
        pipe.InitBuffer(b1Queue_, 2,
                        kBaseKN * kL1Prefetch * sizeof(InType));
        pipe.InitBuffer(b2Queue_, 2, kBaseKN * sizeof(InType));
        pipe.InitBuffer(c1Queue_, 1, kBaseMN * sizeof(OutType));
    }

    __aicore__ inline void ComputeBlock(
        const AscendC::GlobalTensor<InType> &a,
        const AscendC::GlobalTensor<InType> &b,
        const AscendC::GlobalTensor<OutType> &c,
        uint32_t kTotal, uint32_t lda, uint32_t ldb, uint32_t ldc,
        uint32_t nTotal = kBaseN)
    {
        auto c1 = c1Queue_.AllocTensor<OutType>();
        AccumulateBlock(c1, a, b, kTotal, lda, ldb, true, nTotal);
        StoreAccumulator(c1, c, ldc, nTotal);
    }

    // Preserve FP32 accumulation in L0C while reducing the GM rendezvous
    // between Cube and Vector to FP16.  This is intended for internal
    // producer/consumer workspaces; public outputs continue to use the
    // regular FP32 path above.
    __aicore__ inline void ComputeBlockToHalf(
        const AscendC::GlobalTensor<InType> &a,
        const AscendC::GlobalTensor<InType> &b,
        const AscendC::GlobalTensor<half> &c,
        uint32_t kTotal, uint32_t lda, uint32_t ldb, uint32_t ldc,
        uint32_t nTotal = kBaseN)
    {
        auto c1 = c1Queue_.AllocTensor<OutType>();
        AccumulateBlock(c1, a, b, kTotal, lda, ldb, true, nTotal);
        c1Queue_.EnQue(c1);
        CopyOutHalf(c, ldc, nTotal);
    }

    // Explicit accumulator lifetime for a sequence of GEMMs that contribute
    // to the same 64xN output.  The first call initializes L0C; following
    // calls retain it.  Fixpipe runs only once in StoreAccumulator.
    __aicore__ inline AscendC::LocalTensor<OutType> AllocAccumulator()
    {
        return c1Queue_.AllocTensor<OutType>();
    }

    __aicore__ inline void AccumulateBlock(
        const AscendC::LocalTensor<OutType> &c1,
        const AscendC::GlobalTensor<InType> &a,
        const AscendC::GlobalTensor<InType> &b,
        uint32_t kTotal, uint32_t lda, uint32_t ldb,
        bool initialize, uint32_t nTotal = kBaseN)
    {
        const uint32_t kTiles = kTotal / kBaseK;

        for (uint32_t outer = 0; outer < kTiles; outer += kL1Prefetch) {
            const uint32_t remain = kTiles - outer;
            const uint32_t count =
                remain < kL1Prefetch ? remain : kL1Prefetch;
            const uint32_t kLen = count * kBaseK;

            CopyA(a[outer * kBaseK], kLen, lda);
            CopyB(b[outer * kBaseK * ldb], kLen, ldb, nTotal);
            auto a1 = a1Queue_.DeQue<InType>();
            auto b1 = b1Queue_.DeQue<InType>();

            for (uint32_t i = 0; i < count; ++i) {
                SplitA(a1, i * kBaseMK, kBaseM);
                SplitB(b1, i * kBaseK * kC0, kLen, nTotal);
                Compute(c1, initialize && outer + i == 0, nTotal);
            }
            a1Queue_.FreeTensor(a1);
            b1Queue_.FreeTensor(b1);
        }

    }

    __aicore__ inline void StoreAccumulator(
        const AscendC::LocalTensor<OutType> &c1,
        const AscendC::GlobalTensor<OutType> &c,
        uint32_t ldc, uint32_t nTotal = kBaseN)
    {
        c1Queue_.EnQue(c1);
        CopyOut(c, ldc, nTotal);
    }

private:
    __aicore__ inline void CopyA(
        const AscendC::GlobalTensor<InType> &src, uint32_t kLen, uint32_t lda)
    {
        auto dst = a1Queue_.AllocTensor<InType>();
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = kBaseM;
        params.dValue = kLen;
        params.srcNdMatrixStride = 0;
        params.srcDValue = lda;
        params.dstNzC0Stride = kBaseM;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
        AscendC::DataCopy(dst, src, params);
        a1Queue_.EnQue(dst);
    }

    __aicore__ inline void CopyB(
        const AscendC::GlobalTensor<InType> &src, uint32_t kLen, uint32_t ldb,
        uint32_t nTotal)
    {
        auto dst = b1Queue_.AllocTensor<InType>();
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = kLen;
        params.dValue = nTotal;
        params.srcNdMatrixStride = 0;
        params.srcDValue = ldb;
        params.dstNzC0Stride = kLen;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
        AscendC::DataCopy(dst, src, params);
        b1Queue_.EnQue(dst);
    }

    __aicore__ inline void SplitA(
        const AscendC::LocalTensor<InType> &src,
        uint32_t offset, uint32_t colC0Stride)
    {
        auto dst = a2Queue_.AllocTensor<InType>();
        AscendC::LoadData3DParamsV2<InType> params;
        params.l1H = 1;
        params.l1W = colC0Stride;
        params.channelSize = kBaseK;
        params.kExtension = kBaseK;
        params.mExtension = kBaseM;
        params.strideH = 1;
        params.strideW = 1;
        params.filterH = 1;
        params.filterW = 1;
        params.dilationFilterH = 1;
        params.dilationFilterW = 1;
        AscendC::LoadData(dst, src[offset], params);
        a2Queue_.EnQue(dst);
    }

    __aicore__ inline void SplitB(
        const AscendC::LocalTensor<InType> &src,
        uint32_t offset, uint32_t colC0Stride, uint32_t nTotal)
    {
        auto dst = b2Queue_.AllocTensor<InType>();
        AscendC::LoadData3DParamsV2<InType> params;
        params.l1H = 1;
        params.l1W = colC0Stride;
        params.channelSize = nTotal;
        params.kExtension = nTotal;
        params.mExtension = kBaseK;
        params.strideH = 1;
        params.strideW = 1;
        params.filterH = 1;
        params.filterW = 1;
        params.dilationFilterH = 1;
        params.dilationFilterW = 1;
        AscendC::LoadData(dst, src[offset], params);
        b2Queue_.EnQue(dst);
    }

    __aicore__ inline void Compute(
        const AscendC::LocalTensor<OutType> &c, bool initialize,
        uint32_t nTotal)
    {
        auto a = a2Queue_.DeQue<InType>();
        auto b = b2Queue_.DeQue<InType>();
        AscendC::MmadParams params;
        params.m = kBaseM;
        params.n = nTotal;
        params.k = kBaseK;
        params.cmatrixInitVal = initialize;
        AscendC::Mmad(c, a, b, params);
        a2Queue_.FreeTensor(a);
        b2Queue_.FreeTensor(b);
    }

    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<OutType> &dst, uint32_t ldc,
        uint32_t nTotal)
    {
        auto src = c1Queue_.DeQue<OutType>();
        AscendC::FixpipeParamsV220 params;
        params.nSize = nTotal;
        params.mSize = kBaseM;
        params.srcStride = kBaseM;
        params.dstStride = ldc;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        AscendC::Fixpipe(dst, src, params);
        // c1Queue_ has a single L0C slot and ComputeBlock is called several
        // times back-to-back.  Fixpipe is asynchronous with respect to the
        // next Mmad, so do not release/reuse L0C until its GM write completed.
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_M));
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventId);
        c1Queue_.FreeTensor(src);
    }

    __aicore__ inline void CopyOutHalf(
        const AscendC::GlobalTensor<half> &dst, uint32_t ldc,
        uint32_t nTotal)
    {
        auto src = c1Queue_.DeQue<OutType>();
        AscendC::FixpipeParamsV220 params;
        params.nSize = nTotal;
        params.mSize = kBaseM;
        params.srcStride = kBaseM;
        params.dstStride = ldc;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        params.quantPre = QuantMode_t::F322F16;
        AscendC::Fixpipe(dst, src, params);
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_M));
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventId);
        c1Queue_.FreeTensor(src);
    }

    AscendC::TQue<AscendC::TPosition::A1, 1> a1Queue_;
    AscendC::TQue<AscendC::TPosition::A2, 1> a2Queue_;
    AscendC::TQue<AscendC::TPosition::B1, 1> b1Queue_;
    AscendC::TQue<AscendC::TPosition::B2, 1> b2Queue_;
    AscendC::TQue<AscendC::TPosition::CO1, 1> c1Queue_;
};

#endif // MAMBA2_DYNAMIC_MATMUL_H
