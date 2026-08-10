/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * T=P=N=64 MIX kernel for the off-diagonal chunk-scan backward.  The AIVs
 * paired with one AIC prepare architecture-owned half-precision row blocks.
 * The AIC performs Q^T@C and Q@S with FP32 accumulation, then the paired
 * AIVs form the matching per-head dA-cumsum contribution from dC_head and C.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/pad/broadcast.h"
#include "mamba2_dynamic_matmul.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 64;
constexpr uint32_t kTileElements = kTile * kTile;
// CANN builds a separate binary for each compute unit.  Ascend 950PR
// (NPU arch 3510; some CANN headers also expose __DAV_310R6__) has one
// Vector core paired with each Cube core.  910B keeps two Vector cores per
// Cube.  Derive row ownership from the binary topology so both paths cover
// exactly all 64 rows without changing the public layout or workspace.
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
constexpr uint32_t kAivPerAic = 1;
#else
constexpr uint32_t kAivPerAic = 2;
#endif
static_assert(kAivPerAic == 1 || kAivPerAic == 2,
              "unsupported MIX Vector/Cube ratio");
static_assert(kTile % kAivPerAic == 0,
              "Vector row ownership must cover a full tile");
constexpr uint32_t kRowsPerAiv = kTile / kAivPerAic;
constexpr uint32_t kBlockElements = kRowsPerAiv * kTile;
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements =
    kTransposeTile * kTransposeTile;
constexpr uint16_t kVectorReady = 0x8;
constexpr uint16_t kCubeReady = 0x9;

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
constexpr MatmulConfig kArch35MatmulConfig = GetBasicConfig(64, 64, 64);
using Arch35AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Arch35BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Arch35CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using Arch35BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

class Arch35BwdOffMatmul64 {
public:
    __aicore__ inline void Init(const TCubeTiling *cubeTiling)
    {
        object_.Init(cubeTiling);
    }

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &a,
        const GlobalTensor<half> &b,
        const GlobalTensor<float> &c)
    {
        object_.SetOrgShape(kTile, kTile, kTile, kTile, kTile);
        object_.SetSingleShape(kTile, kTile, kTile);
        object_.SetTensorA(a, false);
        object_.SetTensorB(b, false);
        object_.IterateAll(c, false);
        object_.End();
    }

private:
    matmul::MatmulImpl<Arch35AType, Arch35BType, Arch35CType,
                       Arch35BiasType, kArch35MatmulConfig> object_;
};
#endif

class VectorChunkScanBwdOff {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(csIn_, 1, kRowsPerAiv * sizeof(float));
        pipe->InitBuffer(gyIn_, 1, kBlockElements * sizeof(float));
        pipe->InitBuffer(qOut_, 1, kBlockElements * sizeof(half));
        pipe->InitBuffer(stateIn_, 1, kBlockElements * sizeof(float));
        pipe->InitBuffer(stateOut_, 1, kBlockElements * sizeof(half));
        pipe->InitBuffer(transposeIn_, 1,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(transposeOut_, 1,
                         kTransposeElements * sizeof(half));
        pipe->InitBuffer(dCIn_, 1, kBlockElements * sizeof(float));
        pipe->InitBuffer(cIn_, 1, kBlockElements * sizeof(half));
        pipe->InitBuffer(gDaOut_, 1, kRowsPerAiv * sizeof(float));
        pipe->InitBuffer(decayMatrix_,
                         kBlockElements * sizeof(float));
        pipe->InitBuffer(cFloat_, kBlockElements * sizeof(float));
        pipe->InitBuffer(product_, kBlockElements * sizeof(float));
        pipe->InitBuffer(broadcastTmp_,
                         2 * kBlockElements * sizeof(uint8_t));
    }

    __aicore__ inline void Prepare(
        const GlobalTensor<float> &gy,
        const GlobalTensor<float> &statesStart,
        const GlobalTensor<float> &dACumsum,
        GlobalTensor<half> q,
        GlobalTensor<half> qT,
        GlobalTensor<half> stateHalf)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        auto cs = csIn_.AllocTensor<float>();
        CopyIn(cs, dACumsum, rowBegin, kRowsPerAiv);
        csIn_.EnQue(cs);
        cs = csIn_.DeQue<float>();
        Exp(cs, cs, kRowsPerAiv);
        PipeBarrier<PIPE_V>();

        auto decay = decayMatrix_.Get<float>();
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t srcShape[2] = {kRowsPerAiv, 1U};
        const uint32_t dstShape[2] = {kRowsPerAiv, kTile};
        Broadcast<float, 2, 1>(decay, cs, dstShape, srcShape,
                               broadcastTmp);
        PipeBarrier<PIPE_V>();

        auto gyLocal = gyIn_.AllocTensor<float>();
        CopyIn(gyLocal, gy, rowBegin * kTile, kBlockElements);
        gyIn_.EnQue(gyLocal);
        gyLocal = gyIn_.DeQue<float>();
        Mul(gyLocal, gyLocal, decay, kBlockElements);
        PipeBarrier<PIPE_V>();

        auto qLocal = qOut_.AllocTensor<half>();
        Cast(qLocal, gyLocal, RoundMode::CAST_RINT, kBlockElements);
        gyIn_.FreeTensor(gyLocal);
        csIn_.FreeTensor(cs);
        qOut_.EnQue(qLocal);
        qLocal = qOut_.DeQue<half>();
        CopyOut(q, qLocal, rowBegin * kTile, kBlockElements);

        // Transpose directly from the still-live UB Q block.  Each AIV owns
        // 32 source rows and therefore disjoint destination columns in Q^T.
        for (uint32_t localRow = 0; localRow < kRowsPerAiv;
             localRow += kTransposeTile) {
            for (uint32_t col = 0; col < kTile;
                 col += kTransposeTile) {
                TransposeLocalTile(
                    qLocal[localRow * kTile + col],
                    qT[col * kTile + rowBegin + localRow]);
            }
        }
        qOut_.FreeTensor(qLocal);

        auto stateLocal = stateIn_.AllocTensor<float>();
        CopyIn(stateLocal, statesStart, rowBegin * kTile,
               kBlockElements);
        stateIn_.EnQue(stateLocal);
        stateLocal = stateIn_.DeQue<float>();
        auto stateHalfLocal = stateOut_.AllocTensor<half>();
        Cast(stateHalfLocal, stateLocal, RoundMode::CAST_RINT,
             kBlockElements);
        stateIn_.FreeTensor(stateLocal);
        stateOut_.EnQue(stateHalfLocal);
        stateHalfLocal = stateOut_.DeQue<half>();
        CopyOut(stateHalf, stateHalfLocal, rowBegin * kTile,
                kBlockElements);
        stateOut_.FreeTensor(stateHalfLocal);
    }

    __aicore__ inline void ComputeGdA(
        const GlobalTensor<float> &dCHead,
        const GlobalTensor<half> &cCube,
        GlobalTensor<float> gDa)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        auto dCLocal = dCIn_.AllocTensor<float>();
        CopyIn(dCLocal, dCHead, rowBegin * kTile, kBlockElements);
        dCIn_.EnQue(dCLocal);
        dCLocal = dCIn_.DeQue<float>();

        auto cLocal = cIn_.AllocTensor<half>();
        CopyIn(cLocal, cCube, rowBegin * kTile, kBlockElements);
        cIn_.EnQue(cLocal);
        cLocal = cIn_.DeQue<half>();
        auto cFloat = cFloat_.Get<float>();
        Cast(cFloat, cLocal, RoundMode::CAST_NONE, kBlockElements);
        cIn_.FreeTensor(cLocal);
        PipeBarrier<PIPE_V>();

        auto product = product_.Get<float>();
        Mul(product, dCLocal, cFloat, kBlockElements);
        dCIn_.FreeTensor(dCLocal);
        PipeBarrier<PIPE_V>();

        auto gDaLocal = gDaOut_.AllocTensor<float>();
        // WholeReduceSum emits one contiguous scalar per repeat.  Reduce the
        // architecture-owned row slab (32 rows on 910B, 64 on 950) with one
        // instruction instead of per-row reductions and barriers.
        WholeReduceSum<float, true>(
            gDaLocal, product, static_cast<int32_t>(kTile),
            static_cast<int32_t>(kRowsPerAiv), 1, 1,
            static_cast<int32_t>(kTile * sizeof(float) /
                                 DEFAULT_C0_SIZE));
        PipeBarrier<PIPE_V>();
        gDaOut_.EnQue(gDaLocal);
        gDaLocal = gDaOut_.DeQue<float>();
        CopyOut(gDa, gDaLocal, rowBegin, kRowsPerAiv);
        gDaOut_.FreeTensor(gDaLocal);
    }

private:
    template <typename T>
    __aicore__ inline void CopyIn(
        const LocalTensor<T> &dst, const GlobalTensor<T> &src,
        uint64_t offset, uint32_t elements)
    {
        DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], copy, pad);
    }

    template <typename T>
    __aicore__ inline void CopyOut(
        GlobalTensor<T> dst, const LocalTensor<T> &src,
        uint64_t offset, uint32_t elements)
    {
        DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copy);
    }

    __aicore__ inline void TransposeLocalTile(
        const LocalTensor<half> &src, GlobalTensor<half> dst)
    {
        auto packed = transposeIn_.AllocTensor<half>();
        DataCopyParams pack{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((kTile - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(packed, src, pack);
        transposeIn_.EnQue(packed);
        packed = transposeIn_.DeQue<half>();

        auto transposed = transposeOut_.AllocTensor<half>();
        Transpose(transposed, packed);
        transposeIn_.FreeTensor(packed);
        transposeOut_.EnQue(transposed);
        transposed = transposeOut_.DeQue<half>();
        DataCopyExtParams store{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint32_t>(kTransposeTile * sizeof(half)),
            0,
            static_cast<uint32_t>((kTile - kTransposeTile) * sizeof(half)),
            0};
        DataCopyPad(dst, transposed, store);
        transposeOut_.FreeTensor(transposed);
    }

    TQue<TPosition::VECIN, 1> csIn_;
    TQue<TPosition::VECIN, 1> gyIn_;
    TQue<TPosition::VECOUT, 1> qOut_;
    TQue<TPosition::VECIN, 1> stateIn_;
    TQue<TPosition::VECOUT, 1> stateOut_;
    TQue<TPosition::VECIN, 1> transposeIn_;
    TQue<TPosition::VECOUT, 1> transposeOut_;
    TQue<TPosition::VECIN, 1> dCIn_;
    TQue<TPosition::VECIN, 1> cIn_;
    TQue<TPosition::VECOUT, 1> gDaOut_;
    TBuf<TPosition::VECCALC> decayMatrix_;
    TBuf<TPosition::VECCALC> cFloat_;
    TBuf<TPosition::VECCALC> product_;
    TBuf<TPosition::VECCALC> broadcastTmp_;
};

class KernelMamba2SsdChunkScanBwdOff {
public:
    __aicore__ inline void Init(
        GM_ADDR gy, GM_ADDR statesStart, GM_ADDR dACumsum, GM_ADDR cCube,
        GM_ADDR dStatesStart, GM_ADDR dCHead, GM_ADDR gDaCsOff,
        GM_ADDR workspace,
        const Mamba2SsdChunkScanBwdOffTilingData &tiling,
        TPipe *pipe)
    {
        tiling_ = tiling;
        const uint64_t headTasks = tiling_.taskCount;
        const uint64_t groupTasks =
            static_cast<uint64_t>(tiling_.batch) * tiling_.chunks *
            tiling_.groups;
        gyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gy),
                              headTasks * kTileElements);
        stateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(statesStart),
            headTasks * kTileElements);
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum),
                              headTasks * kTile);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube),
                             groupTasks * kTileElements);
        dStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dStatesStart),
            headTasks * kTileElements);
        dCHeadGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dCHead),
                                  headTasks * kTileElements);
        gDaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gDaCsOff),
                               headTasks * kTile);

        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) *
                tiling_.workspaceBytesPerCore;
        workspaceHalf_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(coreWorkspace),
            3 * kTileElements);
        if ASCEND_IS_AIC {
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
            matmul_.Init(&tiling_.cubeTilingData);
#else
            matmul_.Init(*pipe);
#endif
        }
        if ASCEND_IS_AIV {
            vector_.Init(pipe);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t task = coreIdx_; task < tiling_.taskCount;
             task += tiling_.usedCoreNum) {
            const uint32_t chunk = task % tiling_.chunks;
            const uint32_t outer = task / tiling_.chunks;
            const uint32_t head = outer % tiling_.heads;
            const uint32_t batch = outer / tiling_.heads;
            const uint32_t group = head / tiling_.headsPerGroup;
            const uint32_t groupTask =
                (batch * tiling_.chunks + chunk) * tiling_.groups + group;
            const uint64_t matrixOffset =
                static_cast<uint64_t>(task) * kTileElements;
            const uint64_t vectorOffset =
                static_cast<uint64_t>(task) * kTile;
            const uint64_t cOffset =
                static_cast<uint64_t>(groupTask) * kTileElements;
            auto q = workspaceHalf_[0];
            auto qT = workspaceHalf_[kTileElements];
            auto stateHalf = workspaceHalf_[2 * kTileElements];

            if ASCEND_IS_AIV {
                vector_.Prepare(
                    gyGm_[matrixOffset], stateGm_[matrixOffset],
                    dAGm_[vectorOffset], q, qT, stateHalf);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(kVectorReady);
                CrossCoreWaitFlag<0x2>(kCubeReady);
                vector_.ComputeGdA(
                    dCHeadGm_[matrixOffset], cGm_[cOffset],
                    gDaGm_[vectorOffset]);
            }
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<0x2>(kVectorReady);
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
                matmul_.ComputeBlock(
                    qT, cGm_[cOffset], dStateGm_[matrixOffset]);
                matmul_.ComputeBlock(
                    q, stateHalf, dCHeadGm_[matrixOffset]);
#else
                matmul_.ComputeBlock(
                    qT, cGm_[cOffset], dStateGm_[matrixOffset],
                    kTile, kTile, kTile, kTile);
                matmul_.ComputeBlock(
                    q, stateHalf, dCHeadGm_[matrixOffset],
                    kTile, kTile, kTile, kTile);
#endif
                CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeReady);
            }
        }
    }

private:
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
    Arch35BwdOffMatmul64 matmul_;
#else
    Mamba2DynamicMatmul<half, float, 64> matmul_;
#endif
    VectorChunkScanBwdOff vector_;
    GlobalTensor<float> gyGm_;
    GlobalTensor<float> stateGm_;
    GlobalTensor<float> dAGm_;
    GlobalTensor<half> cGm_;
    GlobalTensor<float> dStateGm_;
    GlobalTensor<float> dCHeadGm_;
    GlobalTensor<float> gDaGm_;
    GlobalTensor<half> workspaceHalf_;
    Mamba2SsdChunkScanBwdOffTilingData tiling_;
    uint32_t coreIdx_ = 0;
};
} // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_chunk_scan_bwd_off(
    GM_ADDR gy, GM_ADDR states_start, GM_ADDR d_a_cumsum, GM_ADDR c_cube,
    GM_ADDR d_states_start, GM_ADDR d_c_head, GM_ADDR g_d_a_cs_off,
    GM_ADDR workspace, GM_ADDR tiling)
{
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelMamba2SsdChunkScanBwdOff op;
        op.Init(gy, states_start, d_a_cumsum, c_cube,
                d_states_start, d_c_head, g_d_a_cs_off,
                workspace, tilingData, &pipe);
        op.Process();
    }
}
