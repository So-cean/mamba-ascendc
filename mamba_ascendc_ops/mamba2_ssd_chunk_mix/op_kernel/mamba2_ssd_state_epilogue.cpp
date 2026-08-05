/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * One MIX block owns one [B,H] stream.  Vector keeps the recurrent state in
 * UB across chunks while Cube projects C[T,N] @ state[N,P].  Two GM slots
 * pipeline state transpose, Cube projection, and the public-layout epilogue.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/pad/broadcast.h"
#include "mamba2_dynamic_matmul.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 64;
constexpr uint32_t kTileElements = kTile * kTile;
constexpr uint32_t kRowsPerAiv = kTile / 2;
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements = kTransposeTile * kTransposeTile;
constexpr uint16_t kStateReady0 = 0x8;
constexpr uint16_t kStateReady1 = 0x9;
constexpr uint16_t kCubeReady0 = 0xA;
constexpr uint16_t kCubeReady1 = 0xB;

template <uint32_t ChunkSize>
class VectorStateEpilogueT {
public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t stateDim,
                                uint32_t headsPerTask = 1)
    {
        stateDim_ = stateDim;
        stateNpLayout_ = stateDim_ == kTile || ChunkSize == 128;
        stateRowsPerAiv_ = stateNpLayout_ ? stateDim_ / 2 : kRowsPerAiv;
        stateRowStride_ = stateNpLayout_ ? kTile : stateDim_;
        stateBlockElements_ = stateRowsPerAiv_ * stateRowStride_;
        headsPerTask_ = headsPerTask;
        pipe->InitBuffer(state_, headsPerTask_ * stateBlockElements_ *
                                     sizeof(float));
        pipe->InitBuffer(stateContribution_, 1,
                         stateBlockElements_ * sizeof(float));
        pipe->InitBuffer(finalOut_, 1,
                         stateBlockElements_ * sizeof(float));
        pipe->InitBuffer(stateFloatIn_, 1,
                         kTransposeElements * sizeof(float));
        pipe->InitBuffer(stateHalfOut_, 1,
                         kTransposeElements * sizeof(half));
        if (!stateNpLayout_) {
            pipe->InitBuffer(transposeOut_, 1,
                             kTransposeElements * sizeof(half));
        }
#ifndef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        // Standalone StateEpilogue can use at most eight VECIN queues on
        // Ascend 910B.  Keep the one-shot initial-state load on the state
        // contribution queue, but give the three hot epilogue inputs their
        // own queues so their DMA events do not serialize with dA/x.
        pipe->InitBuffer(dIn_, 1, kTile * sizeof(float));
        pipe->InitBuffer(yOffIn_, 1,
                         kTokenRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(zIn_, 1,
                         kTokenRowsPerAiv * kTile * sizeof(float));
#endif
        // dA rows and D are consumed at disjoint phases.  A single 64-value
        // queue covers both and saves one scarce VECIN queue event when this
        // component is embedded in a larger MIX kernel.
        pipe->InitBuffer(dAIn_, 1, kTile * sizeof(float));
        pipe->InitBuffer(yDiagIn_, 1,
                         kTokenRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(xIn_, 1,
                         kTokenRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(out_, 1,
                         kTokenRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(matrix0_,
                         kTokenRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(matrix1_,
                         kTokenRowsPerAiv * kTile * sizeof(float));
        pipe->InitBuffer(dMatrix_, headsPerTask_ * kTokenRowsPerAiv * kTile *
                                       sizeof(float));
        pipe->InitBuffer(broadcastTmp_,
                         2 * kTokenRowsPerAiv * kTile * sizeof(uint8_t));
    }

    __aicore__ inline void PrepareD(const GlobalTensor<float> &d,
                                    uint32_t headSlot = 0)
    {
        constexpr uint32_t blockElements = kTokenRowsPerAiv * kTile;
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        auto dRow = dAIn_.AllocTensor<float>();
#else
        auto dRow = dIn_.AllocTensor<float>();
#endif
        DataCopy(dRow, d, kTile);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        dAIn_.EnQue(dRow);
        dRow = dAIn_.DeQue<float>();
#else
        dIn_.EnQue(dRow);
        dRow = dIn_.DeQue<float>();
#endif
        const uint32_t dSrcShape[2] = {1U, kTile};
        const uint32_t matrixShape[2] = {kTokenRowsPerAiv, kTile};
        auto dMatrix = dMatrix_.Get<float>()[headSlot * blockElements];
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        Broadcast<float, 2, 0>(dMatrix, dRow, matrixShape,
                               dSrcShape, broadcastTmp);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        dAIn_.FreeTensor(dRow);
#else
        dIn_.FreeTensor(dRow);
#endif
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitializeState(
        const GlobalTensor<float> &initial, uint32_t hasInitial,
        uint32_t headSlot = 0)
    {
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        if (hasInitial != 0) {
            const uint32_t rowBegin = GetSubBlockIdx() * stateRowsPerAiv_;
            auto initialLocal = stateContribution_.AllocTensor<float>();
            DataCopy(initialLocal, initial[rowBegin * stateRowStride_],
                     stateBlockElements_);
            stateContribution_.EnQue(initialLocal);
            initialLocal = stateContribution_.DeQue<float>();
            Adds(state, initialLocal, 0.0f, stateBlockElements_);
            stateContribution_.FreeTensor(initialLocal);
        } else {
            Duplicate(state, 0.0f, stateBlockElements_);
        }
    }

    __aicore__ inline void PrepareState(
        GlobalTensor<half> stateT, uint32_t headSlot = 0,
        uint32_t outputCols = kTile)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * stateRowsPerAiv_;
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        if (stateNpLayout_) {
            // The single-head N/P layout is already contiguous in both UB
            // and workspace.  Cast and store the complete AIV-owned half in
            // one operation instead of issuing one load/cast/store sequence
            // for every 16x16 tile.
            if (headsPerTask_ == 1) {
                auto stateHalf = finalOut_.AllocTensor<half>();
                Cast(stateHalf, state, RoundMode::CAST_RINT,
                     stateBlockElements_);
                finalOut_.EnQue(stateHalf);
                stateHalf = finalOut_.DeQue<half>();
                DataCopyExtParams store{
                    1,
                    static_cast<uint32_t>(
                        stateBlockElements_ * sizeof(half)),
                    0,
                    0,
                    0};
                DataCopyPad(stateT[rowBegin * kTile], stateHalf, store);
                finalOut_.FreeTensor(stateHalf);
                return;
            }
            for (uint32_t localRow = 0; localRow < stateRowsPerAiv_;
                 localRow += kTransposeTile) {
                for (uint32_t col = 0; col < kTile;
                     col += kTransposeTile) {
                    LocalFloatToHalfTile(
                        state[localRow * kTile + col],
                        stateT[(rowBegin + localRow) * outputCols +
                               headSlot * kTile + col], outputCols);
                }
            }
        } else {
            for (uint32_t localRow = 0; localRow < kRowsPerAiv;
                 localRow += kTransposeTile) {
                for (uint32_t col = 0; col < stateDim_;
                     col += kTransposeTile) {
                    LocalFloatToHalfTransposeTile(
                        state[localRow * stateDim_ + col],
                        stateT[col * outputCols + headSlot * kTile +
                               rowBegin + localRow], outputCols);
                }
            }
        }
    }

    __aicore__ inline void UpdateState(
        const GlobalTensor<float> &chunkState,
        const GlobalTensor<float> &dA, uint32_t headSlot = 0)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * stateRowsPerAiv_;
        const float decay = ScalarExp(dA.GetValue(ChunkSize - 1));
        auto contribution = stateContribution_.AllocTensor<float>();
        DataCopy(contribution, chunkState[rowBegin * stateRowStride_],
                 stateBlockElements_);
        stateContribution_.EnQue(contribution);
        contribution = stateContribution_.DeQue<float>();
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        Muls(state, state, decay, stateBlockElements_);
        PipeBarrier<PIPE_V>();
        Add(state, state, contribution, stateBlockElements_);
        stateContribution_.FreeTensor(contribution);
    }

    __aicore__ inline void StoreFinal(GlobalTensor<float> finalState,
                                      uint32_t headSlot = 0)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * stateRowsPerAiv_;
        auto result = finalOut_.AllocTensor<float>();
        Adds(result, state_.Get<float>()[headSlot * stateBlockElements_],
             0.0f, stateBlockElements_);
        finalOut_.EnQue(result);
        result = finalOut_.DeQue<float>();
        DataCopy(finalState[rowBegin * stateRowStride_], result,
                 stateBlockElements_);
        finalOut_.FreeTensor(result);
    }

    __aicore__ inline void Epilogue(
        const GlobalTensor<float> &yOff,
        const GlobalTensor<float> &yDiag,
        const GlobalTensor<float> &dA,
        const GlobalTensor<float> &x,
        const GlobalTensor<float> &z,
        GlobalTensor<float> out,
        uint32_t heads, uint32_t headSlot = 0,
        uint32_t yOffRowStride = kTile)
    {
        constexpr uint32_t blockElements = kTokenRowsPerAiv * kTile;
        const uint32_t rowBegin = GetSubBlockIdx() * kTokenRowsPerAiv;
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        auto matrix0 = matrix0_.Get<float>();
        auto matrix1 = matrix1_.Get<float>();

        auto daLocal = dAIn_.AllocTensor<float>();
        DataCopy(daLocal, dA[rowBegin], kTokenRowsPerAiv);
        dAIn_.EnQue(daLocal);
        daLocal = dAIn_.DeQue<float>();
        const uint32_t daSrcShape[2] = {kTokenRowsPerAiv, 1U};
        const uint32_t matrixShape[2] = {kTokenRowsPerAiv, kTile};
        Broadcast<float, 2, 1>(matrix0, daLocal, matrixShape,
                               daSrcShape, broadcastTmp);
        PipeBarrier<PIPE_V>();
        Exp(matrix0, matrix0, blockElements);
        PipeBarrier<PIPE_V>();

#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        auto yOffLocal = stateContribution_.AllocTensor<float>();
#else
        auto yOffLocal = yOffIn_.AllocTensor<float>();
#endif
        DataCopyParams yOffLoad{
            static_cast<uint16_t>(kTokenRowsPerAiv),
            static_cast<uint16_t>(kTile * sizeof(float) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((yOffRowStride - kTile) * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(yOffLocal, yOff[rowBegin * yOffRowStride], yOffLoad);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        stateContribution_.EnQue(yOffLocal);
        yOffLocal = stateContribution_.DeQue<float>();
#else
        yOffIn_.EnQue(yOffLocal);
        yOffLocal = yOffIn_.DeQue<float>();
#endif
        auto yDiagLocal = yDiagIn_.AllocTensor<float>();
        DataCopy(yDiagLocal, yDiag[rowBegin * kTile], blockElements);
        yDiagIn_.EnQue(yDiagLocal);
        yDiagLocal = yDiagIn_.DeQue<float>();
        auto outLocal = out_.AllocTensor<float>();
        Mul(yOffLocal, yOffLocal, matrix0, blockElements);
        PipeBarrier<PIPE_V>();
        Add(outLocal, yDiagLocal, yOffLocal, blockElements);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        stateContribution_.FreeTensor(yOffLocal);
#else
        yOffIn_.FreeTensor(yOffLocal);
#endif
        yDiagIn_.FreeTensor(yDiagLocal);
        dAIn_.FreeTensor(daLocal);
        PipeBarrier<PIPE_V>();

        DataCopyParams rawLoad{
            static_cast<uint16_t>(kTokenRowsPerAiv),
            static_cast<uint16_t>(kTile * sizeof(float) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((heads - 1) * kTile * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0};
        auto xLocal = xIn_.AllocTensor<float>();
        DataCopy(xLocal, x[rowBegin * heads * kTile], rawLoad);
        xIn_.EnQue(xLocal);
        xLocal = xIn_.DeQue<float>();
        Mul(xLocal, xLocal,
            dMatrix_.Get<float>()[headSlot * blockElements], blockElements);
        PipeBarrier<PIPE_V>();
        Add(outLocal, outLocal, xLocal, blockElements);
        xIn_.FreeTensor(xLocal);
        PipeBarrier<PIPE_V>();

#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        auto zLocal = xIn_.AllocTensor<float>();
#else
        auto zLocal = zIn_.AllocTensor<float>();
#endif
        DataCopy(zLocal, z[rowBegin * heads * kTile], rawLoad);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        xIn_.EnQue(zLocal);
        zLocal = xIn_.DeQue<float>();
#else
        zIn_.EnQue(zLocal);
        zLocal = zIn_.DeQue<float>();
#endif
        Muls(matrix1, zLocal, -1.0f, blockElements);
        PipeBarrier<PIPE_V>();
        Exp(matrix1, matrix1, blockElements);
        PipeBarrier<PIPE_V>();
        Adds(matrix1, matrix1, 1.0f, blockElements);
        PipeBarrier<PIPE_V>();
        Div(zLocal, zLocal, matrix1, blockElements);
        PipeBarrier<PIPE_V>();
        Mul(outLocal, outLocal, zLocal, blockElements);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        xIn_.FreeTensor(zLocal);
#else
        zIn_.FreeTensor(zLocal);
#endif

        out_.EnQue(outLocal);
        outLocal = out_.DeQue<float>();
        DataCopyParams rawStore{
            static_cast<uint16_t>(kTokenRowsPerAiv),
            static_cast<uint16_t>(kTile * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((heads - 1) * kTile * sizeof(float) /
                                  DEFAULT_C0_SIZE)};
        DataCopy(out[rowBegin * heads * kTile], outLocal, rawStore);
        out_.FreeTensor(outLocal);
    }

private:
    static constexpr uint32_t kTokenRowsPerAiv = ChunkSize / 2;
    __aicore__ inline void LocalFloatToHalfTile(
        const LocalTensor<float> &src, GlobalTensor<half> dst,
        uint32_t dstRowStride)
    {
        auto floatLocal = stateFloatIn_.AllocTensor<float>();
        DataCopyParams load{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((kTile - kTransposeTile) *
                                  sizeof(float) / DEFAULT_C0_SIZE),
            0};
        DataCopy(floatLocal, src, load);
        stateFloatIn_.EnQue(floatLocal);
        floatLocal = stateFloatIn_.DeQue<float>();
        auto halfLocal = stateHalfOut_.AllocTensor<half>();
        Cast(halfLocal, floatLocal, RoundMode::CAST_RINT,
             kTransposeElements);
        stateFloatIn_.FreeTensor(floatLocal);
        stateHalfOut_.EnQue(halfLocal);
        halfLocal = stateHalfOut_.DeQue<half>();
        DataCopyParams store{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((dstRowStride - kTransposeTile) *
                                  sizeof(half) / DEFAULT_C0_SIZE)};
        DataCopy(dst, halfLocal, store);
        stateHalfOut_.FreeTensor(halfLocal);
    }

    __aicore__ inline void LocalFloatToHalfTransposeTile(
        const LocalTensor<float> &src, GlobalTensor<half> dst,
        uint32_t dstRowStride)
    {
        auto floatLocal = stateFloatIn_.AllocTensor<float>();
        DataCopyParams load{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((stateDim_ - kTransposeTile) *
                                  sizeof(float) / DEFAULT_C0_SIZE),
            0};
        DataCopy(floatLocal, src, load);
        stateFloatIn_.EnQue(floatLocal);
        floatLocal = stateFloatIn_.DeQue<float>();
        auto halfLocal = stateHalfOut_.AllocTensor<half>();
        Cast(halfLocal, floatLocal, RoundMode::CAST_RINT,
             kTransposeElements);
        stateFloatIn_.FreeTensor(floatLocal);
        stateHalfOut_.EnQue(halfLocal);
        halfLocal = stateHalfOut_.DeQue<half>();
        auto transposed = transposeOut_.AllocTensor<half>();
        Transpose(transposed, halfLocal);
        stateHalfOut_.FreeTensor(halfLocal);
        transposeOut_.EnQue(transposed);
        transposed = transposeOut_.DeQue<half>();
        DataCopyParams store{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((dstRowStride - kTransposeTile) *
                                  sizeof(half) / DEFAULT_C0_SIZE)};
        DataCopy(dst, transposed, store);
        transposeOut_.FreeTensor(transposed);
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

    TBuf<TPosition::VECCALC> state_;
    TQue<TPosition::VECIN, 1> stateContribution_;
    TQue<TPosition::VECOUT, 1> finalOut_;
    TQue<TPosition::VECIN, 1> stateFloatIn_;
    TQue<TPosition::VECOUT, 1> stateHalfOut_;
    TQue<TPosition::VECOUT, 1> transposeOut_;
#ifndef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
    TQue<TPosition::VECIN, 1> dIn_;
    TQue<TPosition::VECIN, 1> yOffIn_;
    TQue<TPosition::VECIN, 1> zIn_;
#endif
    TQue<TPosition::VECIN, 1> dAIn_;
    TQue<TPosition::VECIN, 1> yDiagIn_;
    TQue<TPosition::VECIN, 1> xIn_;
    TQue<TPosition::VECOUT, 1> out_;
    TBuf<TPosition::VECCALC> matrix0_;
    TBuf<TPosition::VECCALC> matrix1_;
    TBuf<TPosition::VECCALC> dMatrix_;
    TBuf<TPosition::VECCALC> broadcastTmp_;
    uint32_t stateDim_ = 0;
    uint32_t stateRowsPerAiv_ = 0;
    uint32_t stateRowStride_ = 0;
    uint32_t stateBlockElements_ = 0;
    uint32_t headsPerTask_ = 1;
    bool stateNpLayout_ = false;
};

using VectorStateEpilogue = VectorStateEpilogueT<kTile>;

#ifndef MAMBA2_STATE_EPILOGUE_COMPONENT_ONLY
template <uint32_t ChunkSize>
class KernelMamba2SsdStateEpilogueT {
public:
    __aicore__ inline void Init(
        GM_ADDR chunkStates, GM_ADDR dACumsum, GM_ADDR cCube,
        GM_ADDR yDiag, GM_ADDR x, GM_ADDR d, GM_ADDR z,
        GM_ADDR initialStates, GM_ADDR out, GM_ADDR finalState,
        GM_ADDR workspace, const Mamba2SsdStateEpilogueTilingData &tiling,
        TPipe *pipe)
    {
        tiling_ = tiling;
        const uint64_t headChunks = static_cast<uint64_t>(tiling_.batch) *
                                    tiling_.heads * tiling_.chunks;
        const uint64_t groupChunks = static_cast<uint64_t>(tiling_.batch) *
                                     tiling_.groups * tiling_.chunks;
        chunkStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(chunkStates),
            headChunks * kTile * tiling_.stateDim);
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum),
                              headChunks * ChunkSize);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube),
                             groupChunks * ChunkSize * tiling_.stateDim);
        yDiagGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yDiag),
                                 headChunks * ChunkSize * kTile);
        const uint64_t rawElements = static_cast<uint64_t>(tiling_.batch) *
                                     tiling_.chunks * ChunkSize *
                                     tiling_.heads * kTile;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), rawElements);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z), rawElements);
        dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d),
                             static_cast<uint64_t>(tiling_.heads) * kTile);
        initialGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(initialStates),
            static_cast<uint64_t>(tiling_.batch) * tiling_.heads *
                kTile * tiling_.stateDim);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out),
                               rawElements);
        finalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(finalState),
                                 static_cast<uint64_t>(tiling_.batch) *
                                     tiling_.heads * kTile * tiling_.stateDim);

        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) * tiling_.workspaceBytesPerCore;
        projectionCols_ = tiling_.headsPerTask * kTile;
        stateTHalfElements_ = tiling_.stateDim * projectionCols_;
        projectionElements_ = ChunkSize * projectionCols_;
        const uint32_t twoStateBytes =
            2 * stateTHalfElements_ * sizeof(half);
        workspaceHalf_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(coreWorkspace),
            2 * stateTHalfElements_);
        workspaceFloat_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(coreWorkspace + twoStateBytes),
            2 * projectionElements_);
        if ASCEND_IS_AIC matmul_.Init(*pipe);
        if ASCEND_IS_AIV {
            vector_.Init(pipe, tiling_.stateDim, tiling_.headsPerTask);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t task = coreIdx_; task < tiling_.taskCount;
             task += tiling_.usedCoreNum) {
            uint32_t batch;
            uint32_t group;
            uint32_t firstHead;
            if (tiling_.headsPerTask == 2) {
                const uint32_t pairsPerGroup = tiling_.headsPerGroup / 2;
                const uint32_t pairsPerBatch = tiling_.groups * pairsPerGroup;
                batch = task / pairsPerBatch;
                const uint32_t batchTask = task % pairsPerBatch;
                group = batchTask / pairsPerGroup;
                firstHead = group * tiling_.headsPerGroup +
                            (batchTask % pairsPerGroup) * 2;
            } else {
                firstHead = task % tiling_.heads;
                batch = task / tiling_.heads;
                group = firstHead / tiling_.headsPerGroup;
            }
            if ASCEND_IS_AIV {
                for (uint32_t headSlot = 0;
                     headSlot < tiling_.headsPerTask; ++headSlot) {
                    const uint32_t head = firstHead + headSlot;
                    const uint64_t stream =
                        static_cast<uint64_t>(batch) * tiling_.heads + head;
                    vector_.InitializeState(
                        initialGm_[stream * kTile * tiling_.stateDim],
                        tiling_.hasInitial, headSlot);
                    vector_.PrepareD(
                        dGm_[static_cast<uint64_t>(head) * kTile], headSlot);
                }
                const uint32_t preload = tiling_.chunks < 2 ?
                                         tiling_.chunks : 2;
                for (uint32_t chunk = 0; chunk < preload; ++chunk) {
                    PrepareChunk(batch, firstHead, chunk, chunk);
                }
                for (uint32_t chunk = 0; chunk < tiling_.chunks; ++chunk) {
                    const uint32_t slot = chunk & 1U;
                    WaitCubeReady(slot);
                    RunEpilogue(batch, firstHead, chunk, slot);
                    const uint32_t next = chunk + 2;
                    if (next < tiling_.chunks) {
                        PrepareChunk(batch, firstHead, next, slot);
                    }
                }
                for (uint32_t headSlot = 0;
                     headSlot < tiling_.headsPerTask; ++headSlot) {
                    const uint32_t head = firstHead + headSlot;
                    const uint64_t stream =
                        static_cast<uint64_t>(batch) * tiling_.heads + head;
                    vector_.StoreFinal(
                        finalGm_[stream * kTile * tiling_.stateDim],
                        headSlot);
                }
            }
            if ASCEND_IS_AIC {
                for (uint32_t chunk = 0; chunk < tiling_.chunks; ++chunk) {
                    const uint32_t slot = chunk & 1U;
                    const uint32_t cIndex =
                        (batch * tiling_.chunks + chunk) * tiling_.groups +
                        group;
                    WaitStateReady(slot);
                    for (uint32_t row = 0; row < ChunkSize; row += kTile) {
                        matmul_.ComputeBlock(
                            cGm_[static_cast<uint64_t>(cIndex) * ChunkSize *
                                 tiling_.stateDim + row * tiling_.stateDim],
                            workspaceHalf_[slot * stateTHalfElements_],
                            workspaceFloat_[slot * projectionElements_ +
                                            row * projectionCols_],
                            tiling_.stateDim, tiling_.stateDim,
                            projectionCols_, projectionCols_,
                            projectionCols_);
                    }
                    SetCubeReady(slot);
                }
            }
        }
    }

private:
    __aicore__ inline void PrepareChunk(uint32_t batch, uint32_t firstHead,
                                        uint32_t chunk, uint32_t slot)
    {
        for (uint32_t headSlot = 0;
             headSlot < tiling_.headsPerTask; ++headSlot) {
            const uint32_t head = firstHead + headSlot;
            const uint64_t stream =
                static_cast<uint64_t>(batch) * tiling_.heads + head;
            const uint64_t headChunk = stream * tiling_.chunks + chunk;
            vector_.PrepareState(
                workspaceHalf_[slot * stateTHalfElements_], headSlot,
                projectionCols_);
            vector_.UpdateState(
                chunkStateGm_[headChunk * kTile * tiling_.stateDim],
                dAGm_[headChunk * ChunkSize], headSlot);
        }
        SetStateReady(slot);
    }

    __aicore__ inline void RunEpilogue(uint32_t batch, uint32_t firstHead,
                                       uint32_t chunk, uint32_t slot)
    {
        for (uint32_t headSlot = 0;
             headSlot < tiling_.headsPerTask; ++headSlot) {
            const uint32_t head = firstHead + headSlot;
            const uint64_t headChunk =
                (static_cast<uint64_t>(batch) * tiling_.heads + head) *
                    tiling_.chunks + chunk;
            const uint64_t rawOffset =
                (static_cast<uint64_t>(batch) * tiling_.chunks * ChunkSize +
                 static_cast<uint64_t>(chunk) * ChunkSize) *
                    tiling_.heads * kTile +
                static_cast<uint64_t>(head) * kTile;
            vector_.Epilogue(
                workspaceFloat_[slot * projectionElements_ +
                                headSlot * kTile],
                yDiagGm_[headChunk * ChunkSize * kTile],
                dAGm_[headChunk * ChunkSize], xGm_[rawOffset], zGm_[rawOffset],
                outGm_[rawOffset], tiling_.heads, headSlot,
                projectionCols_);
        }
    }

    __aicore__ inline void SetStateReady(uint32_t slot)
    {
        if (slot == 0) CrossCoreSetFlag<0x2, PIPE_MTE3>(kStateReady0);
        else CrossCoreSetFlag<0x2, PIPE_MTE3>(kStateReady1);
    }
    __aicore__ inline void WaitStateReady(uint32_t slot)
    {
        if (slot == 0) CrossCoreWaitFlag<0x2>(kStateReady0);
        else CrossCoreWaitFlag<0x2>(kStateReady1);
    }
    __aicore__ inline void SetCubeReady(uint32_t slot)
    {
        if (slot == 0) CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeReady0);
        else CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeReady1);
    }
    __aicore__ inline void WaitCubeReady(uint32_t slot)
    {
        if (slot == 0) CrossCoreWaitFlag<0x2>(kCubeReady0);
        else CrossCoreWaitFlag<0x2>(kCubeReady1);
    }

    Mamba2DynamicMatmul<half, float> matmul_;
    VectorStateEpilogueT<ChunkSize> vector_;
    GlobalTensor<float> chunkStateGm_;
    GlobalTensor<float> dAGm_;
    GlobalTensor<half> cGm_;
    GlobalTensor<float> yDiagGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> dGm_;
    GlobalTensor<float> zGm_;
    GlobalTensor<float> initialGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<float> finalGm_;
    GlobalTensor<half> workspaceHalf_;
    GlobalTensor<float> workspaceFloat_;
    Mamba2SsdStateEpilogueTilingData tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t stateTHalfElements_ = 0;
    uint32_t projectionElements_ = 0;
    uint32_t projectionCols_ = 0;
};
#endif
} // namespace

#ifndef MAMBA2_STATE_EPILOGUE_COMPONENT_ONLY
extern "C" __global__ __aicore__ void mamba2_ssd_state_epilogue(
    GM_ADDR chunk_states, GM_ADDR d_a_cumsum, GM_ADDR c_cube,
    GM_ADDR y_diag, GM_ADDR x, GM_ADDR d, GM_ADDR z,
    GM_ADDR initial_states, GM_ADDR out, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(4)) {
        KernelMamba2SsdStateEpilogueT<64> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, final_state, workspace, tilingData,
                &pipe);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        KernelMamba2SsdStateEpilogueT<128> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, final_state, workspace, tilingData,
                &pipe);
        op.Process();
    }
}
#endif
