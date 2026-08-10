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

// The state-update, projection epilogue, D preload and gate phases do not
// overlap on AIV.  Reusing their VECIN queues releases enough UB for one
// complete 32-token public-layout slab and avoids executing every command
// sequence twice at 16-token granularity.
#define MAMBA2_STATE_EPILOGUE_SHARE_QUEUES

namespace {
constexpr uint32_t kTile = 64;
constexpr uint32_t kTileElements = kTile * kTile;
constexpr uint32_t kLogicalSlabCount = 2;
constexpr uint32_t kRowsPerSlab = kTile / kLogicalSlabCount;
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements = kTransposeTile * kTransposeTile;
constexpr uint32_t kPublicGroupHeads = 4;
constexpr uint32_t kPublicGroupRows = 32;
constexpr uint32_t kPublicGroupWidth = kPublicGroupHeads * kTile;
constexpr uint32_t kPublicGroupElements =
    kPublicGroupRows * kPublicGroupWidth;
constexpr uint16_t kStateReady0 = 0x8;
constexpr uint16_t kStateReady1 = 0x9;
constexpr uint16_t kCubeReady0 = 0xA;
constexpr uint16_t kCubeReady1 = 0xB;

template <uint32_t ChunkSize,
          uint32_t TokenSlabCount = kLogicalSlabCount>
class VectorStateEpilogueT {
public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t stateDim,
                                uint32_t headsPerTask = 1,
                                uint32_t aivPerAic = 2)
    {
        aivPerAic_ = aivPerAic;
        stateDim_ = stateDim;
        stateNpLayout_ = stateDim_ == kTile || ChunkSize == 128;
        stateRowsPerAiv_ = stateNpLayout_ ? stateDim_ / aivPerAic_
                                          : kTile / aivPerAic_;
        stateRowStride_ = stateNpLayout_ ? kTile : stateDim_;
        stateBlockElements_ = stateRowsPerAiv_ * stateRowStride_;
        stateSlabRows_ = stateNpLayout_ ? stateDim_ / kLogicalSlabCount
                                        : kRowsPerSlab;
        stateSlabElements_ = stateSlabRows_ * stateRowStride_;
        headsPerTask_ = headsPerTask;
        pipe->InitBuffer(state_, headsPerTask_ * stateBlockElements_ *
                                     sizeof(float));
        pipe->InitBuffer(stateContribution_, 1,
                         stateSlabElements_ * sizeof(float));
        pipe->InitBuffer(finalOut_, 1,
                         stateSlabElements_ * sizeof(float));
        pipe->InitBuffer(stateFloatIn_, 1,
                         kTransposeElements * sizeof(float));
        const uint32_t publicIoElements =
            headsPerTask_ == kPublicGroupHeads ? kPublicGroupElements
                                               : kTokenRowsPerSlab * kTile;
        pipe->InitBuffer(stateHalfOut_, 1,
                         (headsPerTask_ == kPublicGroupHeads
                              ? kPublicGroupElements
                              : kTransposeElements) * sizeof(half));
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
                         kTokenRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(zIn_, 1,
                         publicIoElements * sizeof(float));
#endif
        // dA rows and D are consumed at disjoint phases.  A single 64-value
        // queue covers both and saves one scarce VECIN queue event when this
        // component is embedded in a larger MIX kernel.
        pipe->InitBuffer(dAIn_, 1, kTile * sizeof(float));
        pipe->InitBuffer(yDiagIn_, 2,
                         kTokenRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(xIn_, 1,
                         publicIoElements * sizeof(float));
        pipe->InitBuffer(out_, 1,
                         publicIoElements * sizeof(float));
        pipe->InitBuffer(matrix0_,
                         kTokenRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(matrix1_,
                         kTokenRowsPerSlab * kTile * sizeof(float));
        pipe->InitBuffer(dRows_, headsPerTask_ * kTile * sizeof(float));
        pipe->InitBuffer(broadcastTmp_,
                         2 * kTokenRowsPerSlab * kTile * sizeof(uint8_t));
    }

    __aicore__ inline void PrepareD(const GlobalTensor<float> &d,
                                    uint32_t headSlot = 0)
    {
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
        Adds(dRows_.Get<float>()[headSlot * kTile], dRow, 0.0f, kTile);
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
            for (uint32_t slab = GetSubBlockIdx();
                 slab < kLogicalSlabCount; slab += aivPerAic_) {
                const uint32_t globalRow = slab * stateSlabRows_;
                const uint32_t localRow = LocalStateRow(slab);
                auto initialLocal = stateContribution_.AllocTensor<float>();
                DataCopy(initialLocal,
                         initial[globalRow * stateRowStride_],
                         stateSlabElements_);
                stateContribution_.EnQue(initialLocal);
                initialLocal = stateContribution_.DeQue<float>();
                Adds(state[localRow * stateRowStride_], initialLocal, 0.0f,
                     stateSlabElements_);
                stateContribution_.FreeTensor(initialLocal);
            }
        } else {
            Duplicate(state, 0.0f, stateBlockElements_);
        }
    }

    __aicore__ inline void PrepareState(
        GlobalTensor<half> stateT, uint32_t headSlot = 0,
        uint32_t outputCols = kTile)
    {
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        if (stateNpLayout_) {
            for (uint32_t slab = GetSubBlockIdx();
                 slab < kLogicalSlabCount; slab += aivPerAic_) {
                const uint32_t globalRow = slab * stateSlabRows_;
                const uint32_t localRow = LocalStateRow(slab);
                auto stateHalf = finalOut_.AllocTensor<half>();
                Cast(stateHalf, state[localRow * stateRowStride_],
                     RoundMode::CAST_RINT, stateSlabElements_);
                finalOut_.EnQue(stateHalf);
                stateHalf = finalOut_.DeQue<half>();
                // For a paired N64 task the two head states share a wide
                // [N, 2P] workspace.  Preserve contiguous rows in UB and use
                // one strided MTE3 descriptor to place this head's P columns;
                // issuing one copy per row would make Scalar/MTE command
                // overhead dominate the small projection.
                DataCopyExtParams store{
                    static_cast<uint16_t>(stateSlabRows_),
                    static_cast<uint32_t>(kTile * sizeof(half)),
                    0,
                    static_cast<uint32_t>(
                        (outputCols - kTile) * sizeof(half)),
                    0};
                DataCopyPad(
                    stateT[globalRow * outputCols + headSlot * kTile],
                    stateHalf, store);
                finalOut_.FreeTensor(stateHalf);
            }
        } else {
            for (uint32_t slab = GetSubBlockIdx();
                 slab < kLogicalSlabCount; slab += aivPerAic_) {
                const uint32_t globalRow = slab * stateSlabRows_;
                const uint32_t localBase = LocalStateRow(slab);
                for (uint32_t local = 0; local < stateSlabRows_;
                     local += kTransposeTile) {
                    for (uint32_t col = 0; col < stateDim_;
                         col += kTransposeTile) {
                        LocalFloatToHalfTransposeTile(
                            state[(localBase + local) * stateDim_ + col],
                            stateT[col * outputCols + headSlot * kTile +
                                   globalRow + local], outputCols);
                    }
                }
            }
        }
    }

    __aicore__ inline void UpdateState(
        const GlobalTensor<float> &chunkState,
        const GlobalTensor<float> &dA, uint32_t headSlot = 0)
    {
        const float decay = ScalarExp(dA.GetValue(ChunkSize - 1));
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        for (uint32_t slab = GetSubBlockIdx(); slab < kLogicalSlabCount;
             slab += aivPerAic_) {
            const uint32_t globalRow = slab * stateSlabRows_;
            const uint32_t localRow = LocalStateRow(slab);
            auto contribution = stateContribution_.AllocTensor<float>();
            DataCopy(contribution,
                     chunkState[globalRow * stateRowStride_],
                     stateSlabElements_);
            stateContribution_.EnQue(contribution);
            contribution = stateContribution_.DeQue<float>();
            auto stateSlab = state[localRow * stateRowStride_];
            Muls(stateSlab, stateSlab, decay, stateSlabElements_);
            PipeBarrier<PIPE_V>();
            Add(stateSlab, stateSlab, contribution, stateSlabElements_);
            stateContribution_.FreeTensor(contribution);
        }
    }

    __aicore__ inline void UpdateStateGrouped(
        const GlobalTensor<float> &chunkState,
        const GlobalTensor<float> &dA, uint32_t headSlot,
        uint32_t groupedRowStride)
    {
        const float decay = ScalarExp(dA.GetValue(ChunkSize - 1));
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        for (uint32_t slab = GetSubBlockIdx(); slab < kLogicalSlabCount;
             slab += aivPerAic_) {
            const uint32_t globalRow = slab * stateSlabRows_;
            const uint32_t localRow = LocalStateRow(slab);
            auto contribution = stateContribution_.AllocTensor<float>();
            DataCopyParams load{
                static_cast<uint16_t>(stateSlabRows_),
                static_cast<uint16_t>(kTile * sizeof(float) /
                                      DEFAULT_C0_SIZE),
                static_cast<uint16_t>((groupedRowStride - kTile) *
                                      sizeof(float) / DEFAULT_C0_SIZE),
                0};
            DataCopy(contribution,
                     chunkState[globalRow * groupedRowStride], load);
            stateContribution_.EnQue(contribution);
            contribution = stateContribution_.DeQue<float>();
            auto stateSlab = state[localRow * stateRowStride_];
            Muls(stateSlab, stateSlab, decay, stateSlabElements_);
            PipeBarrier<PIPE_V>();
            Add(stateSlab, stateSlab, contribution, stateSlabElements_);
            stateContribution_.FreeTensor(contribution);
        }
    }

    __aicore__ inline void StoreFinal(GlobalTensor<float> finalState,
                                      uint32_t headSlot = 0)
    {
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        for (uint32_t slab = GetSubBlockIdx(); slab < kLogicalSlabCount;
             slab += aivPerAic_) {
            const uint32_t globalRow = slab * stateSlabRows_;
            const uint32_t localRow = LocalStateRow(slab);
            auto result = finalOut_.AllocTensor<float>();
            Adds(result, state[localRow * stateRowStride_], 0.0f,
                 stateSlabElements_);
            finalOut_.EnQue(result);
            result = finalOut_.DeQue<float>();
            DataCopy(finalState[globalRow * stateRowStride_], result,
                     stateSlabElements_);
            finalOut_.FreeTensor(result);
        }
    }

    __aicore__ inline void StoreHalfInternal(
        GlobalTensor<half> stateStart, uint32_t headSlot = 0)
    {
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        for (uint32_t slab = GetSubBlockIdx(); slab < kLogicalSlabCount;
             slab += aivPerAic_) {
            const uint32_t globalRow = slab * stateSlabRows_;
            const uint32_t localRow = LocalStateRow(slab);
            auto result = finalOut_.AllocTensor<half>();
            Cast(result, state[localRow * stateRowStride_],
                 RoundMode::CAST_RINT, stateSlabElements_);
            finalOut_.EnQue(result);
            result = finalOut_.DeQue<half>();
            DataCopyExtParams store{
                1,
                static_cast<uint32_t>(stateSlabElements_ * sizeof(half)),
                0,
                0,
                0};
            DataCopyPad(stateStart[globalRow * stateRowStride_], result, store);
            finalOut_.FreeTensor(result);
        }
    }

    __aicore__ inline void StoreHalfGrouped(
        GlobalTensor<half> stateStart, uint32_t headSlot,
        uint32_t groupedRowStride)
    {
        auto state = state_.Get<float>()[headSlot * stateBlockElements_];
        for (uint32_t slab = GetSubBlockIdx(); slab < kLogicalSlabCount;
             slab += aivPerAic_) {
            const uint32_t globalRow = slab * stateSlabRows_;
            const uint32_t localRow = LocalStateRow(slab);
            auto result = finalOut_.AllocTensor<half>();
            Cast(result, state[localRow * stateRowStride_],
                 RoundMode::CAST_RINT, stateSlabElements_);
            finalOut_.EnQue(result);
            result = finalOut_.DeQue<half>();
            DataCopyParams store{
                static_cast<uint16_t>(stateSlabRows_),
                static_cast<uint16_t>(kTile * sizeof(half) /
                                      DEFAULT_C0_SIZE),
                0,
                static_cast<uint16_t>((groupedRowStride - kTile) *
                                      sizeof(half) / DEFAULT_C0_SIZE)};
            DataCopy(stateStart[globalRow * groupedRowStride], result, store);
            finalOut_.FreeTensor(result);
        }
    }

    template <bool SavePreGate = false, typename YOffT = float,
              typename PreGateT = float>
    __aicore__ inline void Epilogue(
        const GlobalTensor<YOffT> &yOff,
        const GlobalTensor<float> &yDiag,
        const GlobalTensor<float> &dA,
        const GlobalTensor<float> &x,
        const GlobalTensor<float> &z,
        GlobalTensor<float> out,
        GlobalTensor<PreGateT> preGate,
        uint32_t heads, uint32_t headSlot = 0,
        uint32_t yOffRowStride = kTile, uint32_t rowBegin = 0,
        uint32_t yDiagRowStride = kTile)
    {
        constexpr uint32_t blockElements = kTokenRowsPerSlab * kTile;
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        auto matrix0 = matrix0_.Get<float>();
        auto matrix1 = matrix1_.Get<float>();

        auto daLocal = dAIn_.AllocTensor<float>();
        DataCopy(daLocal, dA[rowBegin], kTokenRowsPerSlab);
        dAIn_.EnQue(daLocal);
        daLocal = dAIn_.DeQue<float>();
        const uint32_t daSrcShape[2] = {kTokenRowsPerSlab, 1U};
        const uint32_t matrixShape[2] = {kTokenRowsPerSlab, kTile};
        Broadcast<float, 2, 1>(matrix0, daLocal, matrixShape,
                               daSrcShape, broadcastTmp);
        PipeBarrier<PIPE_V>();
        Exp(matrix0, matrix0, blockElements);
        PipeBarrier<PIPE_V>();

#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        auto yOffLocal = stateContribution_.AllocTensor<YOffT>();
#else
        auto yOffLocal = yOffIn_.AllocTensor<YOffT>();
#endif
        DataCopyParams yOffLoad{
            static_cast<uint16_t>(kTokenRowsPerSlab),
            static_cast<uint16_t>(kTile * sizeof(YOffT) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((yOffRowStride - kTile) * sizeof(YOffT) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(yOffLocal, yOff[rowBegin * yOffRowStride], yOffLoad);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        stateContribution_.EnQue(yOffLocal);
        yOffLocal = stateContribution_.DeQue<YOffT>();
#else
        yOffIn_.EnQue(yOffLocal);
        yOffLocal = yOffIn_.DeQue<YOffT>();
#endif
        auto yDiagLocal = yDiagIn_.AllocTensor<float>();
        DataCopyParams yDiagLoad{
            static_cast<uint16_t>(kTokenRowsPerSlab),
            static_cast<uint16_t>(kTile * sizeof(float) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((yDiagRowStride - kTile) * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0};
        DataCopy(yDiagLocal, yDiag[rowBegin * yDiagRowStride], yDiagLoad);
        yDiagIn_.EnQue(yDiagLocal);
        yDiagLocal = yDiagIn_.DeQue<float>();
        auto outLocal = out_.AllocTensor<float>();
        if constexpr (sizeof(YOffT) == sizeof(half)) {
            Cast(matrix1, yOffLocal, RoundMode::CAST_NONE, blockElements);
            PipeBarrier<PIPE_V>();
            Mul(matrix1, matrix1, matrix0, blockElements);
            PipeBarrier<PIPE_V>();
            Add(outLocal, yDiagLocal, matrix1, blockElements);
        } else {
            Mul(yOffLocal, yOffLocal, matrix0, blockElements);
            PipeBarrier<PIPE_V>();
            Add(outLocal, yDiagLocal, yOffLocal, blockElements);
        }
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        stateContribution_.FreeTensor(yOffLocal);
#else
        yOffIn_.FreeTensor(yOffLocal);
#endif
        yDiagIn_.FreeTensor(yDiagLocal);
        dAIn_.FreeTensor(daLocal);
        PipeBarrier<PIPE_V>();

        DataCopyParams rawLoad{
            static_cast<uint16_t>(kTokenRowsPerSlab),
            static_cast<uint16_t>(kTile * sizeof(float) / DEFAULT_C0_SIZE),
            static_cast<uint16_t>((heads - 1) * kTile * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0};
        auto xLocal = xIn_.AllocTensor<float>();
        DataCopy(xLocal, x[rowBegin * heads * kTile], rawLoad);
        xIn_.EnQue(xLocal);
        xLocal = xIn_.DeQue<float>();
        const BinaryRepeatParams dBroadcast(
            1, 1, 1, 8, 8, 0);
        Mul(xLocal, xLocal, dRows_.Get<float>()[headSlot * kTile],
            kTile, kTokenRowsPerSlab, dBroadcast);
        PipeBarrier<PIPE_V>();
        Add(outLocal, outLocal, xLocal, blockElements);
        xIn_.FreeTensor(xLocal);
        PipeBarrier<PIPE_V>();

        DataCopyParams rawStore{
            static_cast<uint16_t>(kTokenRowsPerSlab),
            static_cast<uint16_t>(kTile * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((heads - 1) * kTile * sizeof(float) /
                                  DEFAULT_C0_SIZE)};
        if constexpr (SavePreGate) {
            auto preGateHalf = matrix0_.Get<half>();
            Cast(preGateHalf, outLocal, RoundMode::CAST_RINT, blockElements);
            event_t toMte3 = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(toMte3);
            WaitFlag<HardEvent::V_MTE3>(toMte3);
            DataCopyParams preGateStore{
                static_cast<uint16_t>(kTokenRowsPerSlab),
                static_cast<uint16_t>(kTile * sizeof(half) /
                                      DEFAULT_C0_SIZE),
                0,
                static_cast<uint16_t>((heads - 1) * kTile * sizeof(half) /
                                      DEFAULT_C0_SIZE)};
            DataCopy(preGate[rowBegin * heads * kTile], preGateHalf,
                     preGateStore);
            event_t toVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(toVector);
            WaitFlag<HardEvent::MTE3_V>(toVector);
        }

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
        DataCopy(out[rowBegin * heads * kTile], outLocal, rawStore);
        out_.FreeTensor(outLocal);
    }

    // The public tensor layout is [B, L, H, P].  Four heads belonging to the
    // same group are adjacent for every token, so loading them as one
    // [16, 4, 64] tile replaces four high-stride GM transactions with one.
    // Computation remains head-local in UB; only the GM transfer granularity
    // changes.  This path is used by the 910B 1AIC:2AIV four-head schedule.
    template <bool SavePreGate = false, typename YOffT = float,
              typename PreGateT = half>
    __aicore__ inline void EpiloguePublicGroup4(
        const GlobalTensor<YOffT> &yOff,
        const GlobalTensor<float> &yDiag,
        const GlobalTensor<float> &dA,
        const GlobalTensor<float> &x,
        const GlobalTensor<float> &z,
        GlobalTensor<float> out,
        GlobalTensor<PreGateT> preGate,
        uint32_t heads, uint32_t yOffRowStride,
        uint32_t yDiagHeadStride, uint32_t dAHeadStride,
        uint32_t rowBegin)
    {
        constexpr uint32_t headElements = kPublicGroupRows * kTile;
        constexpr uint64_t vectorMask = kTile;
        constexpr uint8_t repeats = kPublicGroupRows;
        auto broadcastTmp = broadcastTmp_.Get<uint8_t>();
        auto matrix0 = matrix0_.Get<float>();
        auto matrix1 = matrix1_.Get<float>();

        const DataCopyParams groupLoad{
            static_cast<uint16_t>(kPublicGroupRows),
            static_cast<uint16_t>(kPublicGroupWidth * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((heads - kPublicGroupHeads) * kTile *
                                  sizeof(float) / DEFAULT_C0_SIZE),
            0};
        auto xGroup = xIn_.AllocTensor<float>();
        DataCopy(xGroup, x[rowBegin * heads * kTile], groupLoad);
        xIn_.EnQue(xGroup);
        xGroup = xIn_.DeQue<float>();
        auto outGroup = out_.AllocTensor<float>();

        const BinaryRepeatParams groupWithContiguous(
            1, 1, 1, 32, 32, 8);
        const BinaryRepeatParams groupWithD(
            1, 1, 1, 32, 32, 0);
        const uint32_t daSrcShape[2] = {kPublicGroupRows, 1U};
        const uint32_t matrixShape[2] = {kPublicGroupRows, kTile};
        auto yDiagLocal = yDiagIn_.AllocTensor<float>();
        DataCopy(
            yDiagLocal, yDiag[rowBegin * kTile], headElements);
        yDiagIn_.EnQue(yDiagLocal);
        for (uint32_t headSlot = 0; headSlot < kPublicGroupHeads;
             ++headSlot) {
            auto daLocal = dAIn_.AllocTensor<float>();
            DataCopy(daLocal,
                     dA[headSlot * dAHeadStride + rowBegin],
                     kPublicGroupRows);
            dAIn_.EnQue(daLocal);
            daLocal = dAIn_.DeQue<float>();
            Broadcast<float, 2, 1>(matrix0, daLocal, matrixShape,
                                   daSrcShape, broadcastTmp);
            PipeBarrier<PIPE_V>();
            Exp(matrix0, matrix0, headElements);
            PipeBarrier<PIPE_V>();

#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
            auto yOffLocal = stateContribution_.AllocTensor<YOffT>();
#else
            auto yOffLocal = yOffIn_.AllocTensor<YOffT>();
#endif
            const DataCopyParams yOffLoad{
                static_cast<uint16_t>(kPublicGroupRows),
                static_cast<uint16_t>(kTile * sizeof(YOffT) /
                                      DEFAULT_C0_SIZE),
                static_cast<uint16_t>((yOffRowStride - kTile) *
                                      sizeof(YOffT) / DEFAULT_C0_SIZE),
                0};
            DataCopy(yOffLocal,
                     yOff[headSlot * kTile + rowBegin * yOffRowStride],
                     yOffLoad);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
            stateContribution_.EnQue(yOffLocal);
            yOffLocal = stateContribution_.DeQue<YOffT>();
#else
            yOffIn_.EnQue(yOffLocal);
            yOffLocal = yOffIn_.DeQue<YOffT>();
#endif
            if constexpr (sizeof(YOffT) == sizeof(half)) {
                Cast(matrix1, yOffLocal, RoundMode::CAST_NONE,
                     headElements);
                PipeBarrier<PIPE_V>();
                Mul(matrix1, matrix1, matrix0, headElements);
            } else {
                Mul(matrix1, yOffLocal, matrix0, headElements);
            }
            PipeBarrier<PIPE_V>();

            yDiagLocal = yDiagIn_.DeQue<float>();
            if (headSlot + 1U < kPublicGroupHeads) {
                auto yDiagNext = yDiagIn_.AllocTensor<float>();
                DataCopy(
                    yDiagNext,
                    yDiag[(headSlot + 1U) * yDiagHeadStride +
                          rowBegin * kTile],
                    headElements);
                yDiagIn_.EnQue(yDiagNext);
            }
            Add(matrix1, matrix1, yDiagLocal, headElements);
            PipeBarrier<PIPE_V>();

            const uint32_t groupHeadOffset = headSlot * kTile;
            Mul(outGroup[groupHeadOffset], xGroup[groupHeadOffset],
                dRows_.Get<float>()[headSlot * kTile], vectorMask, repeats,
                groupWithD);
            PipeBarrier<PIPE_V>();
            Add(outGroup[groupHeadOffset], outGroup[groupHeadOffset],
                matrix1, vectorMask, repeats, groupWithContiguous);
            PipeBarrier<PIPE_V>();

#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
            stateContribution_.FreeTensor(yOffLocal);
#else
            yOffIn_.FreeTensor(yOffLocal);
#endif
            yDiagIn_.FreeTensor(yDiagLocal);
            dAIn_.FreeTensor(daLocal);
        }
        xIn_.FreeTensor(xGroup);

        if constexpr (SavePreGate) {
            auto preGateGroup = stateHalfOut_.AllocTensor<half>();
            const UnaryRepeatParams castGroupLane(1, 1, 16, 32);
            for (uint32_t headSlot = 0; headSlot < kPublicGroupHeads;
                 ++headSlot) {
                const uint32_t groupHeadOffset = headSlot * kTile;
                Cast(preGateGroup[groupHeadOffset],
                     outGroup[groupHeadOffset], RoundMode::CAST_RINT,
                     vectorMask, repeats, castGroupLane);
            }
            stateHalfOut_.EnQue(preGateGroup);
            preGateGroup = stateHalfOut_.DeQue<half>();
            const DataCopyParams preGateStore{
                static_cast<uint16_t>(kPublicGroupRows),
                static_cast<uint16_t>(kPublicGroupWidth * sizeof(half) /
                                      DEFAULT_C0_SIZE),
                0,
                static_cast<uint16_t>((heads - kPublicGroupHeads) * kTile *
                                      sizeof(half) / DEFAULT_C0_SIZE)};
            DataCopy(preGate[rowBegin * heads * kTile], preGateGroup,
                     preGateStore);
            stateHalfOut_.FreeTensor(preGateGroup);
        }

#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        auto zGroup = xIn_.AllocTensor<float>();
#else
        auto zGroup = zIn_.AllocTensor<float>();
#endif
        DataCopy(zGroup, z[rowBegin * heads * kTile], groupLoad);
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        xIn_.EnQue(zGroup);
        zGroup = xIn_.DeQue<float>();
#else
        zIn_.EnQue(zGroup);
        zGroup = zIn_.DeQue<float>();
#endif
        const UnaryRepeatParams gatherGroupLane(1, 1, 8, 32);
        for (uint32_t headSlot = 0; headSlot < kPublicGroupHeads;
             ++headSlot) {
            const uint32_t groupHeadOffset = headSlot * kTile;
            Adds(matrix0, zGroup[groupHeadOffset], 0.0f, vectorMask,
                 repeats, gatherGroupLane);
            PipeBarrier<PIPE_V>();
            Muls(matrix1, matrix0, -1.0f, headElements);
            PipeBarrier<PIPE_V>();
            Exp(matrix1, matrix1, headElements);
            PipeBarrier<PIPE_V>();
            Adds(matrix1, matrix1, 1.0f, headElements);
            PipeBarrier<PIPE_V>();
            Div(matrix0, matrix0, matrix1, headElements);
            PipeBarrier<PIPE_V>();
            Mul(outGroup[groupHeadOffset], outGroup[groupHeadOffset],
                matrix0, vectorMask, repeats, groupWithContiguous);
            PipeBarrier<PIPE_V>();
        }
#ifdef MAMBA2_STATE_EPILOGUE_SHARE_QUEUES
        xIn_.FreeTensor(zGroup);
#else
        zIn_.FreeTensor(zGroup);
#endif

        out_.EnQue(outGroup);
        outGroup = out_.DeQue<float>();
        const DataCopyParams groupStore{
            static_cast<uint16_t>(kPublicGroupRows),
            static_cast<uint16_t>(kPublicGroupWidth * sizeof(float) /
                                  DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((heads - kPublicGroupHeads) * kTile *
                                  sizeof(float) / DEFAULT_C0_SIZE)};
        DataCopy(out[rowBegin * heads * kTile], outGroup, groupStore);
        out_.FreeTensor(outGroup);
    }

private:
    static constexpr uint32_t kTokenRowsPerSlab =
        ChunkSize / TokenSlabCount;
    __aicore__ inline uint32_t LocalStateRow(uint32_t slab) const
    {
        return aivPerAic_ == 1 ? slab * stateSlabRows_ : 0;
    }
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
    TBuf<TPosition::VECCALC> dRows_;
    TBuf<TPosition::VECCALC> broadcastTmp_;
    uint32_t stateDim_ = 0;
    uint32_t stateRowsPerAiv_ = 0;
    uint32_t stateRowStride_ = 0;
    uint32_t stateBlockElements_ = 0;
    uint32_t stateSlabRows_ = 0;
    uint32_t stateSlabElements_ = 0;
    uint32_t headsPerTask_ = 1;
    uint32_t aivPerAic_ = 2;
    bool stateNpLayout_ = false;
};

#ifndef MAMBA2_STATE_EPILOGUE_VECTOR_ONLY
using VectorStateEpilogue = VectorStateEpilogueT<kTile>;

class LegacyStateProjectionMatmul {
public:
    __aicore__ inline void Init(
        const TCubeTiling *, TPipe *pipe)
    {
        if ASCEND_IS_AIC {
            object_.Init(*pipe);
        }
    }

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &a, const GlobalTensor<half> &b,
        const GlobalTensor<float> &c, uint32_t kTotal, uint32_t lda,
        uint32_t ldb, uint32_t ldc, uint32_t nTotal)
    {
        object_.ComputeBlock(a, b, c, kTotal, lda, ldb, ldc, nTotal);
    }

private:
    // The 910B3 occupancy path may aggregate four N=64 heads into one N=256
    // projection.  The default helper reserves only N=128 L1/L0 storage;
    // passing 256 to that instance corrupts the following Cube buffers even
    // though the task mapping itself is valid.
    Mamba2DynamicMatmul<half, float, 16, 256> object_;
};

class LegacyStateProjectionHalfMatmul {
public:
    __aicore__ inline void Init(const TCubeTiling *, TPipe *pipe)
    {
        if ASCEND_IS_AIC {
            object_.Init(*pipe);
        }
    }

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &a, const GlobalTensor<half> &b,
        const GlobalTensor<half> &c, uint32_t kTotal, uint32_t lda,
        uint32_t ldb, uint32_t ldc, uint32_t nTotal)
    {
        object_.ComputeBlockToHalf(
            a, b, c, kTotal, lda, ldb, ldc, nTotal);
    }

private:
    Mamba2DynamicMatmul<half, float, 16, 256> object_;
};

constexpr MatmulConfig kArch35StateMatmulConfig =
    GetBasicConfig(kTile, kTile, kTile);
using Arch35StateAType =
    MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Arch35StateBType =
    MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Arch35StateCType =
    MatmulType<TPosition::GM, CubeFormat::ND, float>;
using Arch35StateBiasType =
    MatmulType<TPosition::GM, CubeFormat::ND, float>;

// Ascend 950PR T64/N64 precision-first projection.  Both MIX sides enter the
// high-level Matmul rendezvous after AIV prepares one state slot; AIV then
// consumes the completed projection before advancing to the next chunk.
class Arch35StateProjectionMatmul {
public:
    __aicore__ inline void Init(
        const TCubeTiling *cubeTiling, TPipe *)
    {
        if ASCEND_IS_AIC {
            object_.Init(cubeTiling);
        }
    }

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &a, const GlobalTensor<half> &b,
        const GlobalTensor<float> &c, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t)
    {
        object_.SetOrgShape(kTile, kTile, kTile, kTile, kTile);
        object_.SetSingleShape(kTile, kTile, kTile);
        object_.SetTensorA(a, false);
        object_.SetTensorB(b, false);
        object_.IterateAll(c, false);
        object_.End();
    }

private:
    matmul::MatmulImpl<Arch35StateAType, Arch35StateBType,
                       Arch35StateCType, Arch35StateBiasType,
                       kArch35StateMatmulConfig> object_;
};

constexpr uint32_t kGroupedHeads = 4;
constexpr uint32_t kGroupedProjectionCols = kGroupedHeads * kTile;
constexpr MatmulConfig kArch35GroupedStateMatmulConfig =
    GetBasicConfig(kTile, kGroupedProjectionCols, kTile);
using Arch35GroupedStateCType =
    MatmulType<TPosition::GM, CubeFormat::ND, half>;

// Ascend 950PR grouped projection.  FP32 accumulation remains inside Cube;
// Fixpipe writes FP16 to a per-core ping-pong slot consumed by the paired AIV.
// Keeping four heads in one N=256 operation raises MAC density and halves the
// Cube-to-Vector exchange compared with an FP32 projection workspace.
class Arch35GroupedStateProjectionMatmul {
public:
    __aicore__ inline void Init(
        const TCubeTiling *cubeTiling, TPipe *)
    {
        if ASCEND_IS_AIC {
            object_.Init(cubeTiling);
        }
    }

    __aicore__ inline void ComputeBlock(
        const GlobalTensor<half> &a, const GlobalTensor<half> &b,
        const GlobalTensor<half> &c, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t)
    {
        object_.SetOrgShape(
            kTile, kGroupedProjectionCols, kTile, kTile,
            kGroupedProjectionCols);
        object_.SetSingleShape(kTile, kGroupedProjectionCols, kTile);
        object_.SetTensorA(a, false);
        object_.SetTensorB(b, false);
        object_.IterateAll(c, false);
        object_.End();
    }

private:
    matmul::MatmulImpl<Arch35StateAType, Arch35StateBType,
                       Arch35GroupedStateCType, Arch35StateBiasType,
                       kArch35GroupedStateMatmulConfig> object_;
};

template <uint32_t ChunkSize, bool SavePreGate = false,
          bool SaveStatesStart = false,
          typename ProjectionMatmul = LegacyStateProjectionMatmul,
          typename ProjectionT = float>
class KernelMamba2SsdStateEpilogueT {
public:
    ProjectionMatmul matmul_;

    __aicore__ inline void Init(
        GM_ADDR chunkStates, GM_ADDR dACumsum, GM_ADDR cCube,
        GM_ADDR yDiag, GM_ADDR x, GM_ADDR d, GM_ADDR z,
        GM_ADDR initialStates, GM_ADDR out, GM_ADDR preGate,
        GM_ADDR finalState, GM_ADDR statesStart,
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
        preGateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(preGate), rawElements);
        finalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(finalState),
                                 static_cast<uint64_t>(tiling_.batch) *
                                     tiling_.heads * kTile * tiling_.stateDim);
        if constexpr (SaveStatesStart) {
            statesStartGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ half *>(statesStart),
                headChunks * kTile * tiling_.stateDim);
        }

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
        workspaceProjection_.SetGlobalBuffer(
            reinterpret_cast<__gm__ ProjectionT *>(
                coreWorkspace + twoStateBytes),
            2 * projectionElements_);
        matmul_.Init(&tiling_.cubeTilingData, pipe);
        if ASCEND_IS_AIV {
            vector_.Init(pipe, tiling_.stateDim, tiling_.headsPerTask,
                         tiling_.aivPerAic);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t task = coreIdx_; task < tiling_.taskCount;
             task += tiling_.usedCoreNum) {
            uint32_t batch;
            uint32_t group;
            uint32_t firstHead;
            if (tiling_.inputGrouped != 0) {
                batch = task / tiling_.groups;
                group = task % tiling_.groups;
                firstHead = group * tiling_.headsPerGroup;
            } else if (tiling_.headsPerTask > 1) {
                const uint32_t tasksPerGroup =
                    tiling_.headsPerGroup / tiling_.headsPerTask;
                const uint32_t tasksPerBatch =
                    tiling_.groups * tasksPerGroup;
                batch = task / tasksPerBatch;
                const uint32_t batchTask = task % tasksPerBatch;
                group = batchTask / tasksPerGroup;
                firstHead = group * tiling_.headsPerGroup +
                            (batchTask % tasksPerGroup) *
                                tiling_.headsPerTask;
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
                    RunEpilogue(
                        batch, firstHead, chunk,
                        workspaceProjection_[slot * projectionElements_]);
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
                            workspaceProjection_[
                                slot * projectionElements_ +
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
            if constexpr (SaveStatesStart) {
                // The recurrent state is still the state at the beginning of
                // this chunk.  Persist it here while it remains resident in
                // UB, instead of rebuilding all chunk states in backward.
                if (tiling_.inputGrouped != 0) {
                    const uint32_t group =
                        firstHead / tiling_.headsPerGroup;
                    const uint64_t groupTask =
                        (static_cast<uint64_t>(batch) * tiling_.chunks +
                         chunk) * tiling_.groups + group;
                    const uint32_t groupedRowStride =
                        tiling_.headsPerGroup * kTile;
                    vector_.StoreHalfGrouped(
                        statesStartGm_[
                            groupTask * tiling_.stateDim *
                                groupedRowStride +
                            headSlot * kTile],
                        headSlot, groupedRowStride);
                } else {
                    vector_.StoreHalfInternal(
                        statesStartGm_[
                            headChunk * kTile * tiling_.stateDim],
                        headSlot);
                }
            }
            if (tiling_.inputGrouped != 0) {
                const uint32_t group =
                    firstHead / tiling_.headsPerGroup;
                const uint64_t groupTask =
                    (static_cast<uint64_t>(batch) * tiling_.chunks +
                     chunk) * tiling_.groups + group;
                const uint32_t groupedRowStride =
                    tiling_.headsPerGroup * kTile;
                vector_.UpdateStateGrouped(
                    chunkStateGm_[
                        groupTask * tiling_.stateDim * groupedRowStride +
                        headSlot * kTile],
                    dAGm_[headChunk * ChunkSize], headSlot,
                    groupedRowStride);
            } else {
                vector_.UpdateState(
                    chunkStateGm_[headChunk * kTile * tiling_.stateDim],
                    dAGm_[headChunk * ChunkSize], headSlot);
            }
        }
        SetStateReady(slot);
    }

    __aicore__ inline void RunEpilogue(
        uint32_t batch, uint32_t firstHead, uint32_t chunk,
        const GlobalTensor<ProjectionT> &projection)
    {
        if (tiling_.headsPerTask == kPublicGroupHeads) {
            const uint64_t firstHeadChunk =
                (static_cast<uint64_t>(batch) * tiling_.heads + firstHead) *
                    tiling_.chunks + chunk;
            const uint64_t rawOffset =
                (static_cast<uint64_t>(batch) * tiling_.chunks * ChunkSize +
                 static_cast<uint64_t>(chunk) * ChunkSize) *
                    tiling_.heads * kTile +
                static_cast<uint64_t>(firstHead) * kTile;
            const uint32_t rowsPerSlab =
                ChunkSize / kLogicalSlabCount;
            for (uint32_t slab = GetSubBlockIdx();
                 slab < kLogicalSlabCount; slab += tiling_.aivPerAic) {
                const uint32_t slabBegin = slab * rowsPerSlab;
                for (uint32_t row = 0; row < rowsPerSlab;
                     row += kPublicGroupRows) {
                    vector_.template EpiloguePublicGroup4<
                        SavePreGate, ProjectionT>(
                        projection,
                        yDiagGm_[firstHeadChunk * ChunkSize * kTile],
                        dAGm_[firstHeadChunk * ChunkSize], xGm_[rawOffset],
                        zGm_[rawOffset], outGm_[rawOffset],
                        preGateGm_[rawOffset], tiling_.heads,
                        projectionCols_, tiling_.chunks * ChunkSize * kTile,
                        tiling_.chunks * ChunkSize, slabBegin + row);
                }
            }
            return;
        }
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
            for (uint32_t slab = GetSubBlockIdx();
                slab < kLogicalSlabCount; slab += tiling_.aivPerAic) {
                vector_.template Epilogue<SavePreGate, ProjectionT>(
                    projection[headSlot * kTile],
                    yDiagGm_[headChunk * ChunkSize * kTile],
                    dAGm_[headChunk * ChunkSize], xGm_[rawOffset],
                    zGm_[rawOffset], outGm_[rawOffset],
                    preGateGm_[rawOffset], tiling_.heads, headSlot,
                    projectionCols_, slab * (ChunkSize / kLogicalSlabCount));
            }
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
    GlobalTensor<half> preGateGm_;
    GlobalTensor<float> finalGm_;
    GlobalTensor<half> statesStartGm_;
    GlobalTensor<half> workspaceHalf_;
    GlobalTensor<ProjectionT> workspaceProjection_;
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
    KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(14, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(4)) {
        KernelMamba2SsdStateEpilogueT<64> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, out, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(5)) {
        KernelMamba2SsdStateEpilogueT<
            64, false, false, LegacyStateProjectionHalfMatmul, half> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, out, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(14)) {
        KernelMamba2SsdStateEpilogueT<64, false, false,
                                      Arch35StateProjectionMatmul> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, out, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(15)) {
        KernelMamba2SsdStateEpilogueT<
            64, false, false, Arch35GroupedStateProjectionMatmul, half> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, out, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(8)) {
        KernelMamba2SsdStateEpilogueT<128> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, out, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
}
#endif
