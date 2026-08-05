#ifndef MAMBA2_SSD_CHUNK_MIX128_H
#define MAMBA2_SSD_CHUNK_MIX128_H

namespace Mamba2Chunk128 {

using namespace AscendC;

constexpr uint32_t kT = 128;
constexpr uint32_t kP = 64;
constexpr uint32_t kRowsPerAiv = 64;
constexpr uint32_t kTransposeTile = 16;
constexpr uint32_t kTransposeElements = 256;
constexpr uint16_t kVectorCbInputReady = 0x8;
constexpr uint16_t kCubeCbReady = 0x9;
constexpr uint16_t kHeadReady0 = 0xA;
constexpr uint16_t kHeadReady1 = 0xB;
constexpr uint16_t kHeadDone0 = 0xC;
constexpr uint16_t kHeadDone1 = 0xD;

class VectorChunkKernel128 {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        constexpr uint32_t blockElements = kRowsPerAiv * kT;
        // PrepareB still uses 16x16 transpose tiles.  The same queues also
        // stream a 16xP weighted-X row block without transposing it.
        pipe->InitBuffer(transposeIn_, 1,
                         kTransposeTile * kP * sizeof(half));
        pipe->InitBuffer(transposeOut_, 1,
                         kTransposeTile * kP * sizeof(half));
        pipe->InitBuffer(dAIn_, 1, kT * sizeof(float));
        pipe->InitBuffer(cbBlockIn_, 1, blockElements * sizeof(float));
        pipe->InitBuffer(halfBlockOut_, 1, blockElements * sizeof(half));
        pipe->InitBuffer(floatBlock0_, blockElements * sizeof(float));
        pipe->InitBuffer(floatBlock1_, blockElements * sizeof(float));
        pipe->InitBuffer(causalMask_, blockElements * sizeof(float));
        pipe->InitBuffer(causalMaskHalf_, blockElements * sizeof(half));
        pipe->InitBuffer(decayHalf_, kT * sizeof(half));
        pipe->InitBuffer(decayTileHalf_,
                         kTransposeTile * kP * sizeof(half));
        pipe->InitBuffer(broadcastTmp_, 2 * blockElements * sizeof(uint8_t));

        auto mask = causalMask_.Get<float>();
        auto maskHalf = causalMaskHalf_.Get<half>();
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        for (uint32_t localRow = 0; localRow < kRowsPerAiv; ++localRow) {
            auto rowMask = mask[localRow * kT];
            auto rowMaskHalf = maskHalf[localRow * kT];
            Duplicate(rowMask, 0.0f, kT);
            Duplicate(rowMaskHalf, static_cast<half>(0.0f), kT);
            PipeBarrier<PIPE_V>();
            Duplicate(rowMask, 1.0f, rowBegin + localRow + 1);
            Duplicate(rowMaskHalf, static_cast<half>(1.0f),
                      rowBegin + localRow + 1);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareB(const GlobalTensor<half> &b,
                                    GlobalTensor<half> bT,
                                    uint32_t stateDim)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        const uint32_t rowEnd = rowBegin + kRowsPerAiv;
        for (uint32_t row = rowBegin; row < rowEnd; row += kTransposeTile) {
            for (uint32_t col = 0; col < stateDim; col += kTransposeTile) {
                TransposeHalfTile(b[row * stateDim + col],
                                  bT[col * kT + row], stateDim, kT);
            }
        }
    }

    __aicore__ inline void PrepareHead(const GlobalTensor<half> &x,
                                       const GlobalTensor<float> &dA,
                                       GlobalTensor<half> weightedXT)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        const uint32_t rowEnd = rowBegin + kRowsPerAiv;
        dALocal_ = dAIn_.AllocTensor<float>();
        DataCopy(dALocal_, dA, kT);
        dAIn_.EnQue(dALocal_);
        dALocal_ = dAIn_.DeQue<float>();

        auto decay = floatBlock0_.Get<float>();
        const float lastDA = dALocal_.GetValue(kT - 1);
        Muls(decay, dALocal_, -1.0f, kT);
        PipeBarrier<PIPE_V>();
        Adds(decay, decay, lastDA, kT);
        PipeBarrier<PIPE_V>();
        Exp(decay, decay, kT);
        PipeBarrier<PIPE_V>();
        auto decayHalf = decayHalf_.Get<half>();
        Cast(decayHalf, decay, RoundMode::CAST_RINT, kT);
        PipeBarrier<PIPE_V>();

        for (uint32_t row = rowBegin; row < rowEnd; row += kTransposeTile) {
            WeightedHalfBlock(
                x[row * kP], weightedXT[row * kP], decayHalf[row]);
        }
    }

    __aicore__ inline void BuildWeights(const GlobalTensor<float> &cb,
                                        GlobalTensor<half> w)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        constexpr uint32_t blockElements = kRowsPerAiv * kT;
        auto colDA = floatBlock0_.Get<float>();
        auto rowDA = floatBlock1_.Get<float>();
        auto mask = causalMask_.Get<float>();
        auto tmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t colShape[2] = {1U, kT};
        const uint32_t rowShape[2] = {kRowsPerAiv, 1U};
        const uint32_t dstShape[2] = {kRowsPerAiv, kT};
        Broadcast<float, 2, 0>(colDA, dALocal_, dstShape, colShape, tmp);
        PipeBarrier<PIPE_V>();
        Broadcast<float, 2, 1>(rowDA, dALocal_[rowBegin], dstShape,
                               rowShape, tmp);
        PipeBarrier<PIPE_V>();
        Sub(colDA, rowDA, colDA, blockElements);
        PipeBarrier<PIPE_V>();
        Mul(colDA, colDA, mask, blockElements);
        PipeBarrier<PIPE_V>();
        // W is consumed by Cube as FP16.  Keep the potentially cancellation-
        // sensitive dA difference and causal zeroing in FP32, then perform the
        // expensive elementwise epilogue in the representation that is
        // actually stored.  floatBlock1_ is dead after the row broadcast, so
        // reuse its storage for the FP16 decay matrix.
        auto decayHalfMatrix = floatBlock1_.Get<half>();
        Cast(decayHalfMatrix, colDA, RoundMode::CAST_RINT, blockElements);
        PipeBarrier<PIPE_V>();
        Exp(decayHalfMatrix, decayHalfMatrix, blockElements);
        PipeBarrier<PIPE_V>();

        auto cbLocal = cbBlockIn_.AllocTensor<float>();
        DataCopy(cbLocal, cb[rowBegin * kT], blockElements);
        cbBlockIn_.EnQue(cbLocal);
        cbLocal = cbBlockIn_.DeQue<float>();
        auto halfLocal = halfBlockOut_.AllocTensor<half>();
        Cast(halfLocal, cbLocal, RoundMode::CAST_RINT, blockElements);
        cbBlockIn_.FreeTensor(cbLocal);
        PipeBarrier<PIPE_V>();
        Mul(halfLocal, halfLocal, decayHalfMatrix, blockElements);
        PipeBarrier<PIPE_V>();
        Mul(halfLocal, halfLocal, causalMaskHalf_.Get<half>(), blockElements);
        PipeBarrier<PIPE_V>();
        halfBlockOut_.EnQue(halfLocal);
        halfLocal = halfBlockOut_.DeQue<half>();
        DataCopy(w[rowBegin * kT], halfLocal, blockElements);
        halfBlockOut_.FreeTensor(halfLocal);
        dAIn_.FreeTensor(dALocal_);
    }

private:
    __aicore__ inline void WeightedHalfBlock(
        const GlobalTensor<half> &src, GlobalTensor<half> dst,
        const LocalTensor<half> &rowDecay)
    {
        constexpr uint32_t blockElements = kTransposeTile * kP;
        auto input = transposeIn_.AllocTensor<half>();
        DataCopy(input, src, blockElements);
        transposeIn_.EnQue(input);
        input = transposeIn_.DeQue<half>();
        auto decayTile = decayTileHalf_.Get<half>();
        auto tmp = broadcastTmp_.Get<uint8_t>();
        const uint32_t decayShape[2] = {kTransposeTile, 1U};
        const uint32_t tileShape[2] = {kTransposeTile, kP};
        Broadcast<half, 2, 1>(decayTile, rowDecay, tileShape, decayShape, tmp);
        PipeBarrier<PIPE_V>();
        auto output = transposeOut_.AllocTensor<half>();
        Mul(output, input, decayTile, blockElements);
        transposeIn_.FreeTensor(input);
        transposeOut_.EnQue(output);
        output = transposeOut_.DeQue<half>();
        DataCopy(dst, output, blockElements);
        transposeOut_.FreeTensor(output);
    }

    __aicore__ inline void TransposeHalfTile(
        const GlobalTensor<half> &src, GlobalTensor<half> dst,
        uint32_t srcStride, uint32_t dstStride)
    {
        auto input = transposeIn_.AllocTensor<half>();
        DataCopyParams load{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE),
            static_cast<uint16_t>((srcStride - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE), 0};
        DataCopy(input, src, load);
        transposeIn_.EnQue(input);
        input = transposeIn_.DeQue<half>();
        auto output = transposeOut_.AllocTensor<half>();
        Transpose(output, input);
        transposeIn_.FreeTensor(input);
        transposeOut_.EnQue(output);
        output = transposeOut_.DeQue<half>();
        DataCopyParams store{
            static_cast<uint16_t>(kTransposeTile),
            static_cast<uint16_t>(kTransposeTile * sizeof(half) /
                                  DEFAULT_C0_SIZE), 0,
            static_cast<uint16_t>((dstStride - kTransposeTile) * sizeof(half) /
                                  DEFAULT_C0_SIZE)};
        DataCopy(dst, output, store);
        transposeOut_.FreeTensor(output);
    }

    TQue<TPosition::VECIN, 1> transposeIn_;
    TQue<TPosition::VECOUT, 1> transposeOut_;
    TQue<TPosition::VECIN, 1> dAIn_;
    TQue<TPosition::VECIN, 1> cbBlockIn_;
    TQue<TPosition::VECOUT, 1> halfBlockOut_;
    TBuf<TPosition::VECCALC> floatBlock0_;
    TBuf<TPosition::VECCALC> floatBlock1_;
    TBuf<TPosition::VECCALC> causalMask_;
    TBuf<TPosition::VECCALC> causalMaskHalf_;
    TBuf<TPosition::VECCALC> decayHalf_;
    TBuf<TPosition::VECCALC> decayTileHalf_;
    TBuf<TPosition::VECCALC> broadcastTmp_;
    LocalTensor<float> dALocal_;
};

class KernelMamba2SsdChunkMix128 {
public:
    __aicore__ inline void Init(
        GM_ADDR xCube, GM_ADDR dACumsum, GM_ADDR bCube, GM_ADDR cCube,
        GM_ADDR yDiag, GM_ADDR chunkStates, GM_ADDR workspace,
        const Mamba2SsdChunkMixTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        const uint64_t headTasks = static_cast<uint64_t>(tiling_.headTaskCount);
        const uint64_t groupTasks = static_cast<uint64_t>(tiling_.batch) *
                                    tiling_.chunks * tiling_.groups;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(xCube),
                             headTasks * kT * kP);
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum),
                              headTasks * kT);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bCube),
                             groupTasks * kT * tiling_.stateDim);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube),
                             groupTasks * kT * tiling_.stateDim);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yDiag),
                             headTasks * kT * kP);
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(chunkStates),
                                 headTasks * kP * tiling_.stateDim);

        weightedHalfOffset_[0] = 0;
        weightedHalfOffset_[1] = weightedHalfOffset_[0] + kP * kT;
        wHalfOffset_[0] = weightedHalfOffset_[1] + kP * kT;
        wHalfOffset_[1] = wHalfOffset_[0] + kT * kT;
        cbByteOffset_ = (wHalfOffset_[1] + kT * kT) * sizeof(half);

        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) * tiling_.workspaceBytesPerCore;
        workspaceHalf_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(coreWorkspace),
            cbByteOffset_ / sizeof(half));
        cbWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(coreWorkspace + cbByteOffset_),
            kT * kT);
        if ASCEND_IS_AIC matmul_.Init(*pipe);
        if ASCEND_IS_AIV vector_.Init(pipe);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t task = coreIdx_; task < tiling_.taskCount;
             task += tiling_.usedCoreNum) {
            uint32_t batch, chunk, group, headBegin, headCount;
            DecodeTask(task, batch, chunk, group, headBegin, headCount);
            const uint32_t bcIndex =
                (batch * tiling_.chunks + chunk) * tiling_.groups + group;
            const uint64_t bcOffset = static_cast<uint64_t>(bcIndex) *
                                      kT * tiling_.stateDim;
            auto bT = bGm_[bcOffset];
            auto cMat = cGm_[bcOffset];

            if ASCEND_IS_AIV {
                CrossCoreWaitFlag<0x2>(kCubeCbReady);
                for (uint32_t headOffset = 0; headOffset < headCount;
                     ++headOffset) {
                    const uint32_t slot = headOffset & 1U;
                    if (headOffset >= 2) WaitHeadDone(slot);
                    const uint32_t head = headBegin + headOffset;
                    const uint32_t headTask =
                        (batch * tiling_.heads + head) * tiling_.chunks + chunk;
                    auto weighted = workspaceHalf_[weightedHalfOffset_[slot]];
                    auto w = workspaceHalf_[wHalfOffset_[slot]];
                    vector_.PrepareHead(
                        xGm_[static_cast<uint64_t>(headTask) * kT * kP],
                        dAGm_[static_cast<uint64_t>(headTask) * kT], weighted);
                    vector_.BuildWeights(cbWorkspace_, w);
                    SetHeadReady(slot);
                }
                if (headCount == 1) {
                    WaitHeadDone(0);
                } else if (headCount >= 2) {
                    WaitHeadDone((headCount - 2) & 1U);
                    WaitHeadDone((headCount - 1) & 1U);
                }
            }

            if ASCEND_IS_AIC {
                for (uint32_t row = 0; row < kT; row += kP) {
                    matmul_.ComputeBlock(
                        cMat[row * tiling_.stateDim], bT,
                        cbWorkspace_[row * kT], tiling_.stateDim,
                        tiling_.stateDim, kT, kT, kT);
                }
                CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeCbReady);
                for (uint32_t headOffset = 0; headOffset < headCount;
                     ++headOffset) {
                    const uint32_t slot = headOffset & 1U;
                    const uint32_t head = headBegin + headOffset;
                    const uint32_t headTask =
                        (batch * tiling_.heads + head) * tiling_.chunks + chunk;
                    auto x = xGm_[static_cast<uint64_t>(headTask) * kT * kP];
                    auto y = yGm_[static_cast<uint64_t>(headTask) * kT * kP];
                    auto state = stateGm_[static_cast<uint64_t>(headTask) *
                                          kP * tiling_.stateDim];
                    auto weighted = workspaceHalf_[weightedHalfOffset_[slot]];
                    auto w = workspaceHalf_[wHalfOffset_[slot]];
                    WaitHeadReady(slot);
                    for (uint32_t row = 0; row < kT; row += kP) {
                        matmul_.ComputeBlock(
                            w[row * kT], x, y[row * kP], kT, kT, kP,
                            kP, kP);
                    }
                    for (uint32_t row = 0; row < tiling_.stateDim;
                         row += kP) {
                        matmul_.ComputeBlock(
                            bT[row * kT], weighted, state[row * kP],
                            kT, kT, kP, kP, kP);
                    }
                    SetHeadDone(slot);
                }
            }
        }
    }

private:
    __aicore__ inline void DecodeTask(
        uint32_t task, uint32_t &batch, uint32_t &chunk, uint32_t &group,
        uint32_t &headBegin, uint32_t &headCount) const
    {
        chunk = task % tiling_.chunks;
        const uint32_t outer = task / tiling_.chunks;
        if (tiling_.taskMode == 1) {
            group = outer % tiling_.groups;
            batch = outer / tiling_.groups;
            headBegin = group * tiling_.headsPerGroup;
            headCount = tiling_.headsPerGroup;
        } else {
            const uint32_t head = outer % tiling_.heads;
            batch = outer / tiling_.heads;
            group = head / tiling_.headsPerGroup;
            headBegin = head;
            headCount = 1;
        }
    }

    __aicore__ inline void SetHeadReady(uint32_t slot)
    {
        if (slot == 0) CrossCoreSetFlag<0x2, PIPE_MTE3>(kHeadReady0);
        else CrossCoreSetFlag<0x2, PIPE_MTE3>(kHeadReady1);
    }
    __aicore__ inline void WaitHeadReady(uint32_t slot)
    {
        if (slot == 0) CrossCoreWaitFlag<0x2>(kHeadReady0);
        else CrossCoreWaitFlag<0x2>(kHeadReady1);
    }
    __aicore__ inline void SetHeadDone(uint32_t slot)
    {
        if (slot == 0) CrossCoreSetFlag<0x2, PIPE_FIX>(kHeadDone0);
        else CrossCoreSetFlag<0x2, PIPE_FIX>(kHeadDone1);
    }
    __aicore__ inline void WaitHeadDone(uint32_t slot)
    {
        if (slot == 0) CrossCoreWaitFlag<0x2>(kHeadDone0);
        else CrossCoreWaitFlag<0x2>(kHeadDone1);
    }

    Mamba2DynamicMatmul<half, float, 64> matmul_;
    VectorChunkKernel128 vector_;
    GlobalTensor<half> xGm_, bGm_, cGm_, workspaceHalf_;
    GlobalTensor<float> dAGm_, yGm_, stateGm_, cbWorkspace_;
    Mamba2SsdChunkMixTilingData tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t weightedHalfOffset_[2] = {0, 0};
    uint32_t wHalfOffset_[2] = {0, 0};
    uint32_t cbByteOffset_ = 0;
};

} // namespace Mamba2Chunk128

#endif // MAMBA2_SSD_CHUNK_MIX128_H
