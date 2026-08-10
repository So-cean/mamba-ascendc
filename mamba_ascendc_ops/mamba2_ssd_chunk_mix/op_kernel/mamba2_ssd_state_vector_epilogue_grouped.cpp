// Copyright (c) 2026, mamba-ascendc authors.

#define MAMBA2_STATE_EPILOGUE_COMPONENT_ONLY
#define MAMBA2_STATE_EPILOGUE_VECTOR_ONLY
#include "mamba2_ssd_state_epilogue.cpp"

using namespace AscendC;

namespace {
class KernelMamba2SsdStateVectorEpilogueGrouped {
public:
    __aicore__ inline void Init(
        GM_ADDR yGrouped, GM_ADDR dACumsum, GM_ADDR yDiag,
        GM_ADDR x, GM_ADDR d, GM_ADDR z, GM_ADDR out,
        const Mamba2SsdStateVectorEpilogueGroupedTilingData &tiling,
        TPipe *pipe)
    {
        tiling_ = tiling;
        groupedDiag_ = tiling_.groupedDiag != 0;
        const uint64_t headChunks =
            static_cast<uint64_t>(tiling_.batch) * tiling_.heads *
            tiling_.chunks;
        const uint64_t rawElements =
            static_cast<uint64_t>(tiling_.batch) * tiling_.chunks *
            kTile * tiling_.heads * kTile;
        yGroupedGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(yGrouped),
            headChunks * kTileElements);
        dAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dACumsum),
                              headChunks * kTile);
        yDiagGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yDiag),
                                 headChunks * kTileElements);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), rawElements);
        dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d),
                             static_cast<uint64_t>(tiling_.heads) * kTile);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(z), rawElements);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out),
                               rawElements);
        coreIdx_ = GetBlockIdx();
        vector_.Init(pipe, kTile, 1, 1);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t stream = coreIdx_; stream < tiling_.taskCount;
             stream += tiling_.usedCoreNum) {
            const uint32_t head = stream % tiling_.heads;
            const uint32_t batch = stream / tiling_.heads;
            const uint32_t group = head / tiling_.headsPerGroup;
            const uint32_t headInGroup = head % tiling_.headsPerGroup;
            vector_.PrepareD(
                dGm_[static_cast<uint64_t>(head) * kTile], 0);
            for (uint32_t chunk = 0; chunk < tiling_.chunks; ++chunk) {
                const uint64_t headChunk =
                    static_cast<uint64_t>(stream) * tiling_.chunks + chunk;
                const uint64_t groupTask =
                    (static_cast<uint64_t>(batch) * tiling_.chunks + chunk) *
                        tiling_.groups + group;
                const uint64_t yBase =
                    groupTask * kTile * tiling_.headsPerGroup * kTile +
                    static_cast<uint64_t>(headInGroup) * kTile;
                const uint64_t yDiagBase = groupedDiag_
                    ? yBase
                    : headChunk * kTileElements;
                const uint64_t rawOffset =
                    (static_cast<uint64_t>(batch) * tiling_.chunks * kTile +
                     static_cast<uint64_t>(chunk) * kTile) *
                        tiling_.heads * kTile +
                    static_cast<uint64_t>(head) * kTile;
                vector_.template Epilogue<false, half>(
                    yGroupedGm_[yBase],
                    yDiagGm_[yDiagBase],
                    dAGm_[headChunk * kTile],
                    xGm_[rawOffset], zGm_[rawOffset], outGm_[rawOffset],
                    outGm_[rawOffset], tiling_.heads, 0,
                    tiling_.headsPerGroup * kTile, 0,
                    groupedDiag_ ? tiling_.headsPerGroup * kTile : kTile);
            }
        }
    }

private:
    // 950PR has one AIV per MIX core.  Process the full 64-token chunk in
    // one 16-KiB DMA/Vector slab instead of issuing two 8-KiB half-slabs.
    VectorStateEpilogueT<kTile, 1> vector_;
    GlobalTensor<half> yGroupedGm_;
    GlobalTensor<float> dAGm_;
    GlobalTensor<float> yDiagGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> dGm_;
    GlobalTensor<float> zGm_;
    GlobalTensor<float> outGm_;
    Mamba2SsdStateVectorEpilogueGroupedTilingData tiling_;
    uint32_t coreIdx_ = 0;
    bool groupedDiag_ = false;
};

}  // namespace

extern "C" __global__ __aicore__ void
mamba2_ssd_state_vector_epilogue_grouped(
    GM_ADDR y_grouped, GM_ADDR d_a_cumsum, GM_ADDR y_diag,
    GM_ADDR x, GM_ADDR d, GM_ADDR z, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelMamba2SsdStateVectorEpilogueGrouped op;
        op.Init(y_grouped, d_a_cumsum, y_diag, x, d, z, out,
                tilingData, &pipe);
        op.Process();
    }
}
