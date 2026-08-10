// Copyright (c) 2026, mamba-ascendc authors.
// Second launch of the 950PR State path.  This is a pure Vector consumer of
// the dedicated y_off tensor produced by Mamba2SsdStateProjection.

#define MAMBA2_STATE_EPILOGUE_COMPONENT_ONLY
#define MAMBA2_STATE_EPILOGUE_VECTOR_ONLY
#include "mamba2_ssd_state_epilogue.cpp"

using namespace AscendC;

namespace {
class KernelMamba2SsdStateVectorEpilogue {
public:
    __aicore__ inline void Init(
        GM_ADDR yOff, GM_ADDR dACumsum, GM_ADDR yDiag,
        GM_ADDR x, GM_ADDR d, GM_ADDR z, GM_ADDR out,
        const Mamba2SsdStateVectorEpilogueTilingData &tiling,
        TPipe *pipe)
    {
        tiling_ = tiling;
        const uint64_t headChunks =
            static_cast<uint64_t>(tiling_.taskCount);
        const uint64_t rawElements =
            static_cast<uint64_t>(tiling_.batch) * tiling_.chunks *
            kTile * tiling_.heads * kTile;
        yOffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(yOff),
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
        for (uint32_t task = coreIdx_; task < tiling_.taskCount;
             task += tiling_.usedCoreNum) {
            const uint32_t chunk = task % tiling_.chunks;
            const uint32_t stream = task / tiling_.chunks;
            const uint32_t head = stream % tiling_.heads;
            const uint32_t batch = stream / tiling_.heads;
            const uint64_t rawOffset =
                (static_cast<uint64_t>(batch) * tiling_.chunks * kTile +
                 static_cast<uint64_t>(chunk) * kTile) *
                    tiling_.heads * kTile +
                static_cast<uint64_t>(head) * kTile;
            vector_.PrepareD(
                dGm_[static_cast<uint64_t>(head) * kTile], 0);
            for (uint32_t slab = 0; slab < kLogicalSlabCount; ++slab) {
                vector_.template Epilogue<false>(
                    yOffGm_[static_cast<uint64_t>(task) * kTileElements],
                    yDiagGm_[static_cast<uint64_t>(task) * kTileElements],
                    dAGm_[static_cast<uint64_t>(task) * kTile],
                    xGm_[rawOffset], zGm_[rawOffset], outGm_[rawOffset],
                    outGm_[rawOffset], tiling_.heads, 0, kTile,
                    slab * kRowsPerSlab);
            }
        }
    }

private:
    VectorStateEpilogueT<kTile> vector_;
    GlobalTensor<float> yOffGm_;
    GlobalTensor<float> dAGm_;
    GlobalTensor<float> yDiagGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> dGm_;
    GlobalTensor<float> zGm_;
    GlobalTensor<float> outGm_;
    Mamba2SsdStateVectorEpilogueTilingData tiling_;
    uint32_t coreIdx_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_state_vector_epilogue(
    GM_ADDR y_off, GM_ADDR d_a_cumsum, GM_ADDR y_diag,
    GM_ADDR x, GM_ADDR d, GM_ADDR z, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelMamba2SsdStateVectorEpilogue op;
        op.Init(y_off, d_a_cumsum, y_diag, x, d, z, out,
                tilingData, &pipe);
        op.Process();
    }
}
