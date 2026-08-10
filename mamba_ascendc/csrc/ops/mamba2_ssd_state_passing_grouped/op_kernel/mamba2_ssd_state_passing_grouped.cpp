// Copyright (c) 2026, mamba-ascendc authors.
//
// Internal group-contiguous StatePassing path.  ChunkMix already produces
// Cube-native [N,P] states.  Keep that layout through the recurrence and
// write FP16 [B,K,G,N,R,P] directly for the grouped 64x256 projection.

#include "kernel_operator.h"

namespace {
constexpr int64_t kTile = 64;
constexpr int64_t kTileElements = kTile * kTile;

class KernelMamba2SsdStatePassingGrouped {
public:
    __aicore__ inline void Init(
        GM_ADDR chunkStatesNp, GM_ADDR dACumsum, GM_ADDR initialStatesNp,
        GM_ADDR statesGrouped, GM_ADDR finalStateNp,
        int64_t batch, int64_t nheads, int64_t nchunks, int64_t groups,
        int64_t chunkSize, int64_t hasInitial, int64_t usedCoreNum,
        int64_t inputGrouped)
    {
        chunkStatesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(chunkStatesNp));
        dACumsumGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dACumsum));
        initialStatesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(initialStatesNp));
        statesGroupedGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(statesGrouped));
        finalStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(finalStateNp));
        batch_ = batch;
        nheads_ = nheads;
        nchunks_ = nchunks;
        groups_ = groups;
        headsPerGroup_ = nheads / groups;
        chunkSize_ = chunkSize;
        hasInitial_ = hasInitial;
        usedCoreNum_ = usedCoreNum;
        inputGrouped_ = inputGrouped;
        pipe_.InitBuffer(stateBuf_, kTileElements * sizeof(float));
        pipe_.InitBuffer(inQueue_, 1, kTileElements * sizeof(float));
        pipe_.InitBuffer(halfOutQueue_, 1, kTileElements * sizeof(half));
        pipe_.InitBuffer(finalOutQueue_, 1, kTileElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * nheads_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount; task += usedCoreNum_) {
            auto state = stateBuf_.Get<float>();
            if (hasInitial_ != 0) {
                auto initial = inQueue_.AllocTensor<float>();
                AscendC::DataCopy(
                    initial, initialStatesGm_[task * kTileElements],
                    kTileElements);
                inQueue_.EnQue(initial);
                initial = inQueue_.DeQue<float>();
                AscendC::Adds(state, initial, 0.0f, kTileElements);
                inQueue_.FreeTensor(initial);
            } else {
                AscendC::Duplicate(state, 0.0f, kTileElements);
            }

            const int64_t batch = task / nheads_;
            const int64_t head = task % nheads_;
            const int64_t group = head / headsPerGroup_;
            const int64_t headInGroup = head % headsPerGroup_;
            for (int64_t chunk = 0; chunk < nchunks_; ++chunk) {
                StoreGroupedState(
                    batch, chunk, group, headInGroup, state);

                const int64_t decayOffset =
                    (task * nchunks_ + chunk) * chunkSize_ + chunkSize_ - 1;
                const float decay = ScalarExp(
                    dACumsumGm_.GetValue(decayOffset));
                auto contribution = inQueue_.AllocTensor<float>();
                if (inputGrouped_ != 0) {
                    const int64_t groupTask =
                        (batch * nchunks_ + chunk) * groups_ + group;
                    const int64_t groupElements =
                        kTile * headsPerGroup_ * kTile;
                    const int64_t stateOffset =
                        groupTask * groupElements + headInGroup * kTile;
                    AscendC::DataCopyParams copy{
                        static_cast<uint16_t>(kTile),
                        static_cast<uint16_t>(kTile * sizeof(float) /
                                              AscendC::DEFAULT_C0_SIZE),
                        static_cast<uint16_t>((headsPerGroup_ - 1) * kTile *
                                              sizeof(float) /
                                              AscendC::DEFAULT_C0_SIZE),
                        0};
                    AscendC::DataCopy(
                        contribution, chunkStatesGm_[stateOffset], copy);
                } else {
                    const int64_t stateOffset =
                        (task * nchunks_ + chunk) * kTileElements;
                    AscendC::DataCopy(
                        contribution, chunkStatesGm_[stateOffset],
                        kTileElements);
                }
                inQueue_.EnQue(contribution);
                contribution = inQueue_.DeQue<float>();
                AscendC::Muls(state, state, decay, kTileElements);
                AscendC::Add(
                    state, state, contribution, kTileElements);
                inQueue_.FreeTensor(contribution);
            }
            StoreFinalState(task, state);
        }
    }

private:
    __aicore__ inline void StoreGroupedState(
        int64_t batch, int64_t chunk, int64_t group,
        int64_t headInGroup, const AscendC::LocalTensor<float> &state)
    {
        auto stateHalf = halfOutQueue_.AllocTensor<half>();
        AscendC::Cast(stateHalf, state, AscendC::RoundMode::CAST_RINT,
                      kTileElements);
        halfOutQueue_.EnQue(stateHalf);
        stateHalf = halfOutQueue_.DeQue<half>();
        const int64_t groupTask =
            (batch * nchunks_ + chunk) * groups_ + group;
        const int64_t groupElements =
            kTile * headsPerGroup_ * kTile;
        const int64_t destination =
            groupTask * groupElements + headInGroup * kTile;
        AscendC::DataCopyParams copy{
            static_cast<uint16_t>(kTile),
            static_cast<uint16_t>(kTile * sizeof(half) /
                                  AscendC::DEFAULT_C0_SIZE),
            0,
            static_cast<uint16_t>((headsPerGroup_ - 1) * kTile *
                                  sizeof(half) /
                                  AscendC::DEFAULT_C0_SIZE)};
        AscendC::DataCopy(
            statesGroupedGm_[destination], stateHalf, copy);
        halfOutQueue_.FreeTensor(stateHalf);
    }

    __aicore__ inline void StoreFinalState(
        int64_t task, const AscendC::LocalTensor<float> &state)
    {
        auto output = finalOutQueue_.AllocTensor<float>();
        AscendC::Adds(output, state, 0.0f, kTileElements);
        finalOutQueue_.EnQue(output);
        output = finalOutQueue_.DeQue<float>();
        AscendC::DataCopy(
            finalStateGm_[task * kTileElements], output, kTileElements);
        finalOutQueue_.FreeTensor(output);
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

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> halfOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> finalOutQueue_;
    AscendC::GlobalTensor<float> chunkStatesGm_;
    AscendC::GlobalTensor<float> dACumsumGm_;
    AscendC::GlobalTensor<float> initialStatesGm_;
    AscendC::GlobalTensor<half> statesGroupedGm_;
    AscendC::GlobalTensor<float> finalStateGm_;
    int64_t batch_ = 0;
    int64_t nheads_ = 0;
    int64_t nchunks_ = 0;
    int64_t groups_ = 0;
    int64_t headsPerGroup_ = 0;
    int64_t chunkSize_ = 0;
    int64_t hasInitial_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t inputGrouped_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_state_passing_grouped(
    GM_ADDR chunk_states_np, GM_ADDR d_a_cumsum, GM_ADDR initial_states_np,
    GM_ADDR states_grouped, GM_ADDR final_state_np,
    int64_t batch, int64_t nheads, int64_t nchunks, int64_t groups,
    int64_t chunk_size, int64_t has_initial, int64_t used_core_num,
    int64_t input_grouped)
{
    KernelMamba2SsdStatePassingGrouped op;
    op.Init(chunk_states_np, d_a_cumsum, initial_states_np,
            states_grouped, final_state_np, batch, nheads, nchunks, groups,
            chunk_size, has_initial, used_core_num, input_grouped);
    op.Process();
}
