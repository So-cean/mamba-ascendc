// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"

namespace {
constexpr int64_t kTile = 64;
constexpr int64_t kElements = kTile * kTile;

class KernelMamba2SsdBwdOffFinalize {
public:
    __aicore__ inline void Init(
        GM_ADDR dStatesHalf, GM_ADDR dCHeadHalf, GM_ADDR cCube,
        GM_ADDR dStates, GM_ADDR dCGroup, GM_ADDR gDa,
        int64_t batch, int64_t groups, int64_t headsPerGroup,
        int64_t chunks, int64_t usedCoreNum, AscendC::TPipe *pipe)
    {
        // dStates is a host-side alias view of dStatesHalf.  Both arguments
        // remain in the launch ABI, but the kernel intentionally performs no
        // transfer or conversion for that tensor.
        (void)dStatesHalf;
        (void)dStates;
        dCHeadHalfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(dCHeadHalf));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube));
        dCGroupGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dCGroup));
        gDaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gDa));
        batch_ = batch;
        groups_ = groups;
        headsPerGroup_ = headsPerGroup;
        heads_ = groups * headsPerGroup;
        chunks_ = chunks;
        usedCoreNum_ = usedCoreNum;

        pipe->InitBuffer(dCHeadHalfBuf_, kElements * sizeof(half));
        pipe->InitBuffer(cHalfBuf_, kElements * sizeof(half));
        pipe->InitBuffer(dCHeadBuf_, kElements * sizeof(float));
        pipe->InitBuffer(dCAccBuf_, kElements * sizeof(float));
        pipe->InitBuffer(cBuf_, kElements * sizeof(float));
        pipe->InitBuffer(productBuf_, kElements * sizeof(float));
        pipe->InitBuffer(gDaBuf_, kTile * sizeof(float));
        mte2ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE2_V));
        vToMte3_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * groups_ * chunks_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount;
             task += usedCoreNum_) {
            ProcessGroupChunk(task);
        }
    }

private:
    template <typename T>
    __aicore__ inline void CopyIn(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::GlobalTensor<T> &src,
        int64_t offset, int64_t elements)
    {
        AscendC::DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(dst, src[offset], copy, pad);
    }

    template <typename T>
    __aicore__ inline void CopyOut(
        const AscendC::GlobalTensor<T> &dst,
        const AscendC::LocalTensor<T> &src,
        int64_t offset, int64_t elements)
    {
        AscendC::DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(dst[offset], src, copy);
    }

    __aicore__ inline void CopyOutGroupMatrix(
        const AscendC::GlobalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src,
        int64_t batch, int64_t chunk, int64_t group)
    {
        const int64_t offset =
            ((batch * chunks_ + chunk) * kTile * groups_ + group) * kTile;
        AscendC::DataCopyExtParams copy{
            static_cast<uint16_t>(kTile),
            static_cast<uint32_t>(kTile * sizeof(float)),
            0,
            static_cast<uint32_t>((groups_ - 1) * kTile * sizeof(float)),
            0};
        AscendC::DataCopyPad(dst[offset], src, copy);
    }

    __aicore__ inline void LoadBarrier()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void StoreBarrier()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
    }

    __aicore__ inline void FinishStoreForVectorReuse()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void ProcessGroupChunk(int64_t task)
    {
        const int64_t group = task % groups_;
        const int64_t chunk = (task / groups_) % chunks_;
        const int64_t batch = task / (groups_ * chunks_);
        const int64_t cOffset =
            ((batch * chunks_ + chunk) * groups_ + group) * kElements;
        auto dCHeadHalf = dCHeadHalfBuf_.Get<half>();
        auto cHalf = cHalfBuf_.Get<half>();
        auto dCHead = dCHeadBuf_.Get<float>();
        auto dCAcc = dCAccBuf_.Get<float>();
        auto c = cBuf_.Get<float>();
        auto product = productBuf_.Get<float>();
        auto gDa = gDaBuf_.Get<float>();

        CopyIn(cHalf, cGm_, cOffset, kElements);
        LoadBarrier();
        AscendC::Cast(c, cHalf, AscendC::RoundMode::CAST_NONE, kElements);
        AscendC::Duplicate(dCAcc, 0.0f, kElements);
        AscendC::PipeBarrier<PIPE_V>();

        for (int64_t headInGroup = 0;
             headInGroup < headsPerGroup_; ++headInGroup) {
            const int64_t head = group * headsPerGroup_ + headInGroup;
            const int64_t inputTask =
                ((batch * groups_ + group) * headsPerGroup_ + headInGroup) *
                    chunks_ + chunk;
            const int64_t outputTask =
                (batch * heads_ + head) * chunks_ + chunk;
            CopyIn(
                dCHeadHalf, dCHeadHalfGm_,
                inputTask * kElements, kElements);
            LoadBarrier();
            AscendC::Cast(
                dCHead, dCHeadHalf,
                AscendC::RoundMode::CAST_NONE, kElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, dCHead, c, kElements);
            AscendC::Add(dCAcc, dCAcc, dCHead, kElements);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float, true>(
                gDa, product, static_cast<int32_t>(kTile),
                static_cast<int32_t>(kTile), 1, 1,
                static_cast<int32_t>(
                    kTile * sizeof(float) / AscendC::DEFAULT_C0_SIZE));
            AscendC::PipeBarrier<PIPE_V>();
            StoreBarrier();
            CopyOut(gDaGm_, gDa, outputTask * kTile, kTile);
            FinishStoreForVectorReuse();
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        StoreBarrier();
        CopyOutGroupMatrix(dCGroupGm_, dCAcc, batch, chunk, group);
        FinishStoreForVectorReuse();
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> dCHeadHalfBuf_, cHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dCHeadBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dCAccBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cBuf_, productBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gDaBuf_;
    AscendC::GlobalTensor<half> dCHeadHalfGm_, cGm_;
    AscendC::GlobalTensor<float> dCGroupGm_, gDaGm_;
    int64_t batch_, groups_, headsPerGroup_, heads_, chunks_, usedCoreNum_;
    event_t mte2ToV_, vToMte3_, mte3ToV_;
};
}  // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_bwd_off_finalize(
    GM_ADDR d_states_half, GM_ADDR d_c_head_half, GM_ADDR c_cube,
    GM_ADDR d_states, GM_ADDR d_c_group, GM_ADDR g_d_a,
    int64_t batch, int64_t groups, int64_t heads_per_group,
    int64_t chunks, int64_t used_core_num)
{
    KernelMamba2SsdBwdOffFinalize op;
    AscendC::TPipe pipe;
    op.Init(d_states_half, d_c_head_half, c_cube,
            d_states, d_c_group, g_d_a,
            batch, groups, heads_per_group, chunks, used_core_num, &pipe);
    op.Process();
}
