// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include "kernel_operator.h"

namespace {
constexpr int64_t kTile = 64;
constexpr int64_t kHeadsPerGroup = 4;
constexpr int64_t kWide = kHeadsPerGroup * kTile;
constexpr int64_t kRows = 16;

class KernelMamba2SsdBwdOffFinalizeGrouped {
public:
    __aicore__ inline void Init(
        GM_ADDR dCGroupHalf, GM_ADDR qGroupHalf, GM_ADDR yBaseGroupHalf,
        GM_ADDR dCGroup, GM_ADDR gDa, int64_t batch, int64_t chunks,
        int64_t groups, int64_t headsPerGroup, int64_t usedCoreNum,
        AscendC::TPipe *pipe)
    {
        dCHalfGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(dCGroupHalf));
        qGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(qGroupHalf));
        yBaseGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(yBaseGroupHalf));
        dCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dCGroup));
        gDaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gDa));
        batch_ = batch;
        chunks_ = chunks;
        groups_ = groups;
        headsPerGroup_ = headsPerGroup;
        usedCoreNum_ = usedCoreNum;
        pipe->InitBuffer(dCHalfBuf_, kTile * kTile * sizeof(half));
        pipe->InitBuffer(dCFloatBuf_, kTile * kTile * sizeof(float));
        pipe->InitBuffer(qBuf_, kRows * kWide * sizeof(half));
        pipe->InitBuffer(yBuf_, kRows * kWide * sizeof(half));
        pipe->InitBuffer(qFloatBuf_, kRows * kWide * sizeof(float));
        pipe->InitBuffer(yFloatBuf_, kRows * kWide * sizeof(float));
        pipe->InitBuffer(reduceBuf_, kRows * kHeadsPerGroup * sizeof(float));
        pipe->InitBuffer(gDaHeadBuf_, kHeadsPerGroup * kTile * sizeof(float));
        vToMte3_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3ToV_ = static_cast<event_t>(
            pipe->FetchEventID(AscendC::HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = batch_ * chunks_ * groups_;
        const int64_t block = AscendC::GetBlockIdx();
        for (int64_t task = block; task < taskCount;
             task += usedCoreNum_) {
            ProcessTask(task);
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

    __aicore__ inline void ProcessTask(int64_t task)
    {
        const int64_t group = task % groups_;
        const int64_t chunk = (task / groups_) % chunks_;
        const int64_t batch = task / (groups_ * chunks_);
        auto dCHalf = dCHalfBuf_.Get<half>();
        auto dCFloat = dCFloatBuf_.Get<float>();
        CopyIn(dCHalf, dCHalfGm_, task * kTile * kTile,
               kTile * kTile);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(dCFloat, dCHalf, AscendC::RoundMode::CAST_NONE,
                      kTile * kTile);
        AscendC::PipeBarrier<PIPE_V>();
        const int64_t dCOffset =
            ((batch * chunks_ + chunk) * kTile * groups_ + group) * kTile;
        AscendC::DataCopyExtParams dCStore{
            static_cast<uint16_t>(kTile),
            static_cast<uint32_t>(kTile * sizeof(float)),
            0,
            static_cast<uint32_t>((groups_ - 1) * kTile * sizeof(float)),
            0};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::DataCopyPad(dCGm_[dCOffset], dCFloat, dCStore);
        // A strided core processes multiple group tasks.  The next task
        // immediately reuses dCFloat for Cast, so wait for MTE3 completion
        // rather than relying on a Vector-only barrier.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);

        auto gDaHead = gDaHeadBuf_.Get<float>();
        AscendC::Duplicate(gDaHead, 0.0f, headsPerGroup_ * kTile);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t row = 0; row < kTile; row += kRows) {
            const int64_t offset = task * kTile * kWide + row * kWide;
            auto q = qBuf_.Get<half>();
            auto y = yBuf_.Get<half>();
            auto qFloat = qFloatBuf_.Get<float>();
            auto yFloat = yFloatBuf_.Get<float>();
            auto reduced = reduceBuf_.Get<float>();
            CopyIn(q, qGm_, offset, kRows * kWide);
            CopyIn(y, yBaseGm_, offset, kRows * kWide);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(qFloat, q, AscendC::RoundMode::CAST_NONE,
                          kRows * kWide);
            AscendC::Cast(yFloat, y, AscendC::RoundMode::CAST_NONE,
                          kRows * kWide);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(qFloat, qFloat, yFloat, kRows * kWide);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float, true>(
                reduced, qFloat, static_cast<int32_t>(kTile),
                static_cast<int32_t>(kRows * kHeadsPerGroup),
                1, 1,
                static_cast<int32_t>(kTile * sizeof(float) /
                                     AscendC::DEFAULT_C0_SIZE));
            AscendC::PipeBarrier<PIPE_V>();
            for (int64_t localRow = 0; localRow < kRows; ++localRow) {
                for (int64_t head = 0; head < headsPerGroup_; ++head) {
                    gDaHead.SetValue(
                        head * kTile + row + localRow,
                        reduced.GetValue(localRow * headsPerGroup_ + head));
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_);
        for (int64_t head = 0; head < headsPerGroup_; ++head) {
            const int64_t globalHead = group * headsPerGroup_ + head;
            const int64_t outputOffset =
                ((batch * groups_ * headsPerGroup_ + globalHead) *
                 chunks_ + chunk) * kTile;
            AscendC::DataCopyExtParams store{
                1, static_cast<uint32_t>(kTile * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(
                gDaGm_[outputOffset], gDaHead[head * kTile], store);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_);
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> dCHalfBuf_, dCFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qBuf_, yBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qFloatBuf_, yFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_, gDaHeadBuf_;
    AscendC::GlobalTensor<half> dCHalfGm_, qGm_, yBaseGm_;
    AscendC::GlobalTensor<float> dCGm_, gDaGm_;
    event_t vToMte3_, mte3ToV_;
    int64_t batch_, chunks_, groups_, headsPerGroup_, usedCoreNum_;
};
}  // namespace

extern "C" __global__ __aicore__ void
mamba2_ssd_bwd_off_finalize_grouped(
    GM_ADDR d_c_group_half, GM_ADDR q_group_half,
    GM_ADDR y_base_group_half, GM_ADDR d_c_group, GM_ADDR g_d_a,
    int64_t batch, int64_t chunks, int64_t groups,
    int64_t heads_per_group, int64_t used_core_num)
{
    KernelMamba2SsdBwdOffFinalizeGrouped op;
    AscendC::TPipe pipe;
    op.Init(d_c_group_half, q_group_half, y_base_group_half,
            d_c_group, g_d_a, batch, chunks, groups,
            heads_per_group, used_core_num, &pipe);
    op.Process();
}
