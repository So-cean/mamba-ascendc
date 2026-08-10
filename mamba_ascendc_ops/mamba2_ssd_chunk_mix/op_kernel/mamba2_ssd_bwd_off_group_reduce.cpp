/**
 * Copyright (c) 2026, mamba-ascendc authors.
 *
 * Group-owned off-diagonal backward reduction for T=P=N=64.  Cube produces
 * per-head dC tiles into a bounded ping-pong workspace while the paired
 * Vector cores reduce the preceding tile into dC_group and g_dA.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mamba2_dynamic_matmul.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTile = 64;
constexpr uint32_t kTileElements = kTile * kTile;
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
constexpr uint32_t kAivPerAic = 1;
#else
constexpr uint32_t kAivPerAic = 2;
#endif
static_assert(kAivPerAic == 1 || kAivPerAic == 2,
              "unsupported MIX Vector/Cube ratio");
// 910B3 mode-2 synchronization is a 1-AIC:2-AIV aggregation barrier. Both
// AIVs own half of the rows and rendezvous through a mode-1 barrier before
// publishing their mode-2 free tokens. 950PR has a 1:1 topology and owns all
// rows on its single AIV.
constexpr uint32_t kRowsPerAiv = kTile / kAivPerAic;
constexpr uint32_t kBlockElements = kRowsPerAiv * kTile;
constexpr uint16_t kReady0 = 0x8;
constexpr uint16_t kReady1 = 0x9;
constexpr uint16_t kFree0 = 0xA;
constexpr uint16_t kFree1 = 0xB;
#if !((defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
      defined(__DAV_310R6__))
constexpr uint16_t kAivDone0 = 0xC;
constexpr uint16_t kAivDone1 = 0xD;
#endif

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
constexpr MatmulConfig kArch35MatmulConfig = GetBasicConfig(64, 64, 64);
using Arch35AType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Arch35BType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using Arch35CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using Arch35BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

class Arch35GroupReduceMatmul64 {
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

class VectorGroupReduce {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(cHalfBuf_, kBlockElements * sizeof(half));
        pipe->InitBuffer(dCInBuf_, kBlockElements * sizeof(float));
        pipe->InitBuffer(cFloatBuf_, kBlockElements * sizeof(float));
        pipe->InitBuffer(dCAccBuf_, kBlockElements * sizeof(float));
        pipe->InitBuffer(productBuf_, kBlockElements * sizeof(float));
        pipe->InitBuffer(gDaBuf_, kRowsPerAiv * sizeof(float));
        mte2ToV_ = static_cast<event_t>(
            pipe->FetchEventID(HardEvent::MTE2_V));
        vToMte3_ = static_cast<event_t>(
            pipe->FetchEventID(HardEvent::V_MTE3));
        mte3ToV_ = static_cast<event_t>(
            pipe->FetchEventID(HardEvent::MTE3_V));
        vToMte2_ = static_cast<event_t>(
            pipe->FetchEventID(HardEvent::V_MTE2));
    }

    __aicore__ inline void PrepareGroup(
        const GlobalTensor<half> &cCube)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        auto cHalf = cHalfBuf_.Get<half>();
        CopyIn(cHalf, cCube, rowBegin * kTile, kBlockElements);
        LoadBarrier();
        auto cFloat = cFloatBuf_.Get<float>();
        Cast(cFloat, cHalf, RoundMode::CAST_NONE, kBlockElements);
        auto dCAcc = dCAccBuf_.Get<float>();
        Duplicate(dCAcc, 0.0f, kBlockElements);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadHead(
        const GlobalTensor<float> &dCTile, bool reuseInput)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        if (reuseInput) {
            FinishVectorForLoadReuse();
        }
        auto dC = dCInBuf_.Get<float>();
        CopyIn(dC, dCTile, rowBegin * kTile, kBlockElements);
        LoadBarrier();
    }

    __aicore__ inline void ComputeHead(const GlobalTensor<float> &gDa)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        auto dC = dCInBuf_.Get<float>();
        auto cFloat = cFloatBuf_.Get<float>();
        auto dCAcc = dCAccBuf_.Get<float>();
        auto product = productBuf_.Get<float>();
        Mul(product, dC, cFloat, kBlockElements);
        Add(dCAcc, dCAcc, dC, kBlockElements);
        PipeBarrier<PIPE_V>();

        auto gDaLocal = gDaBuf_.Get<float>();
        WholeReduceSum<float, true>(
            gDaLocal, product, static_cast<int32_t>(kTile),
            static_cast<int32_t>(kRowsPerAiv), 1, 1,
            static_cast<int32_t>(
                kTile * sizeof(float) / DEFAULT_C0_SIZE));
        PipeBarrier<PIPE_V>();
        StoreBarrier();
        CopyOut(gDa, gDaLocal, rowBegin, kRowsPerAiv);
        FinishStoreForVectorReuse();
    }

    __aicore__ inline void FinalizeGroup(
        const GlobalTensor<float> &dCGroup,
        uint32_t batch, uint32_t chunk, uint32_t group,
        uint32_t groups, uint32_t chunks)
    {
        const uint32_t rowBegin = GetSubBlockIdx() * kRowsPerAiv;
        const uint64_t offset =
            ((static_cast<uint64_t>(batch) * chunks + chunk) * kTile *
                 groups +
             group) *
                kTile +
            static_cast<uint64_t>(rowBegin) * groups * kTile;
        DataCopyExtParams copy{
            static_cast<uint16_t>(kRowsPerAiv),
            static_cast<uint32_t>(kTile * sizeof(float)), 0,
            static_cast<uint32_t>((groups - 1) * kTile * sizeof(float)),
            0};
        StoreBarrier();
        DataCopyPad(dCGroup[offset], dCAccBuf_.Get<float>(), copy);
        FinishStoreForVectorReuse();
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
        const GlobalTensor<T> &dst, const LocalTensor<T> &src,
        uint64_t offset, uint32_t elements)
    {
        DataCopyExtParams copy{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        DataCopyPad(dst[offset], src, copy);
    }

    __aicore__ inline void LoadBarrier()
    {
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void StoreBarrier()
    {
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
    }

    __aicore__ inline void FinishStoreForVectorReuse()
    {
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void FinishVectorForLoadReuse()
    {
        SetFlag<HardEvent::V_MTE2>(vToMte2_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2_);
    }

    TBuf<TPosition::VECCALC> cHalfBuf_, dCInBuf_;
    TBuf<TPosition::VECCALC> cFloatBuf_, dCAccBuf_, productBuf_;
    TBuf<TPosition::VECCALC> gDaBuf_;
    event_t mte2ToV_, vToMte3_, mte3ToV_, vToMte2_;
};

class KernelMamba2SsdBwdOffGroupReduce {
public:
    __aicore__ inline void Init(
        GM_ADDR qGroup, GM_ADDR stateGroup, GM_ADDR cCube,
        GM_ADDR dCGroup, GM_ADDR gDa, GM_ADDR workspace,
        const Mamba2SsdBwdOffGroupReduceTilingData &tiling,
        TPipe *pipe)
    {
        tiling_ = tiling;
        const uint64_t headTasks =
            static_cast<uint64_t>(tiling_.batch) * tiling_.groups *
            tiling_.headsPerGroup * tiling_.chunks;
        const uint64_t groupTasks = tiling_.taskCount;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qGroup),
                             headTasks * kTileElements);
        stateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(stateGroup),
            headTasks * kTileElements);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cCube),
                             groupTasks * kTileElements);
        dCGroupGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(dCGroup),
            groupTasks * kTileElements);
        gDaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gDa),
                               headTasks * kTile);

        coreIdx_ = GetBlockIdx() / GetSubBlockNum();
        GM_ADDR coreWorkspace = GetUserWorkspace(workspace) +
            static_cast<uint64_t>(coreIdx_) *
                tiling_.workspaceBytesPerCore;
        workspaceFloat_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(coreWorkspace),
            2 * kTileElements);
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
            const uint32_t group = task % tiling_.groups;
            const uint32_t chunk =
                (task / tiling_.groups) % tiling_.chunks;
            const uint32_t batch =
                task / (tiling_.groups * tiling_.chunks);
            const uint64_t cOffset =
                ((static_cast<uint64_t>(batch) * tiling_.chunks + chunk) *
                     tiling_.groups +
                 group) *
                kTileElements;

            if ASCEND_IS_AIV {
                vector_.PrepareGroup(cGm_[cOffset]);
            }

            for (uint32_t headInGroup = 0;
                 headInGroup < tiling_.headsPerGroup; ++headInGroup) {
                const uint32_t slot = headInGroup & 1U;
                const uint64_t inputTask =
                    (((static_cast<uint64_t>(batch) * tiling_.groups +
                       group) *
                          tiling_.headsPerGroup +
                      headInGroup) *
                         tiling_.chunks +
                     chunk);
                const uint32_t head =
                    group * tiling_.headsPerGroup + headInGroup;
                const uint64_t outputTask =
                    (static_cast<uint64_t>(batch) *
                         tiling_.groups * tiling_.headsPerGroup +
                     head) *
                        tiling_.chunks +
                    chunk;
                const uint64_t matrixOffset = inputTask * kTileElements;
                const uint64_t vectorOffset = outputTask * kTile;
                auto temporary = workspaceFloat_[slot * kTileElements];

                if ASCEND_IS_AIC {
                    if (headInGroup >= 2) {
                        WaitFree(slot);
                    }
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
                    matmul_.ComputeBlock(
                        qGm_[matrixOffset], stateGm_[matrixOffset],
                        temporary);
#else
                    matmul_.ComputeBlock(
                        qGm_[matrixOffset], stateGm_[matrixOffset],
                        temporary, kTile, kTile, kTile, kTile);
#endif
                    SetReady(slot);
                }
                if ASCEND_IS_AIV {
                    WaitReady(slot);
                    vector_.LoadHead(temporary, headInGroup != 0);
                    SyncAivs(slot);
                    SetFree(slot);
                    vector_.ComputeHead(gDaGm_[vectorOffset]);
                }
            }

            if ASCEND_IS_AIC {
                if (tiling_.headsPerGroup == 1) {
                    WaitFree(0);
                } else {
                    WaitFree((tiling_.headsPerGroup - 2) & 1U);
                    WaitFree((tiling_.headsPerGroup - 1) & 1U);
                }
            }
            if ASCEND_IS_AIV {
                vector_.FinalizeGroup(
                    dCGroupGm_, batch, chunk, group,
                    tiling_.groups, tiling_.chunks);
            }
        }
    }

private:
    __aicore__ inline void SyncAivs(uint32_t slot)
    {
#if !((defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
      defined(__DAV_310R6__))
        if (slot == 0) {
            CrossCoreSetFlag<0x1, PIPE_MTE2>(kAivDone0);
            CrossCoreWaitFlag<0x1>(kAivDone0);
        } else {
            CrossCoreSetFlag<0x1, PIPE_MTE2>(kAivDone1);
            CrossCoreWaitFlag<0x1>(kAivDone1);
        }
#endif
    }

    __aicore__ inline void SetReady(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(kReady0);
        } else {
            CrossCoreSetFlag<0x2, PIPE_FIX>(kReady1);
        }
    }

    __aicore__ inline void WaitReady(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreWaitFlag<0x2>(kReady0);
        } else {
            CrossCoreWaitFlag<0x2>(kReady1);
        }
    }

    __aicore__ inline void SetFree(uint32_t slot)
    {
        // The shared GM tile is dead as soon as MTE2 has copied the owned
        // row slab into UB.  Publishing from PIPE_MTE3 is incorrect here:
        // it orders the unrelated g_dA store but not the preceding MTE2 read,
        // so Cube may overwrite the ping-pong tile while a 910B3 AIV is still
        // loading it.  PIPE_MTE2 both fixes that race and lets Cube overlap
        // with the remaining Vector arithmetic.
        if (slot == 0) {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(kFree0);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE2>(kFree1);
        }
    }

    __aicore__ inline void WaitFree(uint32_t slot)
    {
        if (slot == 0) {
            CrossCoreWaitFlag<0x2>(kFree0);
        } else {
            CrossCoreWaitFlag<0x2>(kFree1);
        }
    }

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || \
    defined(__DAV_310R6__)
    Arch35GroupReduceMatmul64 matmul_;
#else
    Mamba2DynamicMatmul<half, float, 64> matmul_;
#endif
    VectorGroupReduce vector_;
    GlobalTensor<half> qGm_, stateGm_, cGm_;
    GlobalTensor<float> dCGroupGm_, gDaGm_, workspaceFloat_;
    Mamba2SsdBwdOffGroupReduceTilingData tiling_;
    uint32_t coreIdx_ = 0;
};
} // namespace

extern "C" __global__ __aicore__ void mamba2_ssd_bwd_off_group_reduce(
    GM_ADDR q_group, GM_ADDR state_group, GM_ADDR c_cube,
    GM_ADDR d_c_group, GM_ADDR g_d_a,
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
        KernelMamba2SsdBwdOffGroupReduce op;
        op.Init(q_group, state_group, c_cube,
                d_c_group, g_d_a, workspace, tilingData, &pipe);
        op.Process();
    }
}
