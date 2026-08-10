// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_bwd_off_prepare.h"

namespace ascend_kernel {

namespace {
constexpr int64_t kElementsPerChunk = 64 * 64;
constexpr int64_t kHalfStateBytesPerChunk =
    kElementsPerChunk * (sizeof(at::Half) + sizeof(float) + sizeof(float) +
                         sizeof(at::Half)) +
    64 * sizeof(float) + 2 * kElementsPerChunk;
constexpr int64_t kFloatStateExtraBytesPerChunk =
    kElementsPerChunk * (sizeof(float) + sizeof(at::Half));
constexpr int64_t kUbReserveBytes = 8 * 1024;
constexpr int64_t kMaxChunksPerTask = 3;
}  // namespace

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_bwd_off_prepare(
    const at::Tensor &gy,
    const at::Tensor &statesStart,
    const at::Tensor &dACumsum,
    int64_t groups)
{
    TORCH_CHECK(gy.device().type() == DEVICE_TYPE &&
                statesStart.device() == gy.device() &&
                dACumsum.device() == gy.device(),
                "mamba2_ssd_bwd_off_prepare: inputs must be on one NPU");
    const bool stateIsHalf = statesStart.scalar_type() == at::kHalf;
    TORCH_CHECK(gy.scalar_type() == at::kHalf &&
                (stateIsHalf || statesStart.scalar_type() == at::kFloat) &&
                dACumsum.scalar_type() == at::kFloat,
                "mamba2_ssd_bwd_off_prepare: gy must be float16 and "
                "states_start must be float16/float32 and dA float32");
    TORCH_CHECK(gy.is_contiguous() && statesStart.is_contiguous() &&
                dACumsum.is_contiguous(),
                "mamba2_ssd_bwd_off_prepare: inputs must be contiguous");
    TORCH_CHECK(gy.dim() == 5 && gy.size(3) == 64 && gy.size(4) == 64,
                "mamba2_ssd_bwd_off_prepare: gy must be [B,H,K,64,64]");
    const bool inputGrouped = statesStart.dim() == 6;
    TORCH_CHECK(groups > 0 && gy.size(1) % groups == 0,
                "mamba2_ssd_bwd_off_prepare: groups must divide heads");
    const int64_t headsPerGroup = gy.size(1) / groups;
    if (inputGrouped) {
        TORCH_CHECK(stateIsHalf && headsPerGroup == 4 &&
                    statesStart.sizes() == at::IntArrayRef(
                        {gy.size(0), gy.size(2), groups, 64,
                         headsPerGroup, 64}),
                    "mamba2_ssd_bwd_off_prepare: grouped states must be "
                    "FP16 [B,K,G,64,R,64] with R=4");
    } else {
        TORCH_CHECK(statesStart.sizes() == gy.sizes(),
                    "mamba2_ssd_bwd_off_prepare: states_start shape mismatch");
    }
    TORCH_CHECK(dACumsum.sizes() == at::IntArrayRef(
                    {gy.size(0), gy.size(1), gy.size(2), 64}),
                "mamba2_ssd_bwd_off_prepare: dA shape mismatch");

    auto halfOptions = gy.options().dtype(at::kHalf);
    at::Tensor qHalf = inputGrouped
        ? at::empty(
            {gy.size(0), gy.size(2), groups, 64, headsPerGroup, 64},
            halfOptions)
        : at::empty(gy.sizes(), halfOptions);
    at::Tensor stateHalf = stateIsHalf
        ? statesStart
        : at::empty(gy.sizes(), halfOptions);
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t bytesPerChunk = kHalfStateBytesPerChunk +
        (stateIsHalf ? 0 : kFloatStateExtraBytesPerChunk);
    TORCH_CHECK(static_cast<int64_t>(ubSize) >
                    kUbReserveBytes + bytesPerChunk,
                "mamba2_ssd_bwd_off_prepare: insufficient UB capacity");
    int64_t chunksPerTask =
        (static_cast<int64_t>(ubSize) - kUbReserveBytes) / bytesPerChunk;
    chunksPerTask = std::max<int64_t>(
        1, std::min<int64_t>(chunksPerTask, kMaxChunksPerTask));
    const int64_t batch = gy.size(0);
    const int64_t nheads = gy.size(1);
    const int64_t nchunks = gy.size(2);
    const int64_t chunkTiles =
        (nchunks + chunksPerTask - 1) / chunksPerTask;
    const int64_t taskCount = batch * nheads * chunkTiles;
    const int64_t coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t usedCoreNum = std::min(taskCount, coreNum);
    const int64_t stateIsHalfArg = stateIsHalf ? 1 : 0;
    const int64_t inputGroupedArg = inputGrouped ? 1 : 0;
    TORCH_CHECK(usedCoreNum > 0, "mamba2_ssd_bwd_off_prepare: no AIV cores");
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    EXEC_KERNEL_CMD(
        mamba2_ssd_bwd_off_prepare, blockDim,
        gy, statesStart, dACumsum, qHalf, stateHalf,
        batch, nheads, nchunks, chunksPerTask,
        taskCount, usedCoreNum, stateIsHalfArg, groups,
        inputGroupedArg);
    return std::make_tuple(qHalf, stateHalf);
}

}  // namespace ascend_kernel
