// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_prepare_dcb.h"

namespace ascend_kernel {

at::Tensor mamba2_ssd_prepare_dcb(
    const at::Tensor &dW,
    const at::Tensor &dACumsum,
    int64_t groups)
{
    TORCH_CHECK(dW.device().type() == DEVICE_TYPE &&
                dACumsum.device() == dW.device(),
                "mamba2_ssd_prepare_dcb: inputs must be on one NPU");
    TORCH_CHECK(dW.scalar_type() == at::kHalf &&
                dACumsum.scalar_type() == at::kFloat,
                "mamba2_ssd_prepare_dcb: requires FP16 dW and FP32 dA");
    TORCH_CHECK(dW.is_contiguous() && dACumsum.is_contiguous(),
                "mamba2_ssd_prepare_dcb: inputs must be contiguous");
    TORCH_CHECK(dW.dim() == 5 && dACumsum.dim() == 4,
                "mamba2_ssd_prepare_dcb: expected dW[B,H,K,64,64] "
                "and dA[B,H,K,64]");
    const int64_t batch = dACumsum.size(0);
    const int64_t heads = dACumsum.size(1);
    const int64_t chunks = dACumsum.size(2);
    TORCH_CHECK(dACumsum.size(3) == 64 &&
                dW.sizes() == at::IntArrayRef(
                    {batch, heads, chunks, 64, 64}),
                "mamba2_ssd_prepare_dcb: shape mismatch");
    TORCH_CHECK(groups > 0 && heads % groups == 0,
                "mamba2_ssd_prepare_dcb: groups must divide heads");
    const int64_t headsPerGroup = heads / groups;
    constexpr int64_t kMaxHeadBlock = 4;
    const int64_t headBlockSize = std::min(headsPerGroup, kMaxHeadBlock);
    at::Tensor dCbGroup = at::empty(
        {batch, groups, chunks, 64, 64}, dW.options());
    const int64_t taskCount = batch * groups * chunks;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t requiredUbBytes =
        headBlockSize * 64 * 64 * sizeof(uint16_t) +
        headBlockSize * 64 * sizeof(float) +
        64 * 64 * sizeof(uint16_t) +
        3 * 64 * 64 * sizeof(float) +
        2 * 64 * 64;
    TORCH_CHECK(ubSize >= requiredUbBytes,
                "mamba2_ssd_prepare_dcb: insufficient UB");
    const int64_t coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t usedCoreNum = std::min(taskCount, coreNum);
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    TORCH_CHECK(usedCoreNum > 0, "mamba2_ssd_prepare_dcb: no AIV cores");
    EXEC_KERNEL_CMD(
        mamba2_ssd_prepare_dcb, blockDim,
        dW, dACumsum, dCbGroup, batch, heads, chunks, groups,
        headsPerGroup, headBlockSize, usedCoreNum);
    return dCbGroup;
}

}  // namespace ascend_kernel
