// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_bwd_off_finalize_grouped.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_bwd_off_finalize_grouped(
    const at::Tensor &dStatesHalf,
    const at::Tensor &dCGroupHalf,
    const at::Tensor &qGroupHalf,
    const at::Tensor &yBaseGroupHalf)
{
    for (const auto *tensor :
         {&dStatesHalf, &dCGroupHalf, &qGroupHalf, &yBaseGroupHalf}) {
        TORCH_CHECK(tensor->device().type() == DEVICE_TYPE &&
                    tensor->scalar_type() == at::kHalf &&
                    tensor->is_contiguous(),
                    "mamba2_ssd_bwd_off_finalize_grouped: all inputs must "
                    "be contiguous FP16 NPU tensors");
    }
    TORCH_CHECK(dStatesHalf.dim() == 6 && dCGroupHalf.dim() == 5 &&
                qGroupHalf.dim() == 6 && yBaseGroupHalf.dim() == 6,
                "mamba2_ssd_bwd_off_finalize_grouped: invalid ranks");
    const int64_t batch = qGroupHalf.size(0);
    const int64_t chunks = qGroupHalf.size(1);
    const int64_t groups = qGroupHalf.size(2);
    const int64_t headsPerGroup = qGroupHalf.size(4);
    TORCH_CHECK(batch > 0 && chunks > 0 && groups > 0 &&
                headsPerGroup == 4 && qGroupHalf.size(3) == 64 &&
                qGroupHalf.size(5) == 64 &&
                yBaseGroupHalf.sizes() == qGroupHalf.sizes() &&
                dStatesHalf.sizes() == at::IntArrayRef(
                    {batch, chunks, groups, 64, headsPerGroup, 64}) &&
                dCGroupHalf.sizes() == at::IntArrayRef(
                    {batch, chunks, groups, 64, 64}),
                "mamba2_ssd_bwd_off_finalize_grouped: expected "
                "dState[B,K,G,N,4,P], dC[B,K,G,T,N], and "
                "Q/Y[B,K,G,T,4,P]");

    auto fp32 = qGroupHalf.options().dtype(at::kFloat);
    at::Tensor dCGroup = at::empty(
        {batch, chunks, 64, groups, 64}, fp32);
    at::Tensor gDa = at::empty(
        {batch, groups * headsPerGroup, chunks, 64}, fp32);
    const int64_t taskCount = batch * chunks * groups;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum =
        static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t usedCoreNum = std::min(taskCount, coreNum);
    TORCH_CHECK(usedCoreNum > 0,
                "mamba2_ssd_bwd_off_finalize_grouped: no AIV cores");
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    EXEC_KERNEL_CMD(
        mamba2_ssd_bwd_off_finalize_grouped,
        blockDim,
        dCGroupHalf, qGroupHalf, yBaseGroupHalf, dCGroup, gDa,
        batch, chunks, groups, headsPerGroup, usedCoreNum);
    return std::make_tuple(dStatesHalf, dCGroup, gDa);
}

}  // namespace ascend_kernel
