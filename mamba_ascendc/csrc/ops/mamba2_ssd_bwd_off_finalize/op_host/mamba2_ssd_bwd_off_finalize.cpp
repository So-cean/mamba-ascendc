// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_bwd_off_finalize.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_bwd_off_finalize(
    const at::Tensor &dStatesHalf,
    const at::Tensor &dCHeadHalf,
    const at::Tensor &cCube)
{
    TORCH_CHECK(dStatesHalf.device().type() == DEVICE_TYPE &&
                dCHeadHalf.device() == dStatesHalf.device() &&
                cCube.device() == dStatesHalf.device(),
                "mamba2_ssd_bwd_off_finalize: inputs must be on one NPU");
    TORCH_CHECK(dStatesHalf.scalar_type() == at::kHalf &&
                dCHeadHalf.scalar_type() == at::kHalf &&
                cCube.scalar_type() == at::kHalf,
                "mamba2_ssd_bwd_off_finalize: inputs must be float16");
    TORCH_CHECK(dStatesHalf.is_contiguous() && dCHeadHalf.is_contiguous() &&
                cCube.is_contiguous(),
                "mamba2_ssd_bwd_off_finalize: inputs must be contiguous");
    TORCH_CHECK(dStatesHalf.dim() == 6 &&
                dStatesHalf.size(4) == 64 && dStatesHalf.size(5) == 64,
                "mamba2_ssd_bwd_off_finalize: matrices must be "
                "[B,G,Q,K,64,64]");
    TORCH_CHECK(dCHeadHalf.sizes() == dStatesHalf.sizes(),
                "mamba2_ssd_bwd_off_finalize: BMM output shape mismatch");
    const int64_t batch = dStatesHalf.size(0);
    const int64_t groups = dStatesHalf.size(1);
    const int64_t headsPerGroup = dStatesHalf.size(2);
    const int64_t chunks = dStatesHalf.size(3);
    const int64_t heads = groups * headsPerGroup;
    TORCH_CHECK(headsPerGroup > 0,
                "mamba2_ssd_bwd_off_finalize: heads per group must be positive");
    TORCH_CHECK(cCube.sizes() == at::IntArrayRef(
                    {batch, chunks, groups, 64, 64}),
                "mamba2_ssd_bwd_off_finalize: c_cube shape mismatch");

    auto fp32 = dStatesHalf.options().dtype(at::kFloat);
    // The BMM result is contiguous [B,G,R,K,64,64].  Flattening the adjacent
    // G/R dimensions gives the exact head-major state-gradient layout, so do
    // not read/cast/write this GiB-scale tensor in the finalize kernel.  The
    // state-passing consumer performs the one required FP16->FP32 cast after
    // its own GM read.
    at::Tensor dStates = dStatesHalf.view(
        {batch, heads, chunks, 64, 64});
    // Match the public C layout [B, K, T, G, N] so the two C-gradient
    // branches can be added without materializing device transposes.
    at::Tensor dCGroup = at::empty(
        {batch, chunks, 64, groups, 64}, fp32);
    at::Tensor gDa = at::empty({batch, heads, chunks, 64}, fp32);
    const int64_t taskCount = batch * groups * chunks;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t usedCoreNum = std::min(taskCount, coreNum);
    TORCH_CHECK(usedCoreNum > 0, "mamba2_ssd_bwd_off_finalize: no AIV cores");
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    EXEC_KERNEL_CMD(
        mamba2_ssd_bwd_off_finalize, blockDim,
        dStatesHalf, dCHeadHalf, cCube, dStates, dCGroup, gDa,
        batch, groups, headsPerGroup, chunks, usedCoreNum);
    return std::make_tuple(dStates, dCGroup, gDa);
}

}  // namespace ascend_kernel
