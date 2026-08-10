// Copyright (c) 2026, mamba-ascendc authors.

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_prepare_w.h"
#include "aclrtlaunch_mamba2_ssd_prepare_r.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_prepare_split(
    const at::Tensor &cb,
    const at::Tensor &dACumsum,
    const at::Tensor &xCube)
{
    TORCH_CHECK(cb.device().type() == DEVICE_TYPE &&
                cb.scalar_type() == at::kHalf && cb.is_contiguous(),
                "mamba2_ssd_prepare_split: cb must be contiguous FP16 NPU");
    TORCH_CHECK(dACumsum.device().type() == DEVICE_TYPE &&
                dACumsum.scalar_type() == at::kFloat &&
                dACumsum.is_contiguous(),
                "mamba2_ssd_prepare_split: dA_cumsum must be contiguous FP32 NPU");
    TORCH_CHECK(xCube.device().type() == DEVICE_TYPE &&
                xCube.scalar_type() == at::kHalf && xCube.is_contiguous(),
                "mamba2_ssd_prepare_split: x_cube must be contiguous FP16 NPU");
    TORCH_CHECK(cb.dim() == 5 && dACumsum.dim() == 4 && xCube.dim() == 5,
                "mamba2_ssd_prepare_split: expected cb[B,C,G,64,64], "
                "dA[B,H,C,64], x[B,H,C,64,64]");

    const int64_t batch = dACumsum.size(0);
    const int64_t nheads = dACumsum.size(1);
    const int64_t nchunks = dACumsum.size(2);
    const int64_t chunkSize = dACumsum.size(3);
    const int64_t ngroups = cb.size(2);
    TORCH_CHECK(chunkSize == 64 && xCube.size(3) == 64 && xCube.size(4) == 64,
                "mamba2_ssd_prepare_split: only chunk_size=headdim=64 is supported");
    TORCH_CHECK(cb.size(0) == batch && cb.size(1) == nchunks &&
                cb.size(3) == 64 && cb.size(4) == 64,
                "mamba2_ssd_prepare_split: cb shape mismatch");
    TORCH_CHECK(xCube.size(0) == batch && xCube.size(1) == nheads &&
                xCube.size(2) == nchunks,
                "mamba2_ssd_prepare_split: x shape mismatch");
    TORCH_CHECK(ngroups > 0 && nheads % ngroups == 0,
                "mamba2_ssd_prepare_split: nheads must be divisible by ngroups");

    at::Tensor w = at::empty(
        {batch, nheads, nchunks, 64, 64}, cb.options());
    at::Tensor weightedX = at::empty_like(xCube);
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t taskCount = batch * nheads * nchunks;
    const int64_t usedCoreNum = std::min(taskCount, aivCoreNum);
    TORCH_CHECK(usedCoreNum > 0, "mamba2_ssd_prepare_split: no AIV cores");
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(mamba2_ssd_prepare_w, blockDim,
                    cb, dACumsum, w,
                    batch, nheads, nchunks, ngroups, usedCoreNum);
    EXEC_KERNEL_CMD(mamba2_ssd_prepare_r, blockDim,
                    dACumsum, xCube, weightedX,
                    batch, nheads, nchunks, usedCoreNum);
    return std::make_tuple(w, weightedX);
}

}  // namespace ascend_kernel
