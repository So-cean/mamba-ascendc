// Copyright (c) 2026, mamba-ascendc authors.

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_prepare.h"

namespace ascend_kernel {
namespace {

void CheckHalfNpu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_prepare: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf,
                "mamba2_ssd_prepare: ", name, " must be float16");
    TORCH_CHECK(tensor.is_contiguous(),
                "mamba2_ssd_prepare: ", name, " must be contiguous");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_prepare(
    const at::Tensor &cb,
    const at::Tensor &dACumsum,
    const at::Tensor &xCube)
{
    CheckHalfNpu(cb, "cb");
    CheckHalfNpu(xCube, "x_cube");
    TORCH_CHECK(dACumsum.device().type() == DEVICE_TYPE &&
                dACumsum.scalar_type() == at::kFloat &&
                dACumsum.is_contiguous(),
                "mamba2_ssd_prepare: dA_cumsum must be contiguous FP32 NPU");
    TORCH_CHECK(cb.dim() == 5 && dACumsum.dim() == 4 && xCube.dim() == 5,
                "mamba2_ssd_prepare: expected cb[B,C,G,T,T], dA[B,H,C,T], "
                "x_cube[B,H,C,T,P]");

    const int64_t batch = dACumsum.size(0);
    const int64_t nheads = dACumsum.size(1);
    const int64_t nchunks = dACumsum.size(2);
    const int64_t chunkSize = dACumsum.size(3);
    const int64_t ngroups = cb.size(2);
    const int64_t headdim = xCube.size(4);
    TORCH_CHECK(cb.size(0) == batch && cb.size(1) == nchunks &&
                cb.size(3) == chunkSize && cb.size(4) == chunkSize,
                "mamba2_ssd_prepare: cb shape mismatch");
    TORCH_CHECK(xCube.size(0) == batch && xCube.size(1) == nheads &&
                xCube.size(2) == nchunks && xCube.size(3) == chunkSize,
                "mamba2_ssd_prepare: x_cube shape mismatch");
    TORCH_CHECK(ngroups > 0 && nheads % ngroups == 0,
                "mamba2_ssd_prepare: nheads must be divisible by ngroups");
    TORCH_CHECK(chunkSize > 0 && chunkSize <= 128 && chunkSize % 16 == 0,
                "mamba2_ssd_prepare: chunk_size must be a multiple of 16 and <= 128");
    TORCH_CHECK(headdim > 0 && headdim <= 128 && headdim % 16 == 0,
                "mamba2_ssd_prepare: headdim must be a multiple of 16 and <= 128");

    at::Tensor wCube = at::empty(
        {batch, nheads, nchunks, chunkSize, chunkSize}, cb.options());
    at::Tensor weightedXCube = at::empty_like(xCube);

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t taskCount = batch * nheads * nchunks;
    const int64_t usedCoreNum = std::min(taskCount, aivCoreNum);
    TORCH_CHECK(usedCoreNum > 0, "mamba2_ssd_prepare: no AIV cores");
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(mamba2_ssd_prepare, blockDim,
                    cb, dACumsum, xCube, wCube, weightedXCube,
                    batch, nheads, nchunks, ngroups, chunkSize, headdim,
                    usedCoreNum);
    return std::make_tuple(wCube, weightedXCube);
}

}  // namespace ascend_kernel
