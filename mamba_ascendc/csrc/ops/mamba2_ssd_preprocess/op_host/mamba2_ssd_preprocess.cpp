// Copyright (c) 2026, mamba-ascendc authors.

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_preprocess.h"
#include "aclrtlaunch_mamba2_ssd_preprocess_grouped.h"

namespace ascend_kernel {
namespace {

void CheckFloatNpu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_preprocess: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                "mamba2_ssd_preprocess: ", name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(),
                "mamba2_ssd_preprocess: ", name, " must be contiguous");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_preprocess(
    const at::Tensor &x,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const c10::optional<at::Tensor> &dtBias,
    int64_t chunkSize,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    CheckFloatNpu(x, "x");
    CheckFloatNpu(dt, "dt");
    CheckFloatNpu(A, "A");
    CheckFloatNpu(B, "B");
    CheckFloatNpu(C, "C");
    if (dtBias.has_value()) {
        CheckFloatNpu(dtBias.value(), "dt_bias");
    }
    TORCH_CHECK(x.dim() == 4 && dt.dim() == 3 && A.dim() == 1,
                "mamba2_ssd_preprocess: invalid x/dt/A rank");
    TORCH_CHECK(B.dim() == 4 && B.sizes() == C.sizes(),
                "mamba2_ssd_preprocess: B/C must have matching rank-4 shapes");

    const int64_t batch = x.size(0);
    const int64_t seqlen = x.size(1);
    const int64_t nheads = x.size(2);
    const int64_t headdim = x.size(3);
    const int64_t ngroups = B.size(2);
    const int64_t dstate = B.size(3);
    TORCH_CHECK(dt.sizes() == at::IntArrayRef({batch, seqlen, nheads}),
                "mamba2_ssd_preprocess: dt shape mismatch");
    TORCH_CHECK(A.numel() == nheads && B.size(0) == batch && B.size(1) == seqlen,
                "mamba2_ssd_preprocess: A/B/C shape mismatch");
    TORCH_CHECK(nheads % ngroups == 0,
                "mamba2_ssd_preprocess: nheads must be divisible by ngroups");
    TORCH_CHECK(chunkSize > 0 && seqlen % chunkSize == 0,
                "mamba2_ssd_preprocess: seqlen must be divisible by chunk_size");
    TORCH_CHECK(chunkSize % 16 == 0 && headdim % 16 == 0 && dstate % 16 == 0,
                "mamba2_ssd_preprocess: chunk/headdim/dstate must be multiples of 16");
    TORCH_CHECK(dtLimitMin <= dtLimitMax,
                "mamba2_ssd_preprocess: invalid dt limits");
    if (dtBias.has_value()) {
        TORCH_CHECK(dtBias.value().dim() == 1 && dtBias.value().numel() == nheads,
                    "mamba2_ssd_preprocess: dt_bias shape mismatch");
    }

    const int64_t nchunks = seqlen / chunkSize;
    at::Tensor xCube = at::empty(
        {batch, nheads, nchunks, chunkSize, headdim}, x.options().dtype(at::kHalf));
    at::Tensor dACumsum = at::empty(
        {batch, nheads, nchunks, chunkSize}, x.options());
    // B is consumed twice by Cube as B^T.  Materialize that layout while the
    // fp32 input is already resident in UB so ChunkMix does not reread and
    // transpose B through a per-core GM workspace.
    at::Tensor bCube = at::empty(
        {batch, nchunks, ngroups, dstate, chunkSize}, x.options().dtype(at::kHalf));
    at::Tensor cCube = at::empty(
        {batch, nchunks, ngroups, chunkSize, dstate}, x.options().dtype(at::kHalf));

    at::Tensor dtBiasArg = dtBias.has_value() ? dtBias.value() : A;
    int64_t hasDtBias = dtBias.has_value() ? 1 : 0;
    int64_t dtSoftplusFlag = dtSoftplus ? 1 : 0;
    float dtMin = static_cast<float>(dtLimitMin);
    float dtMax = static_cast<float>(dtLimitMax);
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(aivCoreNum > 0, "mamba2_ssd_preprocess: no AIV cores");
    const int64_t taskCount = std::max(
        batch * (nheads >= 32 && nheads % 4 == 0 && chunkSize == 64 &&
                 headdim == 64 ? nheads / 4 : nheads) * nchunks,
        batch * nchunks * ngroups);
    int64_t usedCoreNum = std::min(taskCount, aivCoreNum);
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(mamba2_ssd_preprocess, blockDim,
                    x, dt, A, B, C, dtBiasArg,
                    xCube, dACumsum, bCube, cCube,
                    batch, seqlen, nheads, headdim, dstate, ngroups,
                    chunkSize, hasDtBias, dtSoftplusFlag, dtMin, dtMax,
                    usedCoreNum);
    return std::make_tuple(xCube, dACumsum, bCube, cCube);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_preprocess_grouped(
    const at::Tensor &x,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const c10::optional<at::Tensor> &dtBias,
    int64_t chunkSize,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    CheckFloatNpu(x, "x");
    CheckFloatNpu(dt, "dt");
    CheckFloatNpu(A, "A");
    CheckFloatNpu(B, "B");
    CheckFloatNpu(C, "C");
    if (dtBias.has_value()) {
        CheckFloatNpu(dtBias.value(), "dt_bias");
    }
    TORCH_CHECK(x.dim() == 4 && dt.dim() == 3 && A.dim() == 1 &&
                    B.dim() == 4 && B.sizes() == C.sizes(),
                "mamba2_ssd_preprocess_grouped: invalid input ranks");
    const int64_t batch = x.size(0);
    const int64_t seqlen = x.size(1);
    const int64_t nheads = x.size(2);
    const int64_t headdim = x.size(3);
    const int64_t ngroups = B.size(2);
    const int64_t dstate = B.size(3);
    TORCH_CHECK(
        batch > 0 && seqlen > 0 && ngroups > 0 && chunkSize == 64 &&
            headdim == 64 && dstate == 64 && nheads == 4 * ngroups &&
            seqlen % chunkSize == 0 &&
            dt.sizes() == at::IntArrayRef({batch, seqlen, nheads}) &&
            A.numel() == nheads && B.size(0) == batch &&
            B.size(1) == seqlen,
        "mamba2_ssd_preprocess_grouped: expected T=N=P=64 and H/G=4");
    TORCH_CHECK(dtLimitMin <= dtLimitMax,
                "mamba2_ssd_preprocess_grouped: invalid dt limits");
    if (dtBias.has_value()) {
        TORCH_CHECK(
            dtBias->dim() == 1 && dtBias->numel() == nheads,
            "mamba2_ssd_preprocess_grouped: dt_bias shape mismatch");
    }

    const int64_t nchunks = seqlen / chunkSize;
    at::Tensor xGrouped = at::empty(
        {batch, nchunks, ngroups, chunkSize, 4, headdim},
        x.options().dtype(at::kHalf));
    // Keep dA head-major in this first vertical slice so the only measured
    // variable is the dominant X producer/consumer layout.
    at::Tensor dACumsum = at::empty(
        {batch, nheads, nchunks, chunkSize}, x.options());
    at::Tensor bCube = at::empty(
        {batch, nchunks, ngroups, dstate, chunkSize},
        x.options().dtype(at::kHalf));
    at::Tensor cCube = at::empty(
        {batch, nchunks, ngroups, chunkSize, dstate},
        x.options().dtype(at::kHalf));

    at::Tensor dtBiasArg = dtBias.has_value() ? dtBias.value() : A;
    const int64_t hasDtBias = dtBias.has_value() ? 1 : 0;
    const int64_t dtSoftplusFlag = dtSoftplus ? 1 : 0;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(aivCoreNum > 0,
                "mamba2_ssd_preprocess_grouped: no AIV cores");
    const int64_t taskCount = batch * nchunks * ngroups;
    const int64_t usedCoreNum = std::min(taskCount, aivCoreNum);
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    float dtMin = static_cast<float>(dtLimitMin);
    float dtMax = static_cast<float>(dtLimitMax);
    EXEC_KERNEL_CMD(
        mamba2_ssd_preprocess_grouped,
        blockDim,
        x, dt, A, B, C, dtBiasArg,
        xGrouped, dACumsum, bCube, cCube,
        batch, seqlen, nheads, headdim, dstate, ngroups,
        chunkSize, hasDtBias, dtSoftplusFlag,
        dtMin, dtMax, usedCoreNum);
    return {xGrouped, dACumsum, bCube, cCube};
}

}  // namespace ascend_kernel
