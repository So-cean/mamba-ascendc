// Copyright (c) 2026, mamba-ascendc authors.

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_state_passing_grouped.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_state_passing_grouped(
    const at::Tensor &chunkStatesNp,
    const at::Tensor &dACumsum,
    const c10::optional<at::Tensor> &initialStatesNp,
    int64_t groups)
{
    TORCH_CHECK(chunkStatesNp.device().type() == DEVICE_TYPE &&
                    dACumsum.device().type() == DEVICE_TYPE,
                "mamba2_ssd_state_passing_grouped: inputs must be NPU "
                "tensors");
    TORCH_CHECK(chunkStatesNp.scalar_type() == at::kFloat &&
                    dACumsum.scalar_type() == at::kFloat,
                "mamba2_ssd_state_passing_grouped: expected FP32 inputs");
    TORCH_CHECK(chunkStatesNp.is_contiguous() && dACumsum.is_contiguous(),
                "mamba2_ssd_state_passing_grouped: inputs must be "
                "contiguous");
    const bool inputGrouped = chunkStatesNp.dim() == 6;
    TORCH_CHECK((chunkStatesNp.dim() == 5 || inputGrouped) &&
                    dACumsum.dim() == 4,
                "mamba2_ssd_state_passing_grouped: expected "
                "[B,H,K,N,P] or [B,K,G,N,R,P], and [B,H,K,T]");
    const int64_t batch = chunkStatesNp.size(0);
    const int64_t nheads = inputGrouped
        ? chunkStatesNp.size(2) * chunkStatesNp.size(4)
        : chunkStatesNp.size(1);
    const int64_t nchunks = inputGrouped
        ? chunkStatesNp.size(1) : chunkStatesNp.size(2);
    const int64_t dstate = inputGrouped
        ? chunkStatesNp.size(3) : chunkStatesNp.size(3);
    const int64_t headdim = inputGrouped
        ? chunkStatesNp.size(5) : chunkStatesNp.size(4);
    const int64_t chunkSize = dACumsum.size(3);
    TORCH_CHECK(batch > 0 && nheads > 0 && nchunks > 0 && groups > 0 &&
                    nheads % groups == 0 && nheads / groups == 4 &&
                    dstate == 64 && headdim == 64 && chunkSize == 64 &&
                    (!inputGrouped ||
                     (chunkStatesNp.size(2) == groups &&
                      chunkStatesNp.size(4) == 4)) &&
                    dACumsum.size(0) == batch &&
                    dACumsum.size(1) == nheads &&
                    dACumsum.size(2) == nchunks,
                "mamba2_ssd_state_passing_grouped: only N=P=T=64 and "
                "H/G=4 are supported");

    if (initialStatesNp.has_value()) {
        const auto &initial = initialStatesNp.value();
        TORCH_CHECK(initial.device().type() == DEVICE_TYPE &&
                        initial.scalar_type() == at::kFloat &&
                        initial.is_contiguous() &&
                        initial.sizes() == at::IntArrayRef(
                            {batch, nheads, dstate, headdim}),
                    "mamba2_ssd_state_passing_grouped: initial_states_np "
                    "must be contiguous FP32 [B,H,N,P]");
    }

    auto statesGrouped = at::empty(
        {batch, nchunks, groups, dstate, 4, headdim},
        chunkStatesNp.options().dtype(at::kHalf));
    auto finalStateNp = at::empty(
        {batch, nheads, dstate, headdim}, chunkStatesNp.options());
    auto initialArg = initialStatesNp.has_value()
        ? initialStatesNp.value() : chunkStatesNp;
    const int64_t hasInitial = initialStatesNp.has_value() ? 1 : 0;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum =
        static_cast<int64_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(aivCoreNum > 0,
                "mamba2_ssd_state_passing_grouped: no AIV cores");
    const int64_t usedCoreNum =
        std::min(batch * nheads, aivCoreNum);
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    int64_t inputGroupedValue = inputGrouped ? 1 : 0;
    EXEC_KERNEL_CMD(
        mamba2_ssd_state_passing_grouped, blockDim,
        chunkStatesNp, dACumsum, initialArg, statesGrouped, finalStateNp,
        batch, nheads, nchunks, groups, chunkSize, hasInitial, usedCoreNum,
        inputGroupedValue);
    return std::make_tuple(statesGrouped, finalStateNp);
}

}  // namespace ascend_kernel
