// Copyright (c) 2026, mamba-ascendc authors.

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_state_passing.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_state_passing(
    const at::Tensor &chunkStates,
    const at::Tensor &dACumsum,
    const c10::optional<at::Tensor> &initialStates)
{
    TORCH_CHECK(chunkStates.device().type() == DEVICE_TYPE &&
                dACumsum.device().type() == DEVICE_TYPE,
                "mamba2_ssd_state_passing: inputs must be NPU tensors");
    TORCH_CHECK(chunkStates.scalar_type() == at::kFloat &&
                dACumsum.scalar_type() == at::kFloat,
                "mamba2_ssd_state_passing: inputs must be float32");
    TORCH_CHECK(chunkStates.is_contiguous() && dACumsum.is_contiguous(),
                "mamba2_ssd_state_passing: inputs must be contiguous");
    TORCH_CHECK(chunkStates.dim() == 5 && dACumsum.dim() == 4,
                "mamba2_ssd_state_passing: expected [B,H,C,P,N] and [B,H,C,T]");
    const int64_t batch = chunkStates.size(0);
    const int64_t nheads = chunkStates.size(1);
    const int64_t nchunks = chunkStates.size(2);
    const int64_t headdim = chunkStates.size(3);
    const int64_t dstate = chunkStates.size(4);
    const int64_t chunkSize = dACumsum.size(3);
    TORCH_CHECK(dACumsum.size(0) == batch && dACumsum.size(1) == nheads &&
                dACumsum.size(2) == nchunks,
                "mamba2_ssd_state_passing: dA_cumsum shape mismatch");
    TORCH_CHECK(dstate > 0 && dstate % 8 == 0,
                "mamba2_ssd_state_passing: dstate must be a positive multiple of 8");

    if (initialStates.has_value()) {
        const at::Tensor &initial = initialStates.value();
        TORCH_CHECK(initial.device().type() == DEVICE_TYPE &&
                    initial.scalar_type() == at::kFloat && initial.is_contiguous(),
                    "mamba2_ssd_state_passing: initial_states must be contiguous FP32 NPU");
        TORCH_CHECK(initial.sizes() ==
                    at::IntArrayRef({batch, nheads, headdim, dstate}),
                    "mamba2_ssd_state_passing: initial_states shape mismatch");
    }

    at::Tensor statesStart = at::empty_like(chunkStates);
    at::Tensor finalState = at::empty(
        {batch, nheads, headdim, dstate}, chunkStates.options());
    at::Tensor initialArg = initialStates.has_value() ? initialStates.value() : chunkStates;
    int64_t hasInitial = initialStates.has_value() ? 1 : 0;
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t stateBytes = static_cast<uint64_t>(headdim) * dstate * sizeof(float);
    TORCH_CHECK(3 * stateBytes + 16 * 1024 <= ubSize,
                "mamba2_ssd_state_passing: three full-state buffers exceed UB");
    const int64_t headTaskCount = batch * nheads;
    int64_t usedCoreNum = std::min(headTaskCount, aivCoreNum);
    TORCH_CHECK(usedCoreNum > 0, "mamba2_ssd_state_passing: no AIV cores");
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(mamba2_ssd_state_passing, blockDim,
                    chunkStates, dACumsum, initialArg, statesStart, finalState,
                    batch, nheads, nchunks, headdim, dstate, chunkSize,
                    hasInitial, usedCoreNum);
    return std::make_tuple(statesStart, finalState);
}

}  // namespace ascend_kernel
