// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_state_passing_bwd.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_state_passing_bwd(
    const at::Tensor &statesStart,
    const at::Tensor &dStatesStart,
    const at::Tensor &dACumsum,
    const c10::optional<at::Tensor> &dfinalState)
{
    TORCH_CHECK(statesStart.device().type() == DEVICE_TYPE &&
                dStatesStart.device().type() == DEVICE_TYPE &&
                dACumsum.device().type() == DEVICE_TYPE,
                "mamba2_ssd_state_passing_bwd: inputs must be NPU tensors");
    TORCH_CHECK(statesStart.scalar_type() == at::kFloat &&
                dStatesStart.scalar_type() == at::kFloat &&
                dACumsum.scalar_type() == at::kFloat,
                "mamba2_ssd_state_passing_bwd: inputs must be float32");
    TORCH_CHECK(statesStart.is_contiguous() &&
                dStatesStart.is_contiguous() && dACumsum.is_contiguous(),
                "mamba2_ssd_state_passing_bwd: inputs must be contiguous");
    TORCH_CHECK(statesStart.dim() == 5 && dStatesStart.dim() == 5 &&
                dACumsum.dim() == 4,
                "mamba2_ssd_state_passing_bwd: expected [B,H,K,P,N], "
                "[B,H,K,P,N], and [B,H,K,T]");
    TORCH_CHECK(statesStart.sizes() == dStatesStart.sizes(),
                "mamba2_ssd_state_passing_bwd: state gradient shape mismatch");

    const int64_t batch = statesStart.size(0);
    const int64_t nheads = statesStart.size(1);
    const int64_t nchunks = statesStart.size(2);
    const int64_t headdim = statesStart.size(3);
    const int64_t dstate = statesStart.size(4);
    const int64_t chunkSize = dACumsum.size(3);
    TORCH_CHECK(batch > 0 && nheads > 0 && nchunks > 0,
                "mamba2_ssd_state_passing_bwd: B, H, and K must be positive");
    TORCH_CHECK(headdim == 64,
                "mamba2_ssd_state_passing_bwd: headdim must be 64");
    TORCH_CHECK(dstate == 64 || dstate == 128,
                "mamba2_ssd_state_passing_bwd: dstate must be 64 or 128");
    TORCH_CHECK(chunkSize == 64 || chunkSize == 128,
                "mamba2_ssd_state_passing_bwd: chunk size must be 64 or 128");
    TORCH_CHECK(dACumsum.size(0) == batch &&
                dACumsum.size(1) == nheads &&
                dACumsum.size(2) == nchunks,
                "mamba2_ssd_state_passing_bwd: dA_cumsum shape mismatch");

    if (dfinalState.has_value()) {
        const at::Tensor &dfinal = dfinalState.value();
        TORCH_CHECK(dfinal.device().type() == DEVICE_TYPE &&
                    dfinal.scalar_type() == at::kFloat &&
                    dfinal.is_contiguous(),
                    "mamba2_ssd_state_passing_bwd: dfinal_state must be "
                    "contiguous FP32 NPU");
        TORCH_CHECK(dfinal.sizes() ==
                    at::IntArrayRef({batch, nheads, headdim, dstate}),
                    "mamba2_ssd_state_passing_bwd: dfinal_state shape mismatch");
    }

    at::Tensor dChunkStates = at::empty_like(statesStart);
    at::Tensor dinitialState = at::empty(
        {batch, nheads, headdim, dstate}, statesStart.options());
    at::Tensor dAChunkLast = at::empty(
        {batch, nheads, nchunks}, statesStart.options());
    at::Tensor dfinalArg =
        dfinalState.has_value() ? dfinalState.value() : statesStart;
    int64_t hasDfinal = dfinalState.has_value() ? 1 : 0;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum =
        static_cast<int64_t>(platform->GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t stateBytes =
        static_cast<uint64_t>(headdim) * dstate * sizeof(float);
    TORCH_CHECK(5 * stateBytes + 4096 <= ubSize,
                "mamba2_ssd_state_passing_bwd: state buffers exceed UB");

    const int64_t taskCount = batch * nheads;
    int64_t usedCoreNum = std::min(taskCount, aivCoreNum);
    TORCH_CHECK(usedCoreNum > 0,
                "mamba2_ssd_state_passing_bwd: no AIV cores");
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(mamba2_ssd_state_passing_bwd, blockDim,
                    statesStart, dStatesStart, dACumsum, dfinalArg,
                    dChunkStates, dinitialState, dAChunkLast,
                    batch, nheads, nchunks, headdim, dstate, chunkSize,
                    hasDfinal, usedCoreNum);
    return std::make_tuple(dChunkStates, dinitialState, dAChunkLast);
}

}  // namespace ascend_kernel
