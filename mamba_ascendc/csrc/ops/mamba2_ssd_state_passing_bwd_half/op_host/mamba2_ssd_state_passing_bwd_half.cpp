// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_state_passing_bwd_half.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_state_passing_bwd_half(
    const at::Tensor &statesStart,
    const at::Tensor &dStatesStart,
    const at::Tensor &dACumsum,
    const c10::optional<at::Tensor> &dfinalState)
{
    TORCH_CHECK(statesStart.device().type() == DEVICE_TYPE &&
                dStatesStart.device() == statesStart.device() &&
                dACumsum.device() == statesStart.device(),
                "mamba2_ssd_state_passing_bwd_half: inputs must be on one NPU");
    const bool stateIsHalf = statesStart.scalar_type() == at::kHalf;
    const bool dStateIsHalf = dStatesStart.scalar_type() == at::kHalf;
    TORCH_CHECK((stateIsHalf || statesStart.scalar_type() == at::kFloat) &&
                (dStateIsHalf || dStatesStart.scalar_type() == at::kFloat) &&
                dACumsum.scalar_type() == at::kFloat,
                "mamba2_ssd_state_passing_bwd_half: states_start and "
                "d_states_start must be float16/float32 and dA float32");
    TORCH_CHECK(statesStart.is_contiguous() &&
                dStatesStart.is_contiguous() && dACumsum.is_contiguous(),
                "mamba2_ssd_state_passing_bwd_half: inputs must be contiguous");
    const bool inputGrouped = statesStart.dim() == 6;
    TORCH_CHECK((statesStart.dim() == 5 || inputGrouped) &&
                dStatesStart.dim() == statesStart.dim() &&
                dACumsum.dim() == 4,
                "mamba2_ssd_state_passing_bwd_half: expected [B,H,K,P,N] "
                "or grouped [B,K,G,N,R,P], plus [B,H,K,T]");
    TORCH_CHECK(statesStart.sizes() == dStatesStart.sizes(),
                "mamba2_ssd_state_passing_bwd_half: state gradient shape mismatch");

    const int64_t batch = statesStart.size(0);
    const int64_t groups = inputGrouped ? statesStart.size(2) : 1;
    const int64_t headsPerGroup = inputGrouped ? statesStart.size(4) : 1;
    const int64_t nheads = inputGrouped
        ? groups * headsPerGroup : statesStart.size(1);
    const int64_t nchunks = inputGrouped
        ? statesStart.size(1) : statesStart.size(2);
    const int64_t headdim = inputGrouped
        ? statesStart.size(5) : statesStart.size(3);
    const int64_t dstate = inputGrouped
        ? statesStart.size(3) : statesStart.size(4);
    const int64_t chunkSize = dACumsum.size(3);
    TORCH_CHECK(batch > 0 && nheads > 0 && nchunks > 0,
                "mamba2_ssd_state_passing_bwd_half: B, H, and K must be positive");
    TORCH_CHECK(headdim == 64,
                "mamba2_ssd_state_passing_bwd_half: headdim must be 64");
    TORCH_CHECK(dstate == 64 || dstate == 128,
                "mamba2_ssd_state_passing_bwd_half: dstate must be 64 or 128");
    TORCH_CHECK(!inputGrouped ||
                (stateIsHalf && dStateIsHalf && headsPerGroup == 4 &&
                 dstate == 64 && headdim == 64),
                "mamba2_ssd_state_passing_bwd_half: grouped input requires "
                "FP16 [B,K,G,64,4,64]");
    TORCH_CHECK(chunkSize == 64 || chunkSize == 128,
                "mamba2_ssd_state_passing_bwd_half: chunk size must be 64 or 128");
    TORCH_CHECK(dACumsum.sizes() == at::IntArrayRef(
                    {batch, nheads, nchunks, chunkSize}),
                "mamba2_ssd_state_passing_bwd_half: dA_cumsum shape mismatch");

    if (dfinalState.has_value()) {
        const at::Tensor &dfinal = dfinalState.value();
        TORCH_CHECK(dfinal.device() == statesStart.device() &&
                    dfinal.scalar_type() == at::kFloat &&
                    dfinal.is_contiguous(),
                    "mamba2_ssd_state_passing_bwd_half: dfinal_state must be "
                    "contiguous FP32 NPU");
        TORCH_CHECK(dfinal.sizes() ==
                    at::IntArrayRef({batch, nheads, headdim, dstate}),
                    "mamba2_ssd_state_passing_bwd_half: dfinal_state shape mismatch");
    }

    at::Tensor dChunkStatesHalf = inputGrouped
        ? at::empty(
              {batch, nchunks, groups, dstate, headsPerGroup, headdim},
              dStatesStart.options().dtype(at::kHalf))
        : at::empty(
              {batch, nheads, nchunks, headdim, dstate},
              dStatesStart.options().dtype(at::kHalf));
    at::Tensor dinitialState = at::empty(
        {batch, nheads, headdim, dstate}, dACumsum.options());
    at::Tensor dAChunkLast = at::empty(
        {batch, nheads, nchunks}, dACumsum.options());
    at::Tensor dfinalArg =
        dfinalState.has_value() ? dfinalState.value() : dStatesStart;
    const int64_t hasDfinal = dfinalState.has_value() ? 1 : 0;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum =
        static_cast<int64_t>(platform->GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t stateBytes =
        static_cast<uint64_t>(headdim) * dstate * sizeof(float);
    const uint64_t halfStateBytes = stateBytes / 2;
    // The real heavy path keeps states_start in FP16.  For N=64 the kernel
    // adds two-deep state/dState prefetch queues so GM->UB can overlap the
    // current chunk's Vector work.  N=128 retains the serial path because
    // the same double buffers would exceed both 910B and 950 UB budgets.
    uint64_t requiredUb = 5 * stateBytes + halfStateBytes + 4096;
    if (stateIsHalf && dstate == 64) {
        requiredUb += 2 * halfStateBytes;
        requiredUb += dStateIsHalf ? 2 * halfStateBytes : 2 * stateBytes;
    }
    if (inputGrouped) {
        requiredUb += stateBytes;  // FP32 transpose gather offsets.
    }
    TORCH_CHECK(requiredUb <= ubSize,
                "mamba2_ssd_state_passing_bwd_half: state buffers exceed UB");

    const int64_t taskCount = batch * nheads;
    const int64_t usedCoreNum = std::min(taskCount, aivCoreNum);
    const int64_t stateIsHalfArg = stateIsHalf ? 1 : 0;
    const int64_t dStateIsHalfArg = dStateIsHalf ? 1 : 0;
    const int64_t inputGroupedArg = inputGrouped ? 1 : 0;
    TORCH_CHECK(usedCoreNum > 0,
                "mamba2_ssd_state_passing_bwd_half: no AIV cores");
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(mamba2_ssd_state_passing_bwd_half, blockDim,
                    statesStart, dStatesStart, dACumsum, dfinalArg,
                    dChunkStatesHalf, dinitialState, dAChunkLast,
                    batch, nheads, nchunks, headdim, dstate, chunkSize,
                    hasDfinal, usedCoreNum, stateIsHalfArg,
                    dStateIsHalfArg, groups, headsPerGroup,
                    inputGroupedArg);
    return std::make_tuple(
        dChunkStatesHalf, dinitialState, dAChunkLast);
}

}  // namespace ascend_kernel
