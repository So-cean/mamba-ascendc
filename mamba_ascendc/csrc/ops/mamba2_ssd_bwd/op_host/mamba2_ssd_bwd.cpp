// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <limits>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_bwd.h"

namespace ascend_kernel {

namespace {

void CheckFp32NpuContiguous(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_bwd: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                "mamba2_ssd_bwd: ", name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(),
                "mamba2_ssd_bwd: ", name, " must be contiguous");
}

}  // namespace

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_bwd(
    const at::Tensor &x,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &dout,
    const at::Tensor &finalState,
    const c10::optional<at::Tensor> &D,
    const c10::optional<at::Tensor> &z,
    const c10::optional<at::Tensor> &dtBias,
    const c10::optional<at::Tensor> &initialStates,
    const c10::optional<at::Tensor> &dfinalState,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    CheckFp32NpuContiguous(x, "x");
    CheckFp32NpuContiguous(dt, "dt");
    CheckFp32NpuContiguous(A, "A");
    CheckFp32NpuContiguous(B, "B");
    CheckFp32NpuContiguous(C, "C");
    CheckFp32NpuContiguous(dout, "dout");
    CheckFp32NpuContiguous(finalState, "final_state");

    TORCH_CHECK(x.dim() == 4 && dt.dim() == 3 && A.dim() == 1 &&
                    B.dim() == 4 && C.dim() == 4 && dout.dim() == 4 &&
                    finalState.dim() == 4,
                "mamba2_ssd_bwd: invalid input rank");

    const int64_t batch = x.size(0);
    const int64_t seqlen = x.size(1);
    const int64_t nheads = x.size(2);
    const int64_t headdim = x.size(3);
    const int64_t ngroups = B.size(2);
    const int64_t dstate = B.size(3);
    TORCH_CHECK(dt.sizes() == at::IntArrayRef({batch, seqlen, nheads}),
                "mamba2_ssd_bwd: dt shape mismatch");
    TORCH_CHECK(A.numel() == nheads, "mamba2_ssd_bwd: A shape mismatch");
    TORCH_CHECK(B.sizes() == C.sizes() &&
                    B.size(0) == batch && B.size(1) == seqlen,
                "mamba2_ssd_bwd: B/C shape mismatch");
    TORCH_CHECK(dout.sizes() == x.sizes(),
                "mamba2_ssd_bwd: dout shape mismatch");
    TORCH_CHECK(finalState.sizes() ==
                    at::IntArrayRef({batch, nheads, headdim, dstate}),
                "mamba2_ssd_bwd: final_state shape mismatch");
    TORCH_CHECK(nheads % ngroups == 0,
                "mamba2_ssd_bwd: nheads must be divisible by ngroups");

    // M0 is deliberately narrow: it establishes a native autograd correctness
    // loop before the chunk/Cube backward replaces this direct recurrence.
    TORCH_CHECK(headdim == 64 && dstate == 64 && seqlen % 64 == 0,
                "mamba2_ssd_bwd M0: requires P=N=64 and L divisible by 64");
    TORCH_CHECK(!D.has_value() && !z.has_value() && !dtBias.has_value() &&
                    !initialStates.has_value(),
                "mamba2_ssd_bwd M0: optional D/z/dt_bias/initial_states "
                "are not implemented yet");
    TORCH_CHECK(!dtSoftplus,
                "mamba2_ssd_bwd M0: dt_softplus must be false");
    TORCH_CHECK(dtLimitMin == 0.0 &&
                    dtLimitMax >= static_cast<double>(1.0e30f),
                "mamba2_ssd_bwd M0: only the default dt_limit is supported");

    if (dfinalState.has_value()) {
        CheckFp32NpuContiguous(dfinalState.value(), "dfinal_state");
        TORCH_CHECK(dfinalState.value().sizes() == finalState.sizes(),
                    "mamba2_ssd_bwd: dfinal_state shape mismatch");
    }

    at::Tensor dx = at::empty_like(x);
    at::Tensor ddt = at::empty_like(dt);
    at::Tensor dAPartial = at::empty({batch, nheads}, x.options());
    at::Tensor dB = at::empty_like(B);
    at::Tensor dC = at::empty_like(C);
    at::Tensor empty = at::empty({0}, x.options());
    at::Tensor dfinalArg = dfinalState.has_value() ? dfinalState.value() : finalState;
    int64_t hasDfinal = dfinalState.has_value() ? 1 : 0;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t stateBytes =
        static_cast<uint64_t>(headdim) * dstate * sizeof(float);
    const uint64_t vectorBytes =
        static_cast<uint64_t>(8 * headdim + 4 * dstate + 32) * sizeof(float);
    TORCH_CHECK(3 * stateBytes + vectorBytes <= ubSize,
                "mamba2_ssd_bwd M0: UB allocation exceeds device capacity");
    const int64_t taskCount = batch * ngroups;
    const int64_t usedCoreNum = std::min(taskCount, aivCoreNum);
    TORCH_CHECK(usedCoreNum > 0, "mamba2_ssd_bwd: no AIV cores");
    at::Tensor statesWork = at::empty(
        {usedCoreNum, seqlen, headdim, dstate}, x.options());
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(
        mamba2_ssd_bwd, blockDim,
        x, dt, A, B, C, dout, finalState, statesWork, dfinalArg,
        dx, ddt, dAPartial, dB, dC,
        batch, seqlen, nheads, headdim, dstate, ngroups,
        hasDfinal, usedCoreNum);

    return std::make_tuple(dx, ddt, dAPartial, dB, dC, empty, empty, empty);
}

}  // namespace ascend_kernel
