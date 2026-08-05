// Copyright (c) 2026, mamba-ascendc authors.

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_fwd.h"

namespace ascend_kernel {
namespace {

void CheckNpuFloatContiguous(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_fwd: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                "mamba2_ssd_fwd: ", name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(),
                "mamba2_ssd_fwd: ", name, " must be contiguous");
}

void CheckOptional(const c10::optional<at::Tensor> &tensor, const char *name)
{
    if (tensor.has_value()) {
        CheckNpuFloatContiguous(tensor.value(), name);
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_fwd(
    const at::Tensor &x,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const c10::optional<at::Tensor> &D,
    const c10::optional<at::Tensor> &z,
    const c10::optional<at::Tensor> &dtBias,
    const c10::optional<at::Tensor> &initialStates,
    int64_t chunkSize,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    CheckNpuFloatContiguous(x, "x");
    CheckNpuFloatContiguous(dt, "dt");
    CheckNpuFloatContiguous(A, "A");
    CheckNpuFloatContiguous(B, "B");
    CheckNpuFloatContiguous(C, "C");
    CheckOptional(D, "D");
    CheckOptional(z, "z");
    CheckOptional(dtBias, "dt_bias");
    CheckOptional(initialStates, "initial_states");

    TORCH_CHECK(x.dim() == 4, "mamba2_ssd_fwd: x must have shape [B,L,H,P]");
    TORCH_CHECK(dt.dim() == 3, "mamba2_ssd_fwd: dt must have shape [B,L,H]");
    TORCH_CHECK(A.dim() == 1, "mamba2_ssd_fwd: A must have shape [H]");
    TORCH_CHECK(B.dim() == 4 && C.dim() == 4,
                "mamba2_ssd_fwd: B and C must have shape [B,L,G,N]");

    const int64_t batch = x.size(0);
    const int64_t seqlen = x.size(1);
    const int64_t nheads = x.size(2);
    const int64_t headdim = x.size(3);
    const int64_t ngroups = B.size(2);
    const int64_t dstate = B.size(3);

    TORCH_CHECK(batch > 0 && seqlen > 0 && nheads > 0 && headdim > 0,
                "mamba2_ssd_fwd: zero-sized dimensions are not supported in V1");
    TORCH_CHECK(dstate > 0 && ngroups > 0,
                "mamba2_ssd_fwd: dstate and ngroups must be positive");
    TORCH_CHECK(dt.sizes() == at::IntArrayRef({batch, seqlen, nheads}),
                "mamba2_ssd_fwd: dt shape must equal [B,L,H]");
    TORCH_CHECK(A.size(0) == nheads, "mamba2_ssd_fwd: A shape must equal [H]");
    TORCH_CHECK(B.sizes() == C.sizes(), "mamba2_ssd_fwd: B and C shapes must match");
    TORCH_CHECK(B.size(0) == batch && B.size(1) == seqlen,
                "mamba2_ssd_fwd: B/C batch and sequence dimensions must match x");
    TORCH_CHECK(nheads % ngroups == 0,
                "mamba2_ssd_fwd: nheads must be divisible by ngroups");
    TORCH_CHECK(chunkSize > 0, "mamba2_ssd_fwd: chunk_size must be positive");
    TORCH_CHECK(dtLimitMin <= dtLimitMax,
                "mamba2_ssd_fwd: dt_limit_min must not exceed dt_limit_max");

    int64_t dHasHdim = 0;
    if (D.has_value()) {
        const at::Tensor &d = D.value();
        TORCH_CHECK(d.dim() == 1 || d.dim() == 2,
                    "mamba2_ssd_fwd: D must have shape [H] or [H,P]");
        TORCH_CHECK((d.dim() == 1 && d.size(0) == nheads) ||
                    (d.dim() == 2 && d.size(0) == nheads && d.size(1) == headdim),
                    "mamba2_ssd_fwd: invalid D shape");
        dHasHdim = d.dim() == 2 ? 1 : 0;
    }
    if (z.has_value()) {
        TORCH_CHECK(z.value().sizes() == x.sizes(),
                    "mamba2_ssd_fwd: z shape must match x");
    }
    if (dtBias.has_value()) {
        TORCH_CHECK(dtBias.value().dim() == 1 && dtBias.value().size(0) == nheads,
                    "mamba2_ssd_fwd: dt_bias shape must equal [H]");
    }
    if (initialStates.has_value()) {
        const at::Tensor &initial = initialStates.value();
        TORCH_CHECK(initial.dim() == 4 && initial.size(0) == batch &&
                    initial.size(1) == nheads && initial.size(2) == headdim &&
                    initial.size(3) == dstate,
                    "mamba2_ssd_fwd: initial_states shape must equal [B,H,P,N]");
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    TORCH_CHECK(coreNum > 0, "mamba2_ssd_fwd: platform reports no available AIV cores");
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    constexpr uint64_t kUbReserveBytes = 16 * 1024;
    TORCH_CHECK(ubSize > kUbReserveBytes,
                "mamba2_ssd_fwd: platform UB size is below the required reserve");
    const uint64_t availableUbBytes = ubSize - kUbReserveBytes;
    const uint64_t headdimU64 = static_cast<uint64_t>(headdim);
    const uint64_t dstateU64 = static_cast<uint64_t>(dstate);
    TORCH_CHECK(headdimU64 <= availableUbBytes / sizeof(float) / dstateU64,
                "mamba2_ssd_fwd: FP32 state size overflows or exceeds the safe UB budget; "
                "reduce headdim*dstate or use the future tiled-state path");
    const auto align32 = [](uint64_t bytes) {
        return (bytes + 31U) / 32U * 32U;
    };
    const uint64_t stateBytes = headdimU64 * dstateU64 * sizeof(float);
    const uint64_t outRowBytes = headdimU64 * sizeof(float);
    const uint64_t requiredUbBytes = align32(stateBytes) + align32(outRowBytes);
    TORCH_CHECK(requiredUbBytes <= availableUbBytes,
                "mamba2_ssd_fwd: FP32 state and output row do not fit the safe UB budget; "
                "reduce headdim*dstate or use the future tiled-state path");

    at::Tensor out = at::empty_like(x);
    at::Tensor finalState = at::empty({batch, nheads, headdim, dstate}, x.options());

    at::Tensor dArg = D.has_value() ? D.value() : x;
    at::Tensor zArg = z.has_value() ? z.value() : x;
    at::Tensor dtBiasArg = dtBias.has_value() ? dtBias.value() : A;
    at::Tensor initialArg = initialStates.has_value() ? initialStates.value() : x;

    int64_t hasD = D.has_value() ? 1 : 0;
    int64_t hasZ = z.has_value() ? 1 : 0;
    int64_t hasDtBias = dtBias.has_value() ? 1 : 0;
    int64_t hasInitialState = initialStates.has_value() ? 1 : 0;
    int64_t dtSoftplusFlag = dtSoftplus ? 1 : 0;
    float dtMin = static_cast<float>(dtLimitMin);
    float dtMax = static_cast<float>(dtLimitMax);
    // Independent (batch, head) tasks can safely run on separate AIV cores once
    // an output row is staged in UB and copied to GM as an aligned block. Keep
    // the proven single-core path for non-32-byte-aligned rows in this round.
    const bool outputRowAligned = (headdim * static_cast<int64_t>(sizeof(float))) % 32 == 0;
    int64_t usedCoreNum = outputRowAligned ?
        std::min(batch * nheads, coreNum) : 1;
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(mamba2_ssd_fwd, blockDim,
                    x, dt, A, B, C, dArg, zArg, dtBiasArg, initialArg,
                    out, finalState,
                    batch, seqlen, nheads, headdim, dstate, ngroups,
                    hasD, dHasHdim, hasZ, hasDtBias, hasInitialState,
                    dtSoftplusFlag, dtMin, dtMax, usedCoreNum);

    return std::make_tuple(out, finalState);
}

}  // namespace ascend_kernel
