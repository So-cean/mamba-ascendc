// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_dt_bwd.h"
#include "aclrtlaunch_mamba2_ssd_dt_bwd_d.h"
#include "aclrtlaunch_mamba2_ssd_dt_bwd_grouped_d.h"
#include "aclrtlaunch_mamba2_ssd_dt_bwd_reduce.h"
#include "aclrtlaunch_mamba2_ssd_bwd_gate_reduce.h"

namespace ascend_kernel {

namespace {
void CheckFp32Npu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_dt_bwd: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.is_contiguous(),
                "mamba2_ssd_dt_bwd: ", name,
                " must be contiguous float32");
}

void CheckFp16Npu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_dt_bwd: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf && tensor.is_contiguous(),
                "mamba2_ssd_dt_bwd: ", name,
                " must be contiguous float16");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_dt_bwd(
    const at::Tensor &x,
    const at::Tensor &dXdt,
    const at::Tensor &gCs,
    const at::Tensor &dt,
    const at::Tensor &A,
    const c10::optional<at::Tensor> &dtBias,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    CheckFp32Npu(x, "x");
    CheckFp32Npu(dXdt, "d_xdt_total");
    CheckFp32Npu(gCs, "g_dA_cs_total");
    CheckFp32Npu(dt, "dt");
    CheckFp32Npu(A, "A");
    TORCH_CHECK(x.dim() == 4 && dXdt.dim() == 5 && gCs.dim() == 4 &&
                dt.dim() == 3 && A.dim() == 1,
                "mamba2_ssd_dt_bwd: invalid input rank");
    const int64_t batch = dXdt.size(0);
    const int64_t nheads = dXdt.size(1);
    const int64_t nchunks = dXdt.size(2);
    const int64_t chunkSize = dXdt.size(3);
    const int64_t headdim = dXdt.size(4);
    const int64_t seqlen = nchunks * chunkSize;
    TORCH_CHECK(batch > 0 && nheads > 0 && nchunks > 0,
                "mamba2_ssd_dt_bwd: B/H/K must be positive");
    TORCH_CHECK(chunkSize == 64 || chunkSize == 128,
                "mamba2_ssd_dt_bwd: chunk size must be 64 or 128");
    TORCH_CHECK(headdim == 64,
                "mamba2_ssd_dt_bwd: headdim must be 64");
    TORCH_CHECK(x.sizes() == at::IntArrayRef({batch, seqlen, nheads, headdim}),
                "mamba2_ssd_dt_bwd: x shape mismatch");
    TORCH_CHECK(gCs.sizes() ==
                    at::IntArrayRef({batch, nheads, nchunks, chunkSize}),
                "mamba2_ssd_dt_bwd: g_dA_cs_total shape mismatch");
    TORCH_CHECK(dt.sizes() == at::IntArrayRef({batch, seqlen, nheads}),
                "mamba2_ssd_dt_bwd: dt shape mismatch");
    TORCH_CHECK(A.numel() == nheads,
                "mamba2_ssd_dt_bwd: A shape mismatch");
    TORCH_CHECK(dtLimitMin <= dtLimitMax,
                "mamba2_ssd_dt_bwd: invalid dt limits");
    if (dtBias.has_value()) {
        CheckFp32Npu(dtBias.value(), "dt_bias");
        TORCH_CHECK(dtBias.value().dim() == 1 &&
                    dtBias.value().numel() == nheads,
                    "mamba2_ssd_dt_bwd: dt_bias shape mismatch");
    }

    at::Tensor dx = at::empty_like(x);
    at::Tensor ddt = at::empty_like(dt);
    at::Tensor dAPartial = at::empty({nheads, batch, nchunks}, dt.options());
    at::Tensor dBiasPartial = at::empty_like(dAPartial);
    at::Tensor dA = at::empty_like(A);
    at::Tensor dBias = at::empty_like(A);
    at::Tensor biasArg = dtBias.has_value() ? dtBias.value() : A;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t chunkTasks = batch * nheads * nchunks;
    const int64_t phase0Cores = std::min(chunkTasks, aivCoreNum);
    const int64_t phase1Cores = std::min(nheads, aivCoreNum);
    TORCH_CHECK(phase0Cores > 0 && phase1Cores > 0,
                "mamba2_ssd_dt_bwd: no AIV cores");
    const int64_t hasBias = dtBias.has_value() ? 1 : 0;
    const int64_t softplus = dtSoftplus ? 1 : 0;
    const float limitMin = static_cast<float>(dtLimitMin);
    const float limitMax = static_cast<float>(dtLimitMax);
    uint32_t blockDim0 = static_cast<uint32_t>(phase0Cores);
    uint32_t blockDim1 = static_cast<uint32_t>(phase1Cores);

    EXEC_KERNEL_CMD(
        mamba2_ssd_dt_bwd, blockDim0,
        x, dXdt, gCs, dt, A, biasArg, dx, ddt, dAPartial, dBiasPartial,
        batch, seqlen, nheads, nchunks, chunkSize, headdim,
        hasBias, softplus, limitMin, limitMax, phase0Cores);
    EXEC_KERNEL_CMD(
        mamba2_ssd_dt_bwd_reduce, blockDim1,
        dAPartial, dBiasPartial, dA, dBias,
        batch, nheads, nchunks, phase1Cores);
    return std::make_tuple(dx, ddt, dA, dBias);
}

namespace {
using DtBwdDResult = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

DtBwdDResult Mamba2SsdDtBwdDImpl(
    const at::Tensor &x,
    const at::Tensor &dXdt,
    const at::Tensor &gCs,
    const at::Tensor *gCsOff,
    const at::Tensor *gCsLast,
    int64_t ngroups,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gyHead,
    const at::Tensor &dMatrix,
    const c10::optional<at::Tensor> &dtBias,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax,
    int64_t gCsMode,
    bool computeDD)
{
    CheckFp32Npu(x, "x");
    const bool dXdtIsHalf = dXdt.scalar_type() == at::kHalf;
    if (dXdtIsHalf) {
        CheckFp16Npu(dXdt, "d_xdt_total");
    } else {
        CheckFp32Npu(dXdt, "d_xdt_total");
    }
    CheckFp32Npu(gCs, "g_dA_cs_total");
    const bool hasExtraGcs = gCsMode != 0;
    TORCH_CHECK(hasExtraGcs == (gCsOff != nullptr && gCsLast != nullptr),
                "mamba2_ssd_dt_bwd_d: inconsistent fused g_cs inputs");
    if (hasExtraGcs) {
        CheckFp32Npu(*gCsOff, "g_dA_cs_off");
        CheckFp32Npu(*gCsLast, "g_dA_cs_chunk_last");
    }
    CheckFp32Npu(dt, "dt");
    CheckFp32Npu(A, "A");
    CheckFp16Npu(gyHead, "gy_head");
    CheckFp32Npu(dMatrix, "D");
    TORCH_CHECK(x.dim() == 4 && dXdt.dim() == 5 && gCs.dim() == 4 &&
                dt.dim() == 3 && A.dim() == 1 && gyHead.dim() == 5 &&
                dMatrix.dim() == 2,
                "mamba2_ssd_dt_bwd_d: invalid input rank");
    const int64_t batch = dXdt.size(0);
    const int64_t nheads = dXdt.size(1);
    const int64_t nchunks = dXdt.size(2);
    const int64_t chunkSize = dXdt.size(3);
    const int64_t headdim = dXdt.size(4);
    const int64_t seqlen = nchunks * chunkSize;
    TORCH_CHECK(batch > 0 && nheads > 0 && nchunks > 0,
                "mamba2_ssd_dt_bwd_d: B/H/K must be positive");
    TORCH_CHECK(chunkSize == 64 || chunkSize == 128,
                "mamba2_ssd_dt_bwd_d: chunk size must be 64 or 128");
    TORCH_CHECK(headdim == 64,
                "mamba2_ssd_dt_bwd_d: headdim must be 64");
    TORCH_CHECK(x.sizes() == at::IntArrayRef({batch, seqlen, nheads, headdim}),
                "mamba2_ssd_dt_bwd_d: x shape mismatch");
    TORCH_CHECK(gyHead.sizes() == dXdt.sizes(),
                "mamba2_ssd_dt_bwd_d: gy_head shape mismatch");
    TORCH_CHECK(gCs.sizes() ==
                    at::IntArrayRef({batch, nheads, nchunks, chunkSize}),
                "mamba2_ssd_dt_bwd_d: g_dA_cs_total shape mismatch");
    const int64_t headsPerGroup = gCsMode == 1 ? nheads / ngroups : 1;
    if (gCsMode == 1) {
        TORCH_CHECK(ngroups > 0 && nheads % ngroups == 0 &&
                    headsPerGroup == 4,
                    "mamba2_ssd_dt_bwd_grouped_d: expected four heads per group");
        TORCH_CHECK(gCsOff->sizes() == at::IntArrayRef(
                        {batch, nchunks, ngroups, chunkSize, headsPerGroup}),
                    "mamba2_ssd_dt_bwd_grouped_d: grouped Off shape mismatch");
        TORCH_CHECK(gCsLast->sizes() ==
                        at::IntArrayRef({batch, nheads, nchunks}),
                    "mamba2_ssd_dt_bwd_grouped_d: chunk-last shape mismatch");
    } else if (gCsMode == 2) {
        TORCH_CHECK(gCsOff->sizes() == gCs.sizes(),
                    "mamba2_ssd_dt_bwd_fused_gcs_d: Off shape mismatch");
        TORCH_CHECK(gCsLast->sizes() ==
                        at::IntArrayRef({batch, nheads, nchunks}),
                    "mamba2_ssd_dt_bwd_fused_gcs_d: chunk-last shape mismatch");
    } else if (gCsMode == 3) {
        TORCH_CHECK(gCsLast->sizes() ==
                        at::IntArrayRef({batch, nheads, nchunks}),
                    "mamba2_ssd_dt_bwd_chunk_tail_d: chunk-last shape mismatch");
    }
    TORCH_CHECK(dt.sizes() == at::IntArrayRef({batch, seqlen, nheads}),
                "mamba2_ssd_dt_bwd_d: dt shape mismatch");
    TORCH_CHECK(A.numel() == nheads &&
                dMatrix.sizes() == at::IntArrayRef({nheads, headdim}),
                "mamba2_ssd_dt_bwd_d: A/D shape mismatch");
    TORCH_CHECK(dtLimitMin <= dtLimitMax,
                "mamba2_ssd_dt_bwd_d: invalid dt limits");
    if (dtBias.has_value()) {
        CheckFp32Npu(dtBias.value(), "dt_bias");
        TORCH_CHECK(dtBias.value().dim() == 1 &&
                    dtBias.value().numel() == nheads,
                    "mamba2_ssd_dt_bwd_d: dt_bias shape mismatch");
    }

    at::Tensor dx = at::empty_like(x);
    at::Tensor ddt = at::empty_like(dt);
    at::Tensor dAPartial = at::empty({nheads, batch, nchunks}, dt.options());
    at::Tensor dBiasPartial = at::empty_like(dAPartial);
    at::Tensor dA = at::empty_like(A);
    at::Tensor dBias = at::empty_like(A);
    at::Tensor dDPartial = computeDD
        ? at::empty({nheads, batch, nchunks, headdim}, dt.options())
        : at::empty({1}, dt.options());
    at::Tensor dD = computeDD
        ? at::empty({nheads, headdim}, dt.options())
        : at::empty({0}, dt.options());
    at::Tensor biasArg = dtBias.has_value() ? dtBias.value() : A;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    // T64 D-backward is dominated by public [B,L,H,P] strided x/dx DMA at
    // large H.  Pair adjacent heads so one task owns wider public-layout
    // bursts and retains A/bias/D while walking all K chunks.  T128 keeps the
    // verified single-head path because its larger UB working set leaves no
    // safe room for the extra staging buffers.
    const int64_t headBlock = (chunkSize == 64 && nheads >= 2) ? 2 : 1;
    const int64_t headBlockCount = (nheads + headBlock - 1) / headBlock;
    const int64_t chunkTasks = headBlock == 2
        ? batch * headBlockCount
        : batch * nheads * nchunks;
    const int64_t phase0Cores = std::min(chunkTasks, aivCoreNum);
    const int64_t phase1Cores = std::min(nheads, aivCoreNum);
    TORCH_CHECK(!computeDD || headBlock == 2,
                "mamba2_ssd_dt_bwd_d_dd requires T=64 and H>=2");
    TORCH_CHECK(phase0Cores > 0 && phase1Cores > 0,
                "mamba2_ssd_dt_bwd_d: no AIV cores");
    const int64_t hasBias = dtBias.has_value() ? 1 : 0;
    const int64_t softplus = dtSoftplus ? 1 : 0;
    const float limitMin = static_cast<float>(dtLimitMin);
    const float limitMax = static_cast<float>(dtLimitMax);
    const uint32_t blockDim0 = static_cast<uint32_t>(phase0Cores);
    const uint32_t blockDim1 = static_cast<uint32_t>(phase1Cores);
    const int64_t computeDDArg = computeDD ? 1 : 0;
    const int64_t dXdtIsHalfArg = dXdtIsHalf ? 1 : 0;

    if (hasExtraGcs) {
        const at::Tensor &gCsOffArg = *gCsOff;
        const at::Tensor &gCsLastArg = *gCsLast;
        EXEC_KERNEL_CMD(
            mamba2_ssd_dt_bwd_grouped_d, blockDim0,
            x, dXdt, gCs, gCsOffArg, gCsLastArg, dt, A, biasArg,
            gyHead, dMatrix, dx, ddt, dAPartial, dBiasPartial, dDPartial,
            batch, seqlen, nheads, nchunks, chunkSize, headdim,
            hasBias, softplus, limitMin, limitMax, phase0Cores, headBlock,
            computeDDArg, ngroups, headsPerGroup, gCsMode,
            dXdtIsHalfArg);
    } else {
        EXEC_KERNEL_CMD(
            mamba2_ssd_dt_bwd_d, blockDim0,
            x, dXdt, gCs, dt, A, biasArg, gyHead, dMatrix,
            dx, ddt, dAPartial, dBiasPartial, dDPartial,
            batch, seqlen, nheads, nchunks, chunkSize, headdim,
            hasBias, softplus, limitMin, limitMax, phase0Cores, headBlock,
            computeDDArg, dXdtIsHalfArg);
    }
    EXEC_KERNEL_CMD(
        mamba2_ssd_dt_bwd_reduce, blockDim1,
        dAPartial, dBiasPartial, dA, dBias,
        batch, nheads, nchunks, phase1Cores);
    if (computeDD) {
        EXEC_KERNEL_CMD(
            mamba2_ssd_bwd_gate_reduce, blockDim1,
            dDPartial, dD, batch, nheads, nchunks, phase1Cores);
    }
    return std::make_tuple(dx, ddt, dA, dBias, dD);
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_dt_bwd_d(
    const at::Tensor &x,
    const at::Tensor &dXdt,
    const at::Tensor &gCs,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gyHead,
    const at::Tensor &dMatrix,
    const c10::optional<at::Tensor> &dtBias,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    auto result = Mamba2SsdDtBwdDImpl(
        x, dXdt, gCs, nullptr, nullptr, 1, dt, A, gyHead, dMatrix, dtBias,
        dtSoftplus, dtLimitMin, dtLimitMax, 0, false);
    return std::make_tuple(
        std::get<0>(result), std::get<1>(result),
        std::get<2>(result), std::get<3>(result));
}

DtBwdDResult mamba2_ssd_dt_bwd_d_dd(
    const at::Tensor &x,
    const at::Tensor &dXdt,
    const at::Tensor &gCs,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gyHead,
    const at::Tensor &dMatrix,
    const c10::optional<at::Tensor> &dtBias,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    return Mamba2SsdDtBwdDImpl(
        x, dXdt, gCs, nullptr, nullptr, 1, dt, A, gyHead, dMatrix, dtBias,
        dtSoftplus, dtLimitMin, dtLimitMax, 0, true);
}

DtBwdDResult mamba2_ssd_dt_bwd_grouped_d_dd(
    const at::Tensor &x,
    const at::Tensor &dXdt,
    const at::Tensor &gCsDiag,
    const at::Tensor &gCsOffGrouped,
    const at::Tensor &gCsChunkLast,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gyHead,
    const at::Tensor &dMatrix,
    const c10::optional<at::Tensor> &dtBias,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    const int64_t ngroups = gCsOffGrouped.size(2);
    return Mamba2SsdDtBwdDImpl(
        x, dXdt, gCsDiag, &gCsOffGrouped, &gCsChunkLast, ngroups,
        dt, A, gyHead, dMatrix, dtBias,
        dtSoftplus, dtLimitMin, dtLimitMax, 1, true);
}

DtBwdDResult mamba2_ssd_dt_bwd_fused_gcs_d_dd(
    const at::Tensor &x,
    const at::Tensor &dXdt,
    const at::Tensor &gCsDiag,
    const at::Tensor &gCsOff,
    const at::Tensor &gCsChunkLast,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gyHead,
    const at::Tensor &dMatrix,
    const c10::optional<at::Tensor> &dtBias,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    return Mamba2SsdDtBwdDImpl(
        x, dXdt, gCsDiag, &gCsOff, &gCsChunkLast, 1,
        dt, A, gyHead, dMatrix, dtBias,
        dtSoftplus, dtLimitMin, dtLimitMax, 2, true);
}

DtBwdDResult mamba2_ssd_dt_bwd_chunk_tail_d_dd(
    const at::Tensor &x,
    const at::Tensor &dXdt,
    const at::Tensor &gCs,
    const at::Tensor &gCsChunkLast,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gyHead,
    const at::Tensor &dMatrix,
    const c10::optional<at::Tensor> &dtBias,
    bool dtSoftplus,
    double dtLimitMin,
    double dtLimitMax)
{
    // gCs is also passed as the unused auxiliary pointer so the shared
    // launch path can carry gCsChunkLast without allocating another tensor.
    return Mamba2SsdDtBwdDImpl(
        x, dXdt, gCs, &gCs, &gCsChunkLast, 1,
        dt, A, gyHead, dMatrix, dtBias,
        dtSoftplus, dtLimitMin, dtLimitMax, 3, true);
}

}  // namespace ascend_kernel
