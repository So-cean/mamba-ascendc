// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_bwd_diag_finalize.h"

namespace ascend_kernel {
namespace {

constexpr int64_t kTile = 64;
constexpr int64_t kMaxHeadBlock = 4;

uint64_t RequiredUbBytes(int64_t headBlockSize)
{
    constexpr uint64_t kMatrixHalfBytes = kTile * kTile * sizeof(uint16_t);
    constexpr uint64_t kMatrixFloatBytes = kTile * kTile * sizeof(float);
    constexpr uint64_t kVectorFloatBytes = kTile * sizeof(float);
    // Three FP16 input blocks, five single-head FP32 matrices, block-local
    // dA/state/g vectors, two single-head vectors and Broadcast scratch.
    return 3 * headBlockSize * kMatrixHalfBytes +
           5 * kMatrixFloatBytes +
           (3 * headBlockSize + 2) * kVectorFloatBytes +
           2 * kTile * kTile;
}

void CheckHalfNpu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_bwd_diag_finalize: ", name,
                " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf && tensor.is_contiguous(),
                "mamba2_ssd_bwd_diag_finalize: ", name,
                " must be contiguous float16");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_bwd_diag_finalize(
    const at::Tensor &dxDiag,
    const at::Tensor &dR,
    const at::Tensor &dBDiag,
    const at::Tensor &dBState,
    const at::Tensor &dCDiag,
    const at::Tensor &dW,
    const at::Tensor &w,
    const at::Tensor &r,
    const at::Tensor &dACumsum,
    int64_t groups,
    const c10::optional<at::Tensor> &dCOff)
{
    const std::array<std::pair<const at::Tensor *, const char *>, 5> halves{{
        {&dxDiag, "dx_diag"},
        {&dBState, "db_state"}, {&dW, "d_w"},
        {&w, "w"}, {&r, "r"},
    }};
    for (const auto &entry : halves) {
        CheckHalfNpu(*entry.first, entry.second);
        TORCH_CHECK(entry.first->sizes() == dxDiag.sizes(),
                    "mamba2_ssd_bwd_diag_finalize: all matrix inputs must "
                    "have identical shapes");
    }
    CheckHalfNpu(dR, "d_r");
    CheckHalfNpu(dBDiag, "db_diag");
    CheckHalfNpu(dCDiag, "dc_diag");
    TORCH_CHECK(dxDiag.dim() == 5 && dxDiag.size(3) == kTile &&
                dxDiag.size(4) == kTile,
                "mamba2_ssd_bwd_diag_finalize: matrices must be "
                "[B,H,K,64,64]");
    TORCH_CHECK(dACumsum.device().type() == DEVICE_TYPE &&
                dACumsum.scalar_type() == at::kFloat &&
                dACumsum.is_contiguous(),
                "mamba2_ssd_bwd_diag_finalize: d_a_cumsum must be "
                "contiguous float32 NPU");

    const int64_t batch = dxDiag.size(0);
    const int64_t heads = dxDiag.size(1);
    const int64_t chunks = dxDiag.size(2);
    TORCH_CHECK(dACumsum.sizes() ==
                    at::IntArrayRef({batch, heads, chunks, kTile}),
                "mamba2_ssd_bwd_diag_finalize: d_a_cumsum shape mismatch");
    TORCH_CHECK(batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
                heads % groups == 0,
                "mamba2_ssd_bwd_diag_finalize: requires positive B/H/K/G "
                "and H divisible by G");
    const bool hasDCOff = dCOff.has_value();
    if (hasDCOff) {
        const at::Tensor &value = dCOff.value();
        TORCH_CHECK(value.device().type() == DEVICE_TYPE &&
                    value.scalar_type() == at::kFloat &&
                    value.is_contiguous() &&
                    value.sizes() == at::IntArrayRef(
                        {batch, chunks, kTile, groups, kTile}),
                    "mamba2_ssd_bwd_diag_finalize: d_c_off must be "
                    "contiguous float32 [B,K,64,G,64]");
    }

    const std::array<int64_t, 5> headMatrixSizes{
        batch, heads, chunks, kTile, kTile};
    const std::array<int64_t, 5> groupMatrixSizes{
        batch, groups, chunks, kTile, kTile};
    const at::IntArrayRef headMatrixShape(headMatrixSizes);
    const at::IntArrayRef groupMatrixShape(groupMatrixSizes);
    TORCH_CHECK(
        (dBDiag.sizes() == headMatrixShape &&
         dCDiag.sizes() == headMatrixShape) ||
        (dBDiag.sizes() == groupMatrixShape &&
         dCDiag.sizes() == groupMatrixShape),
        "mamba2_ssd_bwd_diag_finalize: db_diag/dc_diag must both be "
        "[B,H,K,64,64] or [B,G,K,64,64]");
    const int64_t groupedDiag =
        dBDiag.sizes() == groupMatrixShape ? 1 : 0;

    const int64_t headsPerGroup = heads / groups;
    const std::array<int64_t, 6> groupedDRSizes{
        batch, chunks, groups, kTile, headsPerGroup, kTile};
    const bool groupedDR = dR.dim() == 6;
    const int64_t groupedDRArg = groupedDR ? 1 : 0;
    TORCH_CHECK(
        (!groupedDR && dR.sizes() == dxDiag.sizes()) ||
        (groupedDR && dR.sizes() == at::IntArrayRef(groupedDRSizes)),
        "mamba2_ssd_bwd_diag_finalize: d_r must be [B,H,K,64,64] "
        "or grouped [B,K,G,64,R,64]");
    const int64_t headBlockSize = headsPerGroup < kMaxHeadBlock
        ? headsPerGroup : kMaxHeadBlock;
    auto fp32Options = dxDiag.options().dtype(at::kFloat);
    // d_xdt_total is an internal Diag->Dt workspace.  Diag arithmetic stays
    // FP32 in UB; storing FP16 halves both this write and Dt's following read.
    at::Tensor dx = at::empty(dxDiag.sizes(), dxDiag.options());
    // Return the public B/C layout [B, K, T, G, N] directly.  The kernel
    // writes each group's 64 rows with an MTE3 destination stride, avoiding
    // a full-device transpose in the Python epilogue.
    at::Tensor dBGroup = at::empty(
        {batch, chunks, kTile, groups, kTile}, fp32Options);
    at::Tensor dCGroup = at::empty_like(dBGroup);
    at::Tensor gCs = at::empty(
        {batch, heads, chunks, kTile}, fp32Options);

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t requiredUbBytes = RequiredUbBytes(headBlockSize);
    TORCH_CHECK(ubSize >= requiredUbBytes,
                "mamba2_ssd_bwd_diag_finalize: insufficient UB for head block");
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t taskCount = batch * chunks * groups;
    const int64_t usedCoreNum = std::min(taskCount, aivCoreNum);
    TORCH_CHECK(usedCoreNum > 0,
                "mamba2_ssd_bwd_diag_finalize: no AIV cores");
    const uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    const at::Tensor &dCOffArg = hasDCOff ? dCOff.value() : dACumsum;
    const int64_t hasDCOffArg = hasDCOff ? 1 : 0;

    EXEC_KERNEL_CMD(
        mamba2_ssd_bwd_diag_finalize, blockDim,
        dxDiag, dR, dBDiag, dBState, dCDiag, dCOffArg,
        dW, w, r, dACumsum,
        dx, dBGroup, dCGroup, gCs,
        batch, heads, chunks, groups, headsPerGroup, headBlockSize,
        groupedDiag, groupedDRArg, hasDCOffArg, usedCoreNum);
    return std::make_tuple(dx, dBGroup, dCGroup, gCs);
}

}  // namespace ascend_kernel
