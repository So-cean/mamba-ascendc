// Copyright (c) 2026, mamba-ascendc authors.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <tuple>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_mamba2_ssd_bwd_gate.h"
#include "aclrtlaunch_mamba2_ssd_bwd_gate_reduce.h"

namespace ascend_kernel {

namespace {
constexpr int64_t kGateHeadBlock = 8;

void CheckFp32Npu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_bwd_gate: ", name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.is_contiguous(),
                "mamba2_ssd_bwd_gate: ", name,
                " must be contiguous float32");
}

void CheckGateCacheNpu(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE,
                "mamba2_ssd_bwd_gate: y_pre must be an NPU tensor");
    TORCH_CHECK((tensor.scalar_type() == at::kFloat ||
                 tensor.scalar_type() == at::kHalf) &&
                    tensor.is_contiguous(),
                "mamba2_ssd_bwd_gate: y_pre must be contiguous float32 or "
                "float16");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_bwd_gate(
    const at::Tensor &dout,
    const at::Tensor &z,
    const at::Tensor &yPre,
    const at::Tensor &x)
{
    CheckFp32Npu(dout, "dout");
    CheckFp32Npu(z, "z");
    CheckGateCacheNpu(yPre);
    CheckFp32Npu(x, "x");
    TORCH_CHECK(dout.dim() == 4 && dout.sizes() == z.sizes() &&
                dout.sizes() == yPre.sizes() && dout.sizes() == x.sizes(),
                "mamba2_ssd_bwd_gate: inputs must have identical [B,L,H,P]");
    const int64_t batch = dout.size(0);
    const int64_t seqlen = dout.size(1);
    const int64_t nheads = dout.size(2);
    const int64_t headdim = dout.size(3);
    TORCH_CHECK(batch > 0 && nheads > 0 && seqlen % 64 == 0 && headdim == 64,
                "mamba2_ssd_bwd_gate: requires B/H>0, L%64=0 and P=64");
    const int64_t nchunks = seqlen / 64;
    TORCH_CHECK(batch * nchunks <= 512,
                "mamba2_ssd_bwd_gate: B*K must be <=512");

    at::Tensor gyHead = at::empty(
        {batch, nheads, nchunks, 64, headdim},
        dout.options().dtype(at::kHalf));
    at::Tensor dz = at::empty_like(dout);
    at::Tensor dDPartial = at::empty(
        {nheads, batch, nchunks, headdim}, dout.options());
    at::Tensor dD = at::empty({nheads, headdim}, dout.options());

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t nHeadBlocks =
        (nheads + kGateHeadBlock - 1) / kGateHeadBlock;
    const int64_t taskCount = batch * nHeadBlocks * nchunks;
    const int64_t phase0Cores = std::min(taskCount, aivCoreNum);
    const int64_t phase1Cores = std::min(nheads, aivCoreNum);
    TORCH_CHECK(phase0Cores > 0 && phase1Cores > 0,
                "mamba2_ssd_bwd_gate: no AIV cores");
    const uint32_t blockDim0 = static_cast<uint32_t>(phase0Cores);
    const uint32_t blockDim1 = static_cast<uint32_t>(phase1Cores);
    const int64_t computeD = 1;
    const int64_t yPreIsHalf = yPre.scalar_type() == at::kHalf ? 1 : 0;

    EXEC_KERNEL_CMD(
        mamba2_ssd_bwd_gate, blockDim0,
        dout, z, yPre, x, gyHead, dz, dDPartial,
        batch, seqlen, nheads, nchunks, phase0Cores, computeD, yPreIsHalf);
    EXEC_KERNEL_CMD(
        mamba2_ssd_bwd_gate_reduce, blockDim1,
        dDPartial, dD, batch, nheads, nchunks, phase1Cores);
    return std::make_tuple(gyHead, dz, dD);
}

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_bwd_gate_nodd(
    const at::Tensor &dout,
    const at::Tensor &z,
    const at::Tensor &yPre)
{
    CheckFp32Npu(dout, "dout");
    CheckFp32Npu(z, "z");
    CheckGateCacheNpu(yPre);
    TORCH_CHECK(dout.dim() == 4 && dout.sizes() == z.sizes() &&
                dout.sizes() == yPre.sizes(),
                "mamba2_ssd_bwd_gate_nodd: inputs must have identical "
                "[B,L,H,P]");
    const int64_t batch = dout.size(0);
    const int64_t seqlen = dout.size(1);
    const int64_t nheads = dout.size(2);
    const int64_t headdim = dout.size(3);
    TORCH_CHECK(batch > 0 && nheads > 0 && seqlen % 64 == 0 && headdim == 64,
                "mamba2_ssd_bwd_gate_nodd: requires B/H>0, L%64=0 and P=64");
    const int64_t nchunks = seqlen / 64;
    TORCH_CHECK(batch * nchunks <= 512,
                "mamba2_ssd_bwd_gate_nodd: B*K must be <=512");

    at::Tensor gyHead = at::empty(
        {batch, nheads, nchunks, 64, headdim},
        dout.options().dtype(at::kHalf));
    at::Tensor dz = at::empty_like(dout);
    at::Tensor unusedPartial = at::empty({1}, dout.options());

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t aivCoreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t nHeadBlocks =
        (nheads + kGateHeadBlock - 1) / kGateHeadBlock;
    const int64_t taskCount = batch * nHeadBlocks * nchunks;
    const int64_t phase0Cores = std::min(taskCount, aivCoreNum);
    TORCH_CHECK(phase0Cores > 0,
                "mamba2_ssd_bwd_gate_nodd: no AIV cores");
    const uint32_t blockDim0 = static_cast<uint32_t>(phase0Cores);
    const int64_t computeD = 0;
    const int64_t yPreIsHalf = yPre.scalar_type() == at::kHalf ? 1 : 0;

    EXEC_KERNEL_CMD(
        mamba2_ssd_bwd_gate, blockDim0,
        dout, z, yPre, dout, gyHead, dz, unusedPartial,
        batch, seqlen, nheads, nchunks, phase0Cores, computeD, yPreIsHalf);
    return std::make_tuple(gyHead, dz);
}

}  // namespace ascend_kernel
