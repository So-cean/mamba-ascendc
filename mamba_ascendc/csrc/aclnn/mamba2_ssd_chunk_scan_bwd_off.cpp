// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_chunk_scan_bwd_off(
    const at::Tensor &gy,
    const at::Tensor &statesStart,
    const at::Tensor &dACumsum,
    const at::Tensor &cCube)
{
    const auto device = gy.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1 &&
                    statesStart.device() == device &&
                    dACumsum.device() == device && cCube.device() == device,
                "mamba2_ssd_chunk_scan_bwd_off: inputs must be on the same NPU");
    TORCH_CHECK(gy.scalar_type() == at::kFloat &&
                    statesStart.scalar_type() == at::kFloat &&
                    dACumsum.scalar_type() == at::kFloat &&
                    cCube.scalar_type() == at::kHalf,
                "mamba2_ssd_chunk_scan_bwd_off: expected FP32 gy/state/dA "
                "and FP16 C");
    TORCH_CHECK(gy.is_contiguous() && statesStart.is_contiguous() &&
                    dACumsum.is_contiguous() && cCube.is_contiguous(),
                "mamba2_ssd_chunk_scan_bwd_off: inputs must be contiguous ND tensors");
    TORCH_CHECK(gy.dim() == 5 && statesStart.dim() == 5 &&
                    dACumsum.dim() == 4 && cCube.dim() == 5,
                "mamba2_ssd_chunk_scan_bwd_off: expected gy[B,H,K,T,P], "
                "state[B,H,K,P,N], dA[B,H,K,T], C[B,K,G,T,N]");

    const int64_t batch = gy.size(0);
    const int64_t heads = gy.size(1);
    const int64_t chunks = gy.size(2);
    const int64_t chunkSize = gy.size(3);
    const int64_t headDim = gy.size(4);
    const int64_t groups = cCube.size(2);
    const int64_t stateDim = statesStart.size(4);
    TORCH_CHECK(batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
                    heads % groups == 0,
                "mamba2_ssd_chunk_scan_bwd_off: invalid B/H/K/G dimensions");
    TORCH_CHECK(chunkSize == 64 && headDim == 64 &&
                    statesStart.size(3) == 64 && stateDim == 64,
                "mamba2_ssd_chunk_scan_bwd_off M1: requires T=P=N=64");
    TORCH_CHECK(statesStart.sizes() == at::IntArrayRef(
                    {batch, heads, chunks, 64, 64}),
                "mamba2_ssd_chunk_scan_bwd_off: states_start shape mismatch");
    TORCH_CHECK(dACumsum.sizes() == at::IntArrayRef(
                    {batch, heads, chunks, 64}),
                "mamba2_ssd_chunk_scan_bwd_off: dA_cumsum shape mismatch");
    TORCH_CHECK(cCube.sizes() == at::IntArrayRef(
                    {batch, chunks, groups, 64, 64}),
                "mamba2_ssd_chunk_scan_bwd_off: c_cube shape mismatch");

    at::Tensor dStatesStart = at::empty_like(statesStart);
    at::Tensor dCHead = at::empty(
        {batch, heads, chunks, 64, 64}, gy.options());
    at::Tensor gDaCsOff = at::empty_like(dACumsum);
    EXEC_NPU_CMD(aclnnMamba2SsdChunkScanBwdOff,
                 gy, statesStart, dACumsum, cCube,
                 dStatesStart, dCHead, gDaCsOff);
    return {dStatesStart, dCHead, gDaCsOff};
}

} // namespace ascend_kernel
