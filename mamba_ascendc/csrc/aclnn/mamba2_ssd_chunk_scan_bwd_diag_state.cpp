// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_chunk_scan_bwd_diag_state(
    const at::Tensor &gy,
    const at::Tensor &xCube,
    const at::Tensor &dACumsum,
    const at::Tensor &bCube,
    const at::Tensor &cCube,
    const at::Tensor &dChunkStates)
{
    const auto device = gy.device();
    TORCH_CHECK(
        device.type() == c10::DeviceType::PrivateUse1 &&
            xCube.device() == device && dACumsum.device() == device &&
            bCube.device() == device && cCube.device() == device &&
            dChunkStates.device() == device,
        "mamba2_ssd_chunk_scan_bwd_diag_state: inputs must be on the same NPU");
    TORCH_CHECK(
        gy.scalar_type() == at::kFloat && xCube.scalar_type() == at::kHalf &&
            dACumsum.scalar_type() == at::kFloat &&
            bCube.scalar_type() == at::kHalf &&
            cCube.scalar_type() == at::kHalf &&
            dChunkStates.scalar_type() == at::kFloat,
        "mamba2_ssd_chunk_scan_bwd_diag_state: expected FP32 gy/dA/dU "
        "and FP16 x/B/C");
    TORCH_CHECK(
        gy.is_contiguous() && xCube.is_contiguous() &&
            dACumsum.is_contiguous() && bCube.is_contiguous() &&
            cCube.is_contiguous() && dChunkStates.is_contiguous(),
        "mamba2_ssd_chunk_scan_bwd_diag_state: inputs must be contiguous ND tensors");
    TORCH_CHECK(
        gy.dim() == 5 && xCube.dim() == 5 && dACumsum.dim() == 4 &&
            bCube.dim() == 5 && cCube.dim() == 5 &&
            dChunkStates.dim() == 5,
        "mamba2_ssd_chunk_scan_bwd_diag_state: expected gy/x/dU[B,H,K,64,64], "
        "dA[B,H,K,64], B/C[B,K,G,64,64]");

    const int64_t batch = gy.size(0);
    const int64_t heads = gy.size(1);
    const int64_t chunks = gy.size(2);
    const int64_t groups = bCube.size(2);
    TORCH_CHECK(
        batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
            heads % groups == 0,
        "mamba2_ssd_chunk_scan_bwd_diag_state: invalid B/H/K/G dimensions");
    TORCH_CHECK(
        gy.sizes() == at::IntArrayRef({batch, heads, chunks, 64, 64}) &&
            xCube.sizes() ==
                at::IntArrayRef({batch, heads, chunks, 64, 64}) &&
            dChunkStates.sizes() ==
                at::IntArrayRef({batch, heads, chunks, 64, 64}) &&
            dACumsum.sizes() ==
                at::IntArrayRef({batch, heads, chunks, 64}) &&
            bCube.sizes() ==
                at::IntArrayRef({batch, chunks, groups, 64, 64}) &&
            cCube.sizes() ==
                at::IntArrayRef({batch, chunks, groups, 64, 64}),
        "mamba2_ssd_chunk_scan_bwd_diag_state M1 requires T=P=N=64 "
        "and matching batch/head/chunk/group dimensions");

    at::Tensor dXdt = at::empty_like(gy);
    at::Tensor dBGroup = at::empty(cCube.sizes(), gy.options());
    at::Tensor dCDiagGroup = at::empty(cCube.sizes(), gy.options());
    at::Tensor gDaCs = at::empty_like(dACumsum);
    EXEC_NPU_CMD(aclnnMamba2SsdChunkScanBwdDiagState,
                 gy, xCube, dACumsum, bCube, cCube, dChunkStates,
                 dXdt, dBGroup, dCDiagGroup, gDaCs);
    return {dXdt, dBGroup, dCDiagGroup, gDaCs};
}

} // namespace ascend_kernel
