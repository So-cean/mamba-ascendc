// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor>
mamba2_ssd_bwd_off_group_reduce(
    const at::Tensor &qGroup,
    const at::Tensor &stateGroup,
    const at::Tensor &cCube)
{
    const auto device = qGroup.device();
    TORCH_CHECK(
        device.type() == c10::DeviceType::PrivateUse1 &&
            stateGroup.device() == device && cCube.device() == device,
        "mamba2_ssd_bwd_off_group_reduce: inputs must be on the same NPU");
    TORCH_CHECK(
        qGroup.scalar_type() == at::kHalf &&
            stateGroup.scalar_type() == at::kHalf &&
            cCube.scalar_type() == at::kHalf,
        "mamba2_ssd_bwd_off_group_reduce: expected FP16 inputs");
    TORCH_CHECK(
        qGroup.is_contiguous() && stateGroup.is_contiguous() &&
            cCube.is_contiguous(),
        "mamba2_ssd_bwd_off_group_reduce: inputs must be contiguous ND tensors");
    TORCH_CHECK(
        qGroup.dim() == 6 && stateGroup.dim() == 6 && cCube.dim() == 5,
        "mamba2_ssd_bwd_off_group_reduce: expected q/state[B,G,R,K,64,64] "
        "and C[B,K,G,64,64]");

    const int64_t batch = qGroup.size(0);
    const int64_t groups = qGroup.size(1);
    const int64_t headsPerGroup = qGroup.size(2);
    const int64_t chunks = qGroup.size(3);
    TORCH_CHECK(
        batch > 0 && groups > 0 && headsPerGroup > 0 && chunks > 0,
        "mamba2_ssd_bwd_off_group_reduce: invalid B/G/R/K dimensions");
    TORCH_CHECK(
        qGroup.size(4) == 64 && qGroup.size(5) == 64,
        "mamba2_ssd_bwd_off_group_reduce M1 requires T=P=N=64");
    TORCH_CHECK(
        stateGroup.sizes() == qGroup.sizes(),
        "mamba2_ssd_bwd_off_group_reduce: state_group shape mismatch");
    TORCH_CHECK(
        cCube.sizes() == at::IntArrayRef({batch, chunks, groups, 64, 64}),
        "mamba2_ssd_bwd_off_group_reduce: c_cube shape mismatch");

    at::Tensor dCGroup = at::empty(
        {batch, chunks, 64, groups, 64},
        qGroup.options().dtype(at::kFloat));
    at::Tensor gDa = at::empty(
        {batch, groups * headsPerGroup, chunks, 64},
        qGroup.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMamba2SsdBwdOffGroupReduce,
                 qGroup, stateGroup, cCube, dCGroup, gDa);
    return {dCGroup, gDa};
}

} // namespace ascend_kernel
