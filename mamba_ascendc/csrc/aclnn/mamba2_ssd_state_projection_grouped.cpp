// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

at::Tensor mamba2_ssd_state_projection_grouped(
    const at::Tensor &statesGrouped,
    const at::Tensor &cCube)
{
    TORCH_CHECK(
        statesGrouped.device().type() == c10::DeviceType::PrivateUse1 &&
            cCube.device().type() == c10::DeviceType::PrivateUse1,
        "mamba2_ssd_state_projection_grouped: inputs must be NPU tensors");
    TORCH_CHECK(statesGrouped.scalar_type() == at::kHalf &&
                    cCube.scalar_type() == at::kHalf,
                "mamba2_ssd_state_projection_grouped: expected FP16 inputs");
    TORCH_CHECK(statesGrouped.is_contiguous() && cCube.is_contiguous(),
                "mamba2_ssd_state_projection_grouped: inputs must be "
                "contiguous");
    TORCH_CHECK(statesGrouped.dim() == 6 && cCube.dim() == 5,
                "mamba2_ssd_state_projection_grouped: expected "
                "[B,K,G,N,R,P] and [B,K,G,T,N]");
    const int64_t batch = statesGrouped.size(0);
    const int64_t chunks = statesGrouped.size(1);
    const int64_t groups = statesGrouped.size(2);
    TORCH_CHECK(
        batch > 0 && chunks > 0 && groups > 0 &&
            statesGrouped.size(3) == 64 &&
            statesGrouped.size(4) == 4 &&
            statesGrouped.size(5) == 64 && cCube.size(0) == batch &&
            cCube.size(1) == chunks && cCube.size(2) == groups &&
            cCube.size(3) == 64 && cCube.size(4) == 64,
        "mamba2_ssd_state_projection_grouped: only T=N=P=64 and R=4 "
        "are supported");
    auto yGrouped = at::empty(
        {batch, chunks, groups, 64, 4, 64},
        statesGrouped.options().dtype(at::kHalf));
    EXEC_NPU_CMD(aclnnMamba2SsdStateProjectionGrouped,
                 statesGrouped, cCube, yGrouped);
    return yGrouped;
}

}  // namespace ascend_kernel
