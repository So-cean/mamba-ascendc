// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

at::Tensor mamba2_ssd_state_projection(
    const at::Tensor &statesStart,
    const at::Tensor &cCube)
{
    TORCH_CHECK(
        statesStart.device().type() == c10::DeviceType::PrivateUse1 &&
            cCube.device().type() == c10::DeviceType::PrivateUse1,
        "mamba2_ssd_state_projection: inputs must be NPU tensors");
    TORCH_CHECK(statesStart.scalar_type() == at::kFloat &&
                    cCube.scalar_type() == at::kHalf,
                "mamba2_ssd_state_projection: expected FP32 states and FP16 C");
    TORCH_CHECK(statesStart.is_contiguous() && cCube.is_contiguous(),
                "mamba2_ssd_state_projection: inputs must be contiguous");
    TORCH_CHECK(statesStart.dim() == 5 && cCube.dim() == 5,
                "mamba2_ssd_state_projection: expected [B,H,K,P,N] and "
                "[B,K,G,T,N]");
    const int64_t batch = statesStart.size(0);
    const int64_t heads = statesStart.size(1);
    const int64_t chunks = statesStart.size(2);
    const int64_t groups = cCube.size(2);
    TORCH_CHECK(batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
                    heads % groups == 0 && statesStart.size(3) == 64 &&
                    statesStart.size(4) == 64 && cCube.size(0) == batch &&
                    cCube.size(1) == chunks && cCube.size(3) == 64 &&
                    cCube.size(4) == 64,
                "mamba2_ssd_state_projection: only T=N=P=64 is supported");
    auto yOff = at::empty({batch, heads, chunks, 64, 64},
                           statesStart.options());
    EXEC_NPU_CMD(aclnnMamba2SsdStateProjection, statesStart, cCube, yOff);
    return yOff;
}

}  // namespace ascend_kernel
