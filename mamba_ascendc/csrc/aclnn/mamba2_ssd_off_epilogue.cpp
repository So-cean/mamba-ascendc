// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

at::Tensor mamba2_ssd_off_epilogue(
    const at::Tensor &cCube,
    const at::Tensor &statesStart,
    const at::Tensor &dACumsum,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z)
{
    TORCH_CHECK(cCube.device().type() == c10::DeviceType::PrivateUse1 &&
                    statesStart.device().type() == c10::DeviceType::PrivateUse1 &&
                    dACumsum.device().type() == c10::DeviceType::PrivateUse1 &&
                    yDiag.device().type() == c10::DeviceType::PrivateUse1 &&
                    x.device().type() == c10::DeviceType::PrivateUse1 &&
                    d.device().type() == c10::DeviceType::PrivateUse1 &&
                    z.device().type() == c10::DeviceType::PrivateUse1,
                "mamba2_ssd_off_epilogue: all inputs must be NPU tensors");
    TORCH_CHECK(cCube.scalar_type() == at::kHalf &&
                    statesStart.scalar_type() == at::kFloat &&
                    dACumsum.scalar_type() == at::kFloat &&
                    yDiag.scalar_type() == at::kFloat &&
                    x.scalar_type() == at::kFloat && d.scalar_type() == at::kFloat &&
                    z.scalar_type() == at::kFloat,
                "mamba2_ssd_off_epilogue: expected fp16 C and fp32 other inputs");
    TORCH_CHECK(cCube.is_contiguous() && statesStart.is_contiguous() &&
                    dACumsum.is_contiguous() && yDiag.is_contiguous() &&
                    x.is_contiguous() && d.is_contiguous() && z.is_contiguous(),
                "mamba2_ssd_off_epilogue: inputs must be contiguous ND tensors");
    TORCH_CHECK(cCube.dim() == 5 && statesStart.dim() == 5 &&
                    dACumsum.dim() == 4 && yDiag.dim() == 5 &&
                    x.dim() == 4 && d.dim() == 2 && z.dim() == 4,
                "mamba2_ssd_off_epilogue: invalid input ranks");
    TORCH_CHECK(x.sizes() == z.sizes(),
                "mamba2_ssd_off_epilogue: x/z shape mismatch");

    at::Tensor out = at::empty_like(x, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMamba2SsdOffEpilogue, cCube, statesStart,
                 dACumsum, yDiag, x, d, z, out);
    return out;
}

} // namespace ascend_kernel
