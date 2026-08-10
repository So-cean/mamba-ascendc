// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

at::Tensor mamba2_ssd_state_vector_epilogue(
    const at::Tensor &yOff,
    const at::Tensor &dACumsum,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z)
{
    TORCH_CHECK(
        yOff.device().type() == c10::DeviceType::PrivateUse1 &&
            dACumsum.device().type() == c10::DeviceType::PrivateUse1 &&
            yDiag.device().type() == c10::DeviceType::PrivateUse1 &&
            x.device().type() == c10::DeviceType::PrivateUse1 &&
            d.device().type() == c10::DeviceType::PrivateUse1 &&
            z.device().type() == c10::DeviceType::PrivateUse1,
        "mamba2_ssd_state_vector_epilogue: inputs must be NPU tensors");
    TORCH_CHECK(yOff.scalar_type() == at::kFloat &&
                    dACumsum.scalar_type() == at::kFloat &&
                    yDiag.scalar_type() == at::kFloat &&
                    x.scalar_type() == at::kFloat &&
                    d.scalar_type() == at::kFloat &&
                    z.scalar_type() == at::kFloat,
                "mamba2_ssd_state_vector_epilogue: all inputs must be FP32");
    TORCH_CHECK(yOff.is_contiguous() && dACumsum.is_contiguous() &&
                    yDiag.is_contiguous() && x.is_contiguous() &&
                    d.is_contiguous() && z.is_contiguous(),
                "mamba2_ssd_state_vector_epilogue: inputs must be contiguous");
    TORCH_CHECK(yOff.dim() == 5 && dACumsum.dim() == 4 &&
                    yDiag.dim() == 5 && x.dim() == 4 && d.dim() == 2 &&
                    z.dim() == 4 && yOff.sizes() == yDiag.sizes() &&
                    x.sizes() == z.sizes(),
                "mamba2_ssd_state_vector_epilogue: invalid ranks or shapes");
    const int64_t batch = yOff.size(0);
    const int64_t heads = yOff.size(1);
    const int64_t chunks = yOff.size(2);
    TORCH_CHECK(yOff.size(3) == 64 && yOff.size(4) == 64 &&
                    dACumsum.sizes() ==
                        at::IntArrayRef({batch, heads, chunks, 64}) &&
                    x.sizes() ==
                        at::IntArrayRef({batch, chunks * 64, heads, 64}) &&
                    d.sizes() == at::IntArrayRef({heads, 64}),
                "mamba2_ssd_state_vector_epilogue: only T=P=64 is supported");
    at::Tensor out = at::empty_like(x);
    EXEC_NPU_CMD(aclnnMamba2SsdStateVectorEpilogue, yOff, dACumsum,
                 yDiag, x, d, z, out);
    return out;
}

}  // namespace ascend_kernel
