// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

at::Tensor mamba2_ssd_state_vector_epilogue_grouped(
    const at::Tensor &yGrouped,
    const at::Tensor &dACumsum,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z)
{
    TORCH_CHECK(yGrouped.device().type() == c10::DeviceType::PrivateUse1 &&
                    yGrouped.scalar_type() == at::kHalf &&
                    yGrouped.is_contiguous(),
                "mamba2_ssd_state_vector_epilogue_grouped: y_grouped must "
                "be a contiguous FP16 NPU tensor");
    TORCH_CHECK(yDiag.device().type() == c10::DeviceType::PrivateUse1 &&
                    yDiag.scalar_type() == at::kFloat &&
                    yDiag.is_contiguous(),
                "mamba2_ssd_state_vector_epilogue_grouped: y_diag must "
                "be a contiguous FP32 NPU tensor");
    for (const auto *tensor : {&dACumsum, &x, &d, &z}) {
        TORCH_CHECK(tensor->device().type() == c10::DeviceType::PrivateUse1 &&
                        tensor->scalar_type() == at::kFloat &&
                        tensor->is_contiguous(),
                    "mamba2_ssd_state_vector_epilogue_grouped: other "
                    "inputs must be contiguous FP32 NPU tensors");
    }
    TORCH_CHECK(yGrouped.dim() == 6 && dACumsum.dim() == 4 &&
                    (yDiag.dim() == 5 || yDiag.dim() == 6) &&
                    x.dim() == 4 && d.dim() == 2 &&
                    z.dim() == 4 && x.sizes() == z.sizes(),
                "mamba2_ssd_state_vector_epilogue_grouped: invalid ranks");
    const int64_t batch = yGrouped.size(0);
    const int64_t chunks = yGrouped.size(1);
    const int64_t groups = yGrouped.size(2);
    const int64_t headsPerGroup = yGrouped.size(4);
    const int64_t heads = groups * headsPerGroup;
    const bool diagShapeOk = yDiag.dim() == 6
        ? yDiag.sizes() == yGrouped.sizes()
        : yDiag.sizes() ==
              at::IntArrayRef({batch, heads, chunks, 64, 64});
    TORCH_CHECK(
        batch > 0 && chunks > 0 && groups > 0 &&
            yGrouped.size(3) == 64 && headsPerGroup == 4 &&
            yGrouped.size(5) == 64 &&
            dACumsum.sizes() ==
                at::IntArrayRef({batch, heads, chunks, 64}) &&
            diagShapeOk &&
            x.sizes() ==
                at::IntArrayRef({batch, chunks * 64, heads, 64}) &&
            d.sizes() == at::IntArrayRef({heads, 64}),
        "mamba2_ssd_state_vector_epilogue_grouped: only T=P=64 and "
        "H/G=4 are supported");
    auto out = at::empty_like(x);
    EXEC_NPU_CMD(aclnnMamba2SsdStateVectorEpilogueGrouped,
                 yGrouped, dACumsum, yDiag, x, d, z, out);
    return out;
}

}  // namespace ascend_kernel
