// Internal-test-only binding. Remove after the validated Matmul path is
// integrated into the production ChunkMix helper.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

at::Tensor mamba2_arch35_matmul_smoke_internal_test_only(
    const at::Tensor &weight,
    const at::Tensor &x_group)
{
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    x_group.device().type() == c10::DeviceType::PrivateUse1,
                "mamba2_arch35_matmul_smoke: inputs must be NPU tensors");
    TORCH_CHECK(weight.scalar_type() == at::kHalf &&
                    x_group.scalar_type() == at::kHalf,
                "mamba2_arch35_matmul_smoke: inputs must be float16");
    TORCH_CHECK(weight.is_contiguous() && x_group.is_contiguous(),
                "mamba2_arch35_matmul_smoke: inputs must be contiguous ND");
    const bool batched =
        weight.sizes() == at::IntArrayRef({4, 64, 64}) &&
        x_group.sizes() == at::IntArrayRef({4, 64, 64});
    const bool strided =
        weight.sizes() == at::IntArrayRef({64, 64}) &&
        x_group.sizes() == at::IntArrayRef({64, 4, 64});
    TORCH_CHECK(
        batched || strided,
        "mamba2_arch35_matmul_smoke: expected [64,64] x [64,4,64] "
        "or batched [4,64,64] x [4,64,64]");

    at::Tensor y_group = at::empty(
        x_group.sizes(), weight.options().dtype(at::kFloat));
    EXEC_NPU_CMD(
        aclnnMamba2Arch35MatmulSmoke, weight, x_group, y_group);
    return y_group;
}

} // namespace ascend_kernel
