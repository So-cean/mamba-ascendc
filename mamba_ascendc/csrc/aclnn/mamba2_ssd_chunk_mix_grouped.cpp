// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_chunk_mix_grouped(
    const at::Tensor &xCube,
    const at::Tensor &dACumsum,
    const at::Tensor &bCube,
    const at::Tensor &cCube)
{
    for (const auto *tensor : {&xCube, &dACumsum, &bCube, &cCube}) {
        TORCH_CHECK(tensor->device().type() == c10::DeviceType::PrivateUse1 &&
                        tensor->is_contiguous(),
                    "mamba2_ssd_chunk_mix_grouped: inputs must be "
                    "contiguous NPU tensors");
    }
    TORCH_CHECK(xCube.scalar_type() == at::kHalf &&
                    bCube.scalar_type() == at::kHalf &&
                    cCube.scalar_type() == at::kHalf &&
                    dACumsum.scalar_type() == at::kFloat,
                "mamba2_ssd_chunk_mix_grouped: expected FP16 x/B/C and "
                "FP32 dA");
    TORCH_CHECK((xCube.dim() == 5 || xCube.dim() == 6) &&
                    dACumsum.dim() == 4 &&
                    bCube.dim() == 5 && cCube.dim() == 5,
                "mamba2_ssd_chunk_mix_grouped: invalid ranks");
    const bool groupedX = xCube.dim() == 6;
    const int64_t batch = xCube.size(0);
    const int64_t chunks = groupedX ? xCube.size(1) : xCube.size(2);
    const int64_t groups = bCube.size(2);
    const int64_t heads = groupedX ? groups * xCube.size(4) : xCube.size(1);
    const bool xShapeOk = groupedX
        ? xCube.sizes() ==
              at::IntArrayRef({batch, chunks, groups, 64, 4, 64})
        : xCube.sizes() ==
              at::IntArrayRef({batch, heads, chunks, 64, 64});
    TORCH_CHECK(
        batch > 0 && chunks > 0 && groups > 0 && heads == 4 * groups &&
            xShapeOk &&
            dACumsum.sizes() ==
                at::IntArrayRef({batch, heads, chunks, 64}) &&
            bCube.sizes() ==
                at::IntArrayRef({batch, chunks, groups, 64, 64}) &&
            cCube.sizes() ==
                at::IntArrayRef({batch, chunks, groups, 64, 64}),
        "mamba2_ssd_chunk_mix_grouped: only T=N=P=64 and H/G=4 are "
        "supported");
    auto yDiag = at::empty(xCube.sizes(), xCube.options().dtype(at::kFloat));
    auto statesGrouped = at::empty(
        {batch, chunks, groups, 64, 4, 64},
        xCube.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMamba2SsdChunkMixGrouped,
                 xCube, dACumsum, bCube, cCube, yDiag, statesGrouped);
    return {yDiag, statesGrouped};
}

}  // namespace ascend_kernel
