// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_chunk_mix(
    const at::Tensor &xCube,
    const at::Tensor &dACumsum,
    const at::Tensor &bCube,
    const at::Tensor &cCube)
{
    TORCH_CHECK(xCube.device().type() == c10::DeviceType::PrivateUse1 &&
                    dACumsum.device().type() == c10::DeviceType::PrivateUse1 &&
                    bCube.device().type() == c10::DeviceType::PrivateUse1 &&
                    cCube.device().type() == c10::DeviceType::PrivateUse1,
                "mamba2_ssd_chunk_mix: all inputs must be NPU tensors");
    TORCH_CHECK(xCube.scalar_type() == at::kHalf &&
                    bCube.scalar_type() == at::kHalf &&
                    cCube.scalar_type() == at::kHalf &&
                    dACumsum.scalar_type() == at::kFloat,
                "mamba2_ssd_chunk_mix: expected fp16 x/B/C and fp32 dA_cumsum");
    TORCH_CHECK(xCube.is_contiguous() && dACumsum.is_contiguous() &&
                    bCube.is_contiguous() && cCube.is_contiguous(),
                "mamba2_ssd_chunk_mix: inputs must be contiguous ND tensors");
    TORCH_CHECK(xCube.dim() == 5 && dACumsum.dim() == 4 &&
                    bCube.dim() == 5 && cCube.dim() == 5,
                "mamba2_ssd_chunk_mix: expected x[B,H,K,T,P], dA[B,H,K,T], "
                "B_t[B,K,G,N,T], C[B,K,G,T,N]");

    const int64_t batch = xCube.size(0);
    const int64_t heads = xCube.size(1);
    const int64_t chunks = xCube.size(2);
    const int64_t chunkSize = xCube.size(3);
    const int64_t headDim = xCube.size(4);
    const int64_t groups = bCube.size(2);
    const int64_t stateDim = cCube.size(4);
    TORCH_CHECK(batch > 0 && heads > 0 && chunks > 0 && groups > 0 &&
                    heads % groups == 0,
                "mamba2_ssd_chunk_mix: invalid batch/head/chunk/group dimensions");
    TORCH_CHECK(headDim == 64 &&
                    ((chunkSize == 64 &&
                      (stateDim == 64 || stateDim == 128)) ||
                     (chunkSize == 128 && stateDim == 128)),
                "mamba2_ssd_chunk_mix: expected P=64 with "
                "(T,N) in {(64,64),(64,128),(128,128)}");
    TORCH_CHECK(dACumsum.sizes() == at::IntArrayRef({batch, heads, chunks, chunkSize}),
                "mamba2_ssd_chunk_mix: dA_cumsum shape mismatch");
    TORCH_CHECK(bCube.sizes() == at::IntArrayRef(
                    {batch, chunks, groups, stateDim, chunkSize}) &&
                    cCube.sizes() == at::IntArrayRef(
                    {batch, chunks, groups, chunkSize, stateDim}),
                "mamba2_ssd_chunk_mix: expected B_t[B,K,G,N,T] and "
                "C[B,K,G,T,N]");

    at::Tensor yDiag = at::empty(
        {batch, heads, chunks, chunkSize, headDim},
        xCube.options().dtype(at::kFloat));
    const bool stateNpLayout =
        (chunkSize == 64 && stateDim == 64) || chunkSize == 128;
    at::Tensor chunkStates = at::empty(
        stateNpLayout
            ? at::IntArrayRef({batch, heads, chunks, stateDim, headDim})
            : at::IntArrayRef({batch, heads, chunks, headDim, stateDim}),
        xCube.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnMamba2SsdChunkMix, xCube, dACumsum, bCube, cCube,
                 yDiag, chunkStates);
    return {yDiag, chunkStates};
}

} // namespace ascend_kernel
