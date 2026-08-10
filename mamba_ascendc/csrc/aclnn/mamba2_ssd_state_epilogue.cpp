// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_state_epilogue(
    const at::Tensor &chunkStates,
    const at::Tensor &dACumsum,
    const at::Tensor &cCube,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z,
    const c10::optional<at::Tensor> &initialStates)
{
    TORCH_CHECK(chunkStates.device().type() == c10::DeviceType::PrivateUse1 &&
                    dACumsum.device().type() == c10::DeviceType::PrivateUse1 &&
                    cCube.device().type() == c10::DeviceType::PrivateUse1 &&
                    yDiag.device().type() == c10::DeviceType::PrivateUse1 &&
                    x.device().type() == c10::DeviceType::PrivateUse1 &&
                    d.device().type() == c10::DeviceType::PrivateUse1 &&
                    z.device().type() == c10::DeviceType::PrivateUse1,
                "mamba2_ssd_state_epilogue: all inputs must be NPU tensors");
    TORCH_CHECK(chunkStates.scalar_type() == at::kFloat &&
                    dACumsum.scalar_type() == at::kFloat &&
                    cCube.scalar_type() == at::kHalf &&
                    yDiag.scalar_type() == at::kFloat &&
                    x.scalar_type() == at::kFloat && d.scalar_type() == at::kFloat &&
                    z.scalar_type() == at::kFloat,
                "mamba2_ssd_state_epilogue: expected fp16 C and fp32 other inputs");
    TORCH_CHECK(chunkStates.is_contiguous() && dACumsum.is_contiguous() &&
                    cCube.is_contiguous() && yDiag.is_contiguous() &&
                    x.is_contiguous() && d.is_contiguous() && z.is_contiguous(),
                "mamba2_ssd_state_epilogue: inputs must be contiguous");
    const bool groupedInput = chunkStates.dim() == 6;
    TORCH_CHECK((chunkStates.dim() == 5 || groupedInput) &&
                    dACumsum.dim() == 4 &&
                    cCube.dim() == 5 && yDiag.dim() == 5 && x.dim() == 4 &&
                    d.dim() == 2 && z.dim() == 4 && x.sizes() == z.sizes(),
                "mamba2_ssd_state_epilogue: invalid ranks or x/z shape");
    if (initialStates.has_value()) {
        TORCH_CHECK(initialStates->device().type() == c10::DeviceType::PrivateUse1 &&
                        initialStates->scalar_type() == at::kFloat &&
                        initialStates->is_contiguous() && initialStates->dim() == 4,
                    "mamba2_ssd_state_epilogue: initial_states must be contiguous FP32 NPU");
    }

    const int64_t batch = chunkStates.size(0);
    const int64_t groups = groupedInput ? chunkStates.size(2)
                                        : cCube.size(2);
    const int64_t headsPerGroup = groupedInput ? chunkStates.size(4) : 0;
    const int64_t heads = groupedInput
        ? groups * headsPerGroup : chunkStates.size(1);
    const int64_t chunkSize = cCube.size(3);
    const bool stateNpLayout = groupedInput ||
        chunkSize == 128 || (chunkSize == 64 && cCube.size(4) == 64);
    const int64_t headDim = groupedInput
        ? chunkStates.size(5)
        : chunkStates.size(stateNpLayout ? 4 : 3);
    const int64_t stateDim = groupedInput
        ? chunkStates.size(3)
        : chunkStates.size(stateNpLayout ? 3 : 4);
    TORCH_CHECK(
        headDim == 64 &&
            ((chunkSize == 64 && (stateDim == 64 || stateDim == 128)) ||
             (chunkSize == 128 && stateDim == 128)),
        "mamba2_ssd_state_epilogue: expected P=64 and "
        "(T,N) in {(64,64),(64,128),(128,128)}");
    if (groupedInput) {
        TORCH_CHECK(
            chunkSize == 64 && stateDim == 64 && headsPerGroup == 4 &&
                chunkStates.size(1) == dACumsum.size(2) &&
                cCube.size(0) == batch &&
                cCube.size(1) == chunkStates.size(1) &&
                cCube.size(2) == groups,
            "mamba2_ssd_state_epilogue: grouped input must be "
            "[B,K,G,64,4,64]");
    }
    if (initialStates.has_value()) {
        TORCH_CHECK(
            initialStates->sizes() ==
                at::IntArrayRef({batch, heads, headDim, stateDim}),
            "mamba2_ssd_state_epilogue: public initial state must be [B,H,P,N]");
    }
    at::Tensor out = at::empty_like(x);
    at::Tensor finalStateInternal = at::empty(
        stateNpLayout
            ? at::IntArrayRef({batch, heads, stateDim, headDim})
            : at::IntArrayRef({batch, heads, headDim, stateDim}),
        chunkStates.options());
    at::Tensor initialInternal = initialStates.has_value()
        ? (stateNpLayout
               ? initialStates->transpose(-1, -2).contiguous()
               : initialStates.value())
        : chunkStates;
    EXEC_NPU_CMD(aclnnMamba2SsdStateEpilogue, chunkStates, dACumsum,
                 cCube, yDiag, x, d, z, initialInternal, out,
                 finalStateInternal);
    return {out, stateNpLayout ? finalStateInternal.transpose(-1, -2)
                               : finalStateInternal};
}

} // namespace ascend_kernel
