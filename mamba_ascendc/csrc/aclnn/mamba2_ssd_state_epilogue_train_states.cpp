// Copyright (c) 2026, mamba-ascendc authors.

#include "torch_aclnn_helper.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_state_epilogue_train_states(
    const at::Tensor &chunkStates,
    const at::Tensor &dACumsum,
    const at::Tensor &cCube,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z,
    const c10::optional<at::Tensor> &initialStates)
{
    const auto npu = c10::DeviceType::PrivateUse1;
    TORCH_CHECK(chunkStates.device().type() == npu &&
                    dACumsum.device().type() == npu &&
                    cCube.device().type() == npu &&
                    yDiag.device().type() == npu && x.device().type() == npu &&
                    d.device().type() == npu && z.device().type() == npu,
                "mamba2_ssd_state_epilogue_train_states: all inputs must be NPU tensors");
    TORCH_CHECK(chunkStates.scalar_type() == at::kFloat &&
                    dACumsum.scalar_type() == at::kFloat &&
                    cCube.scalar_type() == at::kHalf &&
                    yDiag.scalar_type() == at::kFloat &&
                    x.scalar_type() == at::kFloat &&
                    d.scalar_type() == at::kFloat &&
                    z.scalar_type() == at::kFloat,
                "mamba2_ssd_state_epilogue_train_states: expected fp16 C and fp32 other inputs");
    TORCH_CHECK(chunkStates.is_contiguous() && dACumsum.is_contiguous() &&
                    cCube.is_contiguous() && yDiag.is_contiguous() &&
                    x.is_contiguous() && d.is_contiguous() && z.is_contiguous(),
                "mamba2_ssd_state_epilogue_train_states: inputs must be contiguous");
    const bool groupedInput = chunkStates.dim() == 6;
    TORCH_CHECK((chunkStates.dim() == 5 || groupedInput) &&
                    dACumsum.dim() == 4 &&
                    cCube.dim() == 5 && yDiag.dim() == 5 && x.dim() == 4 &&
                    d.dim() == 2 && z.dim() == 4 && x.sizes() == z.sizes(),
                "mamba2_ssd_state_epilogue_train_states: invalid ranks or x/z shape");
    if (initialStates.has_value()) {
        TORCH_CHECK(initialStates->device().type() == npu &&
                        initialStates->scalar_type() == at::kFloat &&
                        initialStates->is_contiguous() && initialStates->dim() == 4,
                    "mamba2_ssd_state_epilogue_train_states: invalid initial_states");
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
        "mamba2_ssd_state_epilogue_train_states: unsupported P/T/N");
    if (groupedInput) {
        TORCH_CHECK(
            chunkSize == 64 && stateDim == 64 && headsPerGroup == 4 &&
                chunkStates.size(1) == dACumsum.size(2) &&
                cCube.size(0) == batch &&
                cCube.size(1) == chunkStates.size(1) &&
                cCube.size(2) == groups,
            "mamba2_ssd_state_epilogue_train_states: grouped input must be "
            "[B,K,G,64,4,64]");
    }
    if (initialStates.has_value()) {
        TORCH_CHECK(
            initialStates->sizes() ==
                at::IntArrayRef({batch, heads, headDim, stateDim}),
            "mamba2_ssd_state_epilogue_train_states: invalid initial state shape");
    }

    at::Tensor out = at::empty_like(x);
    at::Tensor preGate = at::empty(x.sizes(), x.options().dtype(at::kHalf));
    at::Tensor finalStateInternal = at::empty(
        stateNpLayout
            ? at::IntArrayRef({batch, heads, stateDim, headDim})
            : at::IntArrayRef({batch, heads, headDim, stateDim}),
        chunkStates.options());
    at::Tensor statesStartInternal = at::empty(
        chunkStates.sizes(), chunkStates.options().dtype(at::kHalf));
    at::Tensor initialInternal = initialStates.has_value()
        ? (stateNpLayout
               ? initialStates->transpose(-1, -2).contiguous()
               : initialStates.value())
        : chunkStates;
    EXEC_NPU_CMD(aclnnMamba2SsdStateEpilogueTrainStates,
                 chunkStates, dACumsum, cCube, yDiag, x, d, z,
                 initialInternal, out, finalStateInternal, preGate,
                 statesStartInternal);

    at::Tensor finalState = stateNpLayout
        ? finalStateInternal.transpose(-1, -2)
        : finalStateInternal;
    at::Tensor statesStart = groupedInput
        ? statesStartInternal
        : (stateNpLayout
               ? statesStartInternal.transpose(-1, -2).contiguous()
               : statesStartInternal);
    return {out, finalState, preGate, statesStart};
}

} // namespace ascend_kernel
