/**
 * Training variant of StateEpilogue.  It writes the pre-SiLU value already
 * resident in UB so backward can reuse it instead of recomputing C@state.
 */

#define MAMBA2_STATE_EPILOGUE_COMPONENT_ONLY
#include "mamba2_ssd_state_epilogue.cpp"

extern "C" __global__ __aicore__ void mamba2_ssd_state_epilogue_train(
    GM_ADDR chunk_states, GM_ADDR d_a_cumsum, GM_ADDR c_cube,
    GM_ADDR y_diag, GM_ADDR x, GM_ADDR d, GM_ADDR z,
    GM_ADDR initial_states, GM_ADDR out, GM_ADDR final_state,
    GM_ADDR pre_gate, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(14, KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(15, KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(4)) {
        KernelMamba2SsdStateEpilogueT<64, true> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, pre_gate, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(5)) {
        KernelMamba2SsdStateEpilogueT<
            64, true, false, LegacyStateProjectionHalfMatmul, half> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, pre_gate, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(14)) {
        KernelMamba2SsdStateEpilogueT<64, true, false,
                                      Arch35StateProjectionMatmul> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, pre_gate, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(15)) {
        KernelMamba2SsdStateEpilogueT<
            64, true, false, Arch35GroupedStateProjectionMatmul, half> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, pre_gate, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(8)) {
        KernelMamba2SsdStateEpilogueT<128, true> op;
        op.Init(chunk_states, d_a_cumsum, c_cube, y_diag, x, d, z,
                initial_states, out, pre_gate, final_state, out, workspace,
                tilingData, &pipe);
        op.Process();
    }
}
