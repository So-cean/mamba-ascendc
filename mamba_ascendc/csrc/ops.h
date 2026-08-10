// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPS_H
#define OPS_H

#include <tuple>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

namespace ascend_kernel {

at::Tensor mamba2_arch35_matmul_smoke_internal_test_only(
    const at::Tensor &a,
    const at::Tensor &b);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_chunk_mix(
    const at::Tensor &xCube,
    const at::Tensor &dACumsum,
    const at::Tensor &bCube,
    const at::Tensor &cCube);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_chunk_mix_grouped(
    const at::Tensor &xCube,
    const at::Tensor &dACumsum,
    const at::Tensor &bCube,
    const at::Tensor &cCube);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_chunk_scan_bwd_off(
    const at::Tensor &gy,
    const at::Tensor &statesStart,
    const at::Tensor &dACumsum,
    const at::Tensor &cCube);

std::tuple<at::Tensor, at::Tensor>
mamba2_ssd_bwd_off_group_reduce(
    const at::Tensor &qGroup,
    const at::Tensor &stateGroup,
    const at::Tensor &cCube);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_chunk_scan_bwd_diag_state(
    const at::Tensor &gy,
    const at::Tensor &xCube,
    const at::Tensor &dACumsum,
    const at::Tensor &bCube,
    const at::Tensor &cCube,
    const at::Tensor &dChunkStates);

at::Tensor mamba2_ssd_off_epilogue(
    const at::Tensor &cCube,
    const at::Tensor &statesStart,
    const at::Tensor &dACumsum,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z);

at::Tensor mamba2_ssd_state_projection(
    const at::Tensor &statesStart,
    const at::Tensor &cCube);

at::Tensor mamba2_ssd_state_projection_grouped(
    const at::Tensor &statesGrouped,
    const at::Tensor &cCube);

at::Tensor mamba2_ssd_state_vector_epilogue(
    const at::Tensor &yOff,
    const at::Tensor &dACumsum,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z);

at::Tensor mamba2_ssd_state_vector_epilogue_grouped(
    const at::Tensor &yGrouped,
    const at::Tensor &dACumsum,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z);

std::tuple<at::Tensor, at::Tensor>
mamba2_ssd_state_vector_epilogue_grouped_train(
    const at::Tensor &yGrouped,
    const at::Tensor &dACumsum,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_state_epilogue(
    const at::Tensor &chunkStates,
    const at::Tensor &dACumsum,
    const at::Tensor &cCube,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z,
    const c10::optional<at::Tensor> &initialStates);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_state_epilogue_train(
    const at::Tensor &chunkStates,
    const at::Tensor &dACumsum,
    const at::Tensor &cCube,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z,
    const c10::optional<at::Tensor> &initialStates);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_state_epilogue_train_states(
    const at::Tensor &chunkStates,
    const at::Tensor &dACumsum,
    const at::Tensor &cCube,
    const at::Tensor &yDiag,
    const at::Tensor &x,
    const at::Tensor &d,
    const at::Tensor &z,
    const c10::optional<at::Tensor> &initialStates);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_preprocess(
    const at::Tensor &x,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const c10::optional<at::Tensor> &dt_bias,
    int64_t chunk_size,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_preprocess_grouped(
    const at::Tensor &x,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const c10::optional<at::Tensor> &dt_bias,
    int64_t chunk_size,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_state_passing(
    const at::Tensor &chunk_states,
    const at::Tensor &dA_cumsum,
    const c10::optional<at::Tensor> &initial_states);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_state_passing_grouped(
    const at::Tensor &chunk_states_np,
    const at::Tensor &dA_cumsum,
    const c10::optional<at::Tensor> &initial_states_np,
    int64_t groups);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_state_passing_bwd(
    const at::Tensor &states_start,
    const at::Tensor &d_states_start,
    const at::Tensor &dA_cumsum,
    const c10::optional<at::Tensor> &dfinal_state);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_state_passing_bwd_half(
    const at::Tensor &states_start,
    const at::Tensor &d_states_start,
    const at::Tensor &dA_cumsum,
    const c10::optional<at::Tensor> &dfinal_state);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_dt_bwd(
    const at::Tensor &x,
    const at::Tensor &d_xdt_total,
    const at::Tensor &g_dA_cs_total,
    const at::Tensor &dt,
    const at::Tensor &A,
    const c10::optional<at::Tensor> &dt_bias,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_bwd_gate(
    const at::Tensor &dout,
    const at::Tensor &z,
    const at::Tensor &y_pre,
    const at::Tensor &x);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_bwd_gate_nodd(
    const at::Tensor &dout,
    const at::Tensor &z,
    const at::Tensor &y_pre);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_bwd_diag_finalize(
    const at::Tensor &dx_diag,
    const at::Tensor &d_r,
    const at::Tensor &d_b_diag,
    const at::Tensor &d_b_state,
    const at::Tensor &d_c_diag,
    const at::Tensor &d_w,
    const at::Tensor &w,
    const at::Tensor &r,
    const at::Tensor &d_a_cumsum,
    int64_t groups,
    const c10::optional<at::Tensor> &d_c_off = c10::nullopt);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_bwd_off_prepare(
    const at::Tensor &gy,
    const at::Tensor &states_start,
    const at::Tensor &d_a_cumsum,
    int64_t groups);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_bwd_off_finalize(
    const at::Tensor &d_states_half,
    const at::Tensor &d_c_head_half,
    const at::Tensor &c_cube);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_bwd_off_finalize_grouped(
    const at::Tensor &d_states_half,
    const at::Tensor &d_c_group_half,
    const at::Tensor &q_group_half,
    const at::Tensor &y_base_group_half);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mamba2_ssd_dt_bwd_d(
    const at::Tensor &x,
    const at::Tensor &d_xdt_total,
    const at::Tensor &g_dA_cs_total,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gy_head,
    const at::Tensor &D,
    const c10::optional<at::Tensor> &dt_bias,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_dt_bwd_d_dd(
    const at::Tensor &x,
    const at::Tensor &d_xdt_total,
    const at::Tensor &g_dA_cs_total,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gy_head,
    const at::Tensor &D,
    const c10::optional<at::Tensor> &dt_bias,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_dt_bwd_grouped_d_dd(
    const at::Tensor &x,
    const at::Tensor &d_xdt_total,
    const at::Tensor &g_dA_cs_diag,
    const at::Tensor &g_dA_cs_off_grouped,
    const at::Tensor &g_dA_cs_chunk_last,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gy_head,
    const at::Tensor &D,
    const c10::optional<at::Tensor> &dt_bias,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_dt_bwd_fused_gcs_d_dd(
    const at::Tensor &x,
    const at::Tensor &d_xdt_total,
    const at::Tensor &g_dA_cs_diag,
    const at::Tensor &g_dA_cs_off,
    const at::Tensor &g_dA_cs_chunk_last,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gy_head,
    const at::Tensor &D,
    const c10::optional<at::Tensor> &dt_bias,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_dt_bwd_chunk_tail_d_dd(
    const at::Tensor &x,
    const at::Tensor &d_xdt_total,
    const at::Tensor &g_dA_cs_total,
    const at::Tensor &g_dA_cs_chunk_last,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &gy_head,
    const at::Tensor &D,
    const c10::optional<at::Tensor> &dt_bias,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_prepare(
    const at::Tensor &cb,
    const at::Tensor &dA_cumsum,
    const at::Tensor &x_cube);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_prepare_split(
    const at::Tensor &cb,
    const at::Tensor &dA_cumsum,
    const at::Tensor &x_cube);

at::Tensor mamba2_ssd_prepare_dcb(
    const at::Tensor &d_w,
    const at::Tensor &d_a_cumsum,
    int64_t groups);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_fwd(
    const at::Tensor &x,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const c10::optional<at::Tensor> &D,
    const c10::optional<at::Tensor> &z,
    const c10::optional<at::Tensor> &dt_bias,
    const c10::optional<at::Tensor> &initial_states,
    int64_t chunk_size,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_bwd(
    const at::Tensor &x,
    const at::Tensor &dt,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &dout,
    const at::Tensor &final_state,
    const c10::optional<at::Tensor> &D,
    const c10::optional<at::Tensor> &z,
    const c10::optional<at::Tensor> &dt_bias,
    const c10::optional<at::Tensor> &initial_states,
    const c10::optional<at::Tensor> &dfinal_state,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

} // namespace ascend_kernel

#endif // OPS_H
