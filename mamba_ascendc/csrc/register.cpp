// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {
TORCH_LIBRARY_FRAGMENT(mamba_ascend, m)
{
    m.def(
        "mamba2_arch35_matmul_smoke_internal_test_only(Tensor weight, "
        "Tensor x_group) -> Tensor");
    m.def(
        "mamba2_ssd_chunk_mix(Tensor x_cube, Tensor d_a_cumsum, "
        "Tensor b_cube, Tensor c_cube) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_chunk_mix_grouped(Tensor x_cube, Tensor d_a_cumsum, "
        "Tensor b_cube, Tensor c_cube) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_chunk_scan_bwd_off(Tensor gy, Tensor states_start, "
        "Tensor d_a_cumsum, Tensor c_cube) -> (Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_bwd_off_group_reduce(Tensor q_group, "
        "Tensor state_group, Tensor c_cube) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_chunk_scan_bwd_diag_state(Tensor gy, Tensor x_cube, "
        "Tensor d_a_cumsum, Tensor b_cube, Tensor c_cube, "
        "Tensor d_chunk_states) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_off_epilogue(Tensor c_cube, Tensor states_start, "
        "Tensor d_a_cumsum, Tensor y_diag, Tensor x, Tensor D, Tensor z) "
        "-> Tensor");
    m.def(
        "mamba2_ssd_state_projection(Tensor states_start, Tensor c_cube) "
        "-> Tensor");
    m.def(
        "mamba2_ssd_state_projection_grouped(Tensor states_grouped, "
        "Tensor c_cube) -> Tensor");
    m.def(
        "mamba2_ssd_state_vector_epilogue(Tensor y_off, "
        "Tensor d_a_cumsum, Tensor y_diag, Tensor x, Tensor D, Tensor z) "
        "-> Tensor");
    m.def(
        "mamba2_ssd_state_vector_epilogue_grouped(Tensor y_grouped, "
        "Tensor d_a_cumsum, Tensor y_diag, Tensor x, Tensor D, Tensor z) "
        "-> Tensor");
    m.def(
        "mamba2_ssd_state_vector_epilogue_grouped_train(Tensor y_grouped, "
        "Tensor d_a_cumsum, Tensor y_diag, Tensor x, Tensor D, Tensor z) "
        "-> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_state_epilogue(Tensor chunk_states, Tensor d_a_cumsum, "
        "Tensor c_cube, Tensor y_diag, Tensor x, Tensor D, Tensor z, "
        "Tensor? initial_states=None) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_state_epilogue_train(Tensor chunk_states, "
        "Tensor d_a_cumsum, Tensor c_cube, Tensor y_diag, Tensor x, "
        "Tensor D, Tensor z, Tensor? initial_states=None) "
        "-> (Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_state_epilogue_train_states(Tensor chunk_states, "
        "Tensor d_a_cumsum, Tensor c_cube, Tensor y_diag, Tensor x, "
        "Tensor D, Tensor z, Tensor? initial_states=None) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_preprocess(Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, "
        "Tensor? dt_bias=None, int chunk_size=256, bool dt_softplus=False, "
        "float dt_limit_min=0.0, float dt_limit_max=3.402823466e+38) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_preprocess_grouped(Tensor x, Tensor dt, Tensor A, "
        "Tensor B, Tensor C, Tensor? dt_bias=None, int chunk_size=64, "
        "bool dt_softplus=False, float dt_limit_min=0.0, "
        "float dt_limit_max=3.402823466e+38) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_state_passing(Tensor chunk_states, Tensor dA_cumsum, "
        "Tensor? initial_states=None) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_state_passing_grouped(Tensor chunk_states_np, "
        "Tensor dA_cumsum, Tensor? initial_states_np=None, int groups=1) "
        "-> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_state_passing_bwd(Tensor states_start, "
        "Tensor d_states_start, Tensor dA_cumsum, "
        "Tensor? dfinal_state=None) -> (Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_state_passing_bwd_half(Tensor states_start, "
        "Tensor d_states_start, Tensor dA_cumsum, "
        "Tensor? dfinal_state=None) -> (Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_dt_bwd(Tensor x, Tensor d_xdt_total, "
        "Tensor g_dA_cs_total, Tensor dt, Tensor A, Tensor? dt_bias=None, "
        "bool dt_softplus=False, float dt_limit_min=0.0, "
        "float dt_limit_max=3.402823466e+38) -> "
        "(Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_bwd_gate(Tensor dout, Tensor z, Tensor y_pre, Tensor x) "
        "-> (Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_bwd_gate_nodd(Tensor dout, Tensor z, Tensor y_pre) "
        "-> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_bwd_diag_finalize(Tensor dx_diag, Tensor d_r, "
        "Tensor d_b_diag, Tensor d_b_state, Tensor d_c_diag, Tensor d_w, "
        "Tensor w, Tensor r, Tensor d_a_cumsum, int groups, "
        "Tensor? d_c_off=None) -> "
        "(Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_bwd_off_prepare(Tensor gy, Tensor states_start, "
        "Tensor d_a_cumsum, int groups=1) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_bwd_off_finalize(Tensor d_states_half, "
        "Tensor d_c_head_half, Tensor c_cube) -> "
        "(Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_bwd_off_finalize_grouped(Tensor d_states_half, "
        "Tensor d_c_group_half, Tensor q_group_half, "
        "Tensor y_base_group_half) -> (Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_dt_bwd_d(Tensor x, Tensor d_xdt_total, "
        "Tensor g_dA_cs_total, Tensor dt, Tensor A, Tensor gy_head, Tensor D, "
        "Tensor? dt_bias=None, bool dt_softplus=False, "
        "float dt_limit_min=0.0, float dt_limit_max=3.402823466e+38) -> "
        "(Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_dt_bwd_d_dd(Tensor x, Tensor d_xdt_total, "
        "Tensor g_dA_cs_total, Tensor dt, Tensor A, Tensor gy_head, Tensor D, "
        "Tensor? dt_bias=None, bool dt_softplus=False, "
        "float dt_limit_min=0.0, float dt_limit_max=3.402823466e+38) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_dt_bwd_grouped_d_dd(Tensor x, Tensor d_xdt_total, "
        "Tensor g_dA_cs_diag, Tensor g_dA_cs_off_grouped, "
        "Tensor g_dA_cs_chunk_last, Tensor dt, Tensor A, Tensor gy_head, "
        "Tensor D, Tensor? dt_bias=None, bool dt_softplus=False, "
        "float dt_limit_min=0.0, float dt_limit_max=3.402823466e+38) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_dt_bwd_fused_gcs_d_dd(Tensor x, Tensor d_xdt_total, "
        "Tensor g_dA_cs_diag, Tensor g_dA_cs_off, "
        "Tensor g_dA_cs_chunk_last, Tensor dt, Tensor A, Tensor gy_head, "
        "Tensor D, Tensor? dt_bias=None, bool dt_softplus=False, "
        "float dt_limit_min=0.0, float dt_limit_max=3.402823466e+38) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_dt_bwd_chunk_tail_d_dd(Tensor x, Tensor d_xdt_total, "
        "Tensor g_dA_cs_total, Tensor g_dA_cs_chunk_last, Tensor dt, "
        "Tensor A, Tensor gy_head, Tensor D, Tensor? dt_bias=None, "
        "bool dt_softplus=False, float dt_limit_min=0.0, "
        "float dt_limit_max=3.402823466e+38) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_prepare(Tensor cb, Tensor dA_cumsum, Tensor x_cube) "
        "-> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_prepare_split(Tensor cb, Tensor dA_cumsum, Tensor x_cube) "
        "-> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_prepare_dcb(Tensor d_w, Tensor d_a_cumsum, int groups) "
        "-> Tensor");
    m.def(
        "mamba2_ssd_fwd(Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, "
        "Tensor? D=None, Tensor? z=None, Tensor? dt_bias=None, "
        "Tensor? initial_states=None, int chunk_size=256, "
        "bool dt_softplus=False, float dt_limit_min=0.0, "
        "float dt_limit_max=3.402823466e+38) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_bwd(Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, "
        "Tensor dout, Tensor final_state, Tensor? D=None, Tensor? z=None, "
        "Tensor? dt_bias=None, Tensor? initial_states=None, "
        "Tensor? dfinal_state=None, bool dt_softplus=False, "
        "float dt_limit_min=0.0, float dt_limit_max=3.402823466e+38) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mamba_ascend, PrivateUse1, m)
{
    m.impl("mamba2_arch35_matmul_smoke_internal_test_only",
           TORCH_FN(ascend_kernel::mamba2_arch35_matmul_smoke_internal_test_only));
    m.impl("mamba2_ssd_chunk_mix", TORCH_FN(ascend_kernel::mamba2_ssd_chunk_mix));
    m.impl("mamba2_ssd_chunk_mix_grouped",
           TORCH_FN(ascend_kernel::mamba2_ssd_chunk_mix_grouped));
    m.impl("mamba2_ssd_chunk_scan_bwd_off",
           TORCH_FN(ascend_kernel::mamba2_ssd_chunk_scan_bwd_off));
    m.impl("mamba2_ssd_bwd_off_group_reduce",
           TORCH_FN(ascend_kernel::mamba2_ssd_bwd_off_group_reduce));
    m.impl("mamba2_ssd_chunk_scan_bwd_diag_state",
           TORCH_FN(ascend_kernel::mamba2_ssd_chunk_scan_bwd_diag_state));
    m.impl("mamba2_ssd_off_epilogue", TORCH_FN(ascend_kernel::mamba2_ssd_off_epilogue));
    m.impl("mamba2_ssd_state_projection",
           TORCH_FN(ascend_kernel::mamba2_ssd_state_projection));
    m.impl("mamba2_ssd_state_projection_grouped",
           TORCH_FN(ascend_kernel::mamba2_ssd_state_projection_grouped));
    m.impl("mamba2_ssd_state_vector_epilogue",
           TORCH_FN(ascend_kernel::mamba2_ssd_state_vector_epilogue));
    m.impl("mamba2_ssd_state_vector_epilogue_grouped",
           TORCH_FN(ascend_kernel::mamba2_ssd_state_vector_epilogue_grouped));
    m.impl(
        "mamba2_ssd_state_vector_epilogue_grouped_train",
        TORCH_FN(
            ascend_kernel::mamba2_ssd_state_vector_epilogue_grouped_train));
    m.impl("mamba2_ssd_state_epilogue", TORCH_FN(ascend_kernel::mamba2_ssd_state_epilogue));
    m.impl("mamba2_ssd_state_epilogue_train",
           TORCH_FN(ascend_kernel::mamba2_ssd_state_epilogue_train));
    m.impl("mamba2_ssd_state_epilogue_train_states",
           TORCH_FN(ascend_kernel::mamba2_ssd_state_epilogue_train_states));
    m.impl("mamba2_ssd_preprocess", TORCH_FN(ascend_kernel::mamba2_ssd_preprocess));
    m.impl("mamba2_ssd_preprocess_grouped",
           TORCH_FN(ascend_kernel::mamba2_ssd_preprocess_grouped));
    m.impl("mamba2_ssd_state_passing", TORCH_FN(ascend_kernel::mamba2_ssd_state_passing));
    m.impl("mamba2_ssd_state_passing_grouped",
           TORCH_FN(ascend_kernel::mamba2_ssd_state_passing_grouped));
    m.impl("mamba2_ssd_state_passing_bwd", TORCH_FN(ascend_kernel::mamba2_ssd_state_passing_bwd));
    m.impl("mamba2_ssd_state_passing_bwd_half",
           TORCH_FN(ascend_kernel::mamba2_ssd_state_passing_bwd_half));
    m.impl("mamba2_ssd_dt_bwd", TORCH_FN(ascend_kernel::mamba2_ssd_dt_bwd));
    m.impl("mamba2_ssd_bwd_gate", TORCH_FN(ascend_kernel::mamba2_ssd_bwd_gate));
    m.impl("mamba2_ssd_bwd_gate_nodd",
           TORCH_FN(ascend_kernel::mamba2_ssd_bwd_gate_nodd));
    m.impl("mamba2_ssd_bwd_diag_finalize",
           TORCH_FN(ascend_kernel::mamba2_ssd_bwd_diag_finalize));
    m.impl("mamba2_ssd_bwd_off_prepare",
           TORCH_FN(ascend_kernel::mamba2_ssd_bwd_off_prepare));
    m.impl("mamba2_ssd_bwd_off_finalize",
           TORCH_FN(ascend_kernel::mamba2_ssd_bwd_off_finalize));
    m.impl("mamba2_ssd_bwd_off_finalize_grouped",
           TORCH_FN(ascend_kernel::mamba2_ssd_bwd_off_finalize_grouped));
    m.impl("mamba2_ssd_dt_bwd_d", TORCH_FN(ascend_kernel::mamba2_ssd_dt_bwd_d));
    m.impl("mamba2_ssd_dt_bwd_d_dd",
           TORCH_FN(ascend_kernel::mamba2_ssd_dt_bwd_d_dd));
    m.impl("mamba2_ssd_dt_bwd_grouped_d_dd",
           TORCH_FN(ascend_kernel::mamba2_ssd_dt_bwd_grouped_d_dd));
    m.impl("mamba2_ssd_dt_bwd_fused_gcs_d_dd",
           TORCH_FN(ascend_kernel::mamba2_ssd_dt_bwd_fused_gcs_d_dd));
    m.impl("mamba2_ssd_dt_bwd_chunk_tail_d_dd",
           TORCH_FN(ascend_kernel::mamba2_ssd_dt_bwd_chunk_tail_d_dd));
    m.impl("mamba2_ssd_prepare", TORCH_FN(ascend_kernel::mamba2_ssd_prepare));
    m.impl("mamba2_ssd_prepare_split",
           TORCH_FN(ascend_kernel::mamba2_ssd_prepare_split));
    m.impl("mamba2_ssd_prepare_dcb",
           TORCH_FN(ascend_kernel::mamba2_ssd_prepare_dcb));
    m.impl("mamba2_ssd_fwd", TORCH_FN(ascend_kernel::mamba2_ssd_fwd));
    m.impl("mamba2_ssd_bwd", TORCH_FN(ascend_kernel::mamba2_ssd_bwd));
}
}  // namespace
