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
        "mamba2_ssd_chunk_mix(Tensor x_cube, Tensor d_a_cumsum, "
        "Tensor b_cube, Tensor c_cube) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_off_epilogue(Tensor c_cube, Tensor states_start, "
        "Tensor d_a_cumsum, Tensor y_diag, Tensor x, Tensor D, Tensor z) "
        "-> Tensor");
    m.def(
        "mamba2_ssd_state_epilogue(Tensor chunk_states, Tensor d_a_cumsum, "
        "Tensor c_cube, Tensor y_diag, Tensor x, Tensor D, Tensor z, "
        "Tensor? initial_states=None) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_preprocess(Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, "
        "Tensor? dt_bias=None, int chunk_size=256, bool dt_softplus=False, "
        "float dt_limit_min=0.0, float dt_limit_max=3.402823466e+38) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "mamba2_ssd_state_passing(Tensor chunk_states, Tensor dA_cumsum, "
        "Tensor? initial_states=None) -> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_prepare(Tensor cb, Tensor dA_cumsum, Tensor x_cube) "
        "-> (Tensor, Tensor)");
    m.def(
        "mamba2_ssd_fwd(Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, "
        "Tensor? D=None, Tensor? z=None, Tensor? dt_bias=None, "
        "Tensor? initial_states=None, int chunk_size=256, "
        "bool dt_softplus=False, float dt_limit_min=0.0, "
        "float dt_limit_max=3.402823466e+38) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mamba_ascend, PrivateUse1, m)
{
    m.impl("mamba2_ssd_chunk_mix", TORCH_FN(ascend_kernel::mamba2_ssd_chunk_mix));
    m.impl("mamba2_ssd_off_epilogue", TORCH_FN(ascend_kernel::mamba2_ssd_off_epilogue));
    m.impl("mamba2_ssd_state_epilogue", TORCH_FN(ascend_kernel::mamba2_ssd_state_epilogue));
    m.impl("mamba2_ssd_preprocess", TORCH_FN(ascend_kernel::mamba2_ssd_preprocess));
    m.impl("mamba2_ssd_state_passing", TORCH_FN(ascend_kernel::mamba2_ssd_state_passing));
    m.impl("mamba2_ssd_prepare", TORCH_FN(ascend_kernel::mamba2_ssd_prepare));
    m.impl("mamba2_ssd_fwd", TORCH_FN(ascend_kernel::mamba2_ssd_fwd));
}
}  // namespace
