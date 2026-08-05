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

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_chunk_mix(
    const at::Tensor &xCube,
    const at::Tensor &dACumsum,
    const at::Tensor &bCube,
    const at::Tensor &cCube);

at::Tensor mamba2_ssd_off_epilogue(
    const at::Tensor &cCube,
    const at::Tensor &statesStart,
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

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_state_passing(
    const at::Tensor &chunk_states,
    const at::Tensor &dA_cumsum,
    const c10::optional<at::Tensor> &initial_states);

std::tuple<at::Tensor, at::Tensor> mamba2_ssd_prepare(
    const at::Tensor &cb,
    const at::Tensor &dA_cumsum,
    const at::Tensor &x_cube);

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

} // namespace ascend_kernel

#endif // OPS_H
