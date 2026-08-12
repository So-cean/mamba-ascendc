"""PyTorch-facing Mamba-2 SSD forward API for the AscendC extension."""

import os

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


_ADDITIVE_CAUSAL_MASKS = {}
_NPU_DEVICE_NAMES = {}


def _lower_segsum(cumsum: torch.Tensor) -> torch.Tensor:
    length = cumsum.shape[-1]
    values = cumsum.unsqueeze(-1) - cumsum.unsqueeze(-2)
    key = (str(cumsum.device), length, cumsum.dtype)
    mask = _ADDITIVE_CAUSAL_MASKS.get(key)
    if mask is None:
        valid = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=cumsum.device)
        )
        mask = torch.zeros(length, length, dtype=cumsum.dtype, device=cumsum.device)
        mask.masked_fill_(~valid, -torch.inf)
        _ADDITIVE_CAUSAL_MASKS[key] = mask
    return values + mask


def _can_use_aligned_path(x, B, chunk_size):
    if chunk_size <= 0:
        return False
    if x.dtype != torch.float32 or B.dtype != torch.float32:
        return False
    if x.ndim != 4 or B.ndim != 4 or x.shape[1] % chunk_size:
        return False
    _, _, _, headdim = x.shape
    dstate = B.shape[-1]
    return (
        chunk_size % 16 == 0
        and headdim % 16 == 0
        and dstate % 16 == 0
        and headdim * dstate <= 8192
    )


def _chunk_mix_enabled() -> bool:
    """Return whether the separately packaged Cube/MIX OPS path is available."""
    return os.environ.get("MAMBA_ASCENDC_CHUNK_MIX", "0").lower() in {
        "1", "true", "yes", "on"
    }


def _is_ascend950(x: torch.Tensor) -> bool:
    """Return whether ``x`` resides on an Ascend 950-family device."""
    if x.device.type != "npu":
        return False
    device_index = x.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    name = _NPU_DEVICE_NAMES.get(device_index)
    if name is None:
        name = torch.npu.get_device_name(device_index)
        _NPU_DEVICE_NAMES[device_index] = name
    return "950" in name.upper()


def _can_use_grouped_path(x, B, chunk_size, D, z):
    if not _chunk_mix_enabled() or not _is_ascend950(x):
        return False
    if not _can_use_aligned_path(x, B, chunk_size):
        return False
    batch, seqlen, heads, headdim = x.shape
    groups = B.shape[2]
    nchunks = seqlen // chunk_size
    return (
        chunk_size == 64
        and headdim == 64
        and B.shape[-1] == 64
        and heads == 4 * groups
        and batch * groups * nchunks >= 64
        and D is not None
        and D.ndim == 2
        and tuple(D.shape) == (heads, headdim)
        and z is not None
        and tuple(z.shape) == tuple(x.shape)
    )


def _grouped_ssd_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    z,
    dt_bias,
    dt_softplus,
    dt_limit,
    initial_states,
    return_pre_gate=False,
    return_states_start=False,
    return_preprocess_cache=False,
):
    """Run the group-contiguous state/projection pipeline."""
    groups = B.shape[2]
    x_cube, d_a_cs, b_cube, c_cube = (
        torch.ops.mamba_ascend.mamba2_ssd_preprocess(
            x,
            dt,
            A,
            B,
            C,
            dt_bias,
            chunk_size,
            dt_softplus,
            float(dt_limit[0]),
            float(dt_limit[1]),
        )
    )
    if _is_ascend950(x):
        y_diag, chunk_states = (
            torch.ops.mamba_ascend.mamba2_ssd_chunk_mix_grouped(
                x_cube, d_a_cs, b_cube, c_cube
            )
        )
    else:
        # 910B3 keeps its validated 1AIC:2AIV ChunkMix producer.  The grouped
        # recurrence accepts this [B,H,K,N,P] tensor directly and converts
        # only once at the producer/consumer boundary.
        y_diag, chunk_states = torch.ops.mamba_ascend.mamba2_ssd_chunk_mix(
            x_cube, d_a_cs, b_cube, c_cube
        )
    initial_np = (
        None
        if initial_states is None
        else initial_states.transpose(-1, -2).contiguous()
    )
    states_grouped, final_np = (
        torch.ops.mamba_ascend.mamba2_ssd_state_passing_grouped(
            chunk_states, d_a_cs, initial_np, groups
        )
    )
    y_grouped = torch.ops.mamba_ascend.mamba2_ssd_state_projection_grouped(
        states_grouped, c_cube
    )
    if return_pre_gate:
        out, pre_gate = (
            torch.ops.mamba_ascend
            .mamba2_ssd_state_vector_epilogue_grouped_train(
                y_grouped, d_a_cs, y_diag, x, D, z
            )
        )
    else:
        out = torch.ops.mamba_ascend.mamba2_ssd_state_vector_epilogue_grouped(
            y_grouped, d_a_cs, y_diag, x, D, z
        )
        pre_gate = None
    result = [out, final_np.transpose(-1, -2).contiguous()]
    if return_pre_gate:
        result.append(pre_gate)
    if return_states_start:
        result.append(states_grouped)
    if return_preprocess_cache:
        result.extend((x_cube, d_a_cs, b_cube, c_cube))
    return tuple(result)


def _can_use_chunk_mix_path(x, B, chunk_size):
    if not _chunk_mix_enabled() or not _can_use_aligned_path(x, B, chunk_size):
        return False
    # The legacy key-9 public state/epilogue layout is not numerically valid
    # on 950PR.  Supported 950 inference shapes use the grouped pipeline above.
    if _is_ascend950(x):
        return False
    return (
        x.shape[-1] == 64
        and B.shape[-1] in (64, 128)
        and x.shape[1] % 64 == 0
        and chunk_size in (64, 128)
    )


def _select_execution_chunk_size(x, B, logical_chunk_size):
    """Map a public SSD partition to a supported Cube/MIX micro-tile.

    ``chunk_size`` is an algorithmic partition parameter, not a tensor-shape
    dimension.  The exact SSD recurrence is partition invariant, so inference
    may execute a logical chunk as multiple smaller micro-chunks.  Keep the
    logical value when no validated Cube/MIX tiling divides it; that preserves
    the existing aligned/generic fallback behavior.
    """
    if logical_chunk_size <= 128:
        return logical_chunk_size

    chunk128_enabled = os.environ.get(
        "MAMBA_ASCENDC_CHUNK128", "0"
    ).lower() in {"1", "true", "yes", "on"}
    candidates = []
    if chunk128_enabled and B.shape[-1] == 128:
        candidates.append(128)
    candidates.append(64)
    for micro_chunk in candidates:
        if (
            logical_chunk_size % micro_chunk == 0
            and _can_use_chunk_mix_path(x, B, micro_chunk)
        ):
            return micro_chunk
    return logical_chunk_size


def _chunk_mix_ssd_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    z,
    dt_bias,
    dt_softplus,
    dt_limit,
    initial_states,
    return_pre_gate=False,
    return_states_start=False,
    return_preprocess_cache=False,
):
    """Production OPS path with 64/128-token Cube+Vector microchunks."""
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2:]
    chunk128_enabled = os.environ.get(
        "MAMBA_ASCENDC_CHUNK128", "0"
    ).lower() in {"1", "true", "yes", "on"}
    micro_chunk = (
        128 if chunk128_enabled and chunk_size == 128 and dstate == 128
        else 64
    )
    nchunks = seqlen // micro_chunk
    heads_per_group = nheads // ngroups

    native_preprocess = os.environ.get(
        "MAMBA_ASCENDC_NATIVE_PREPROCESS", "0"
    ).lower() in {"1", "true", "yes", "on"}
    # The vectorized custom path is faster at tiny, medium and extreme gates;
    # retain a native override only for controlled A/B experiments.
    use_fused_preprocess = not native_preprocess
    if use_fused_preprocess:
        x_cube, dA_cs, B_g_cube, C_g_cube = (
            torch.ops.mamba_ascend.mamba2_ssd_preprocess(
                x,
                dt,
                A,
                B,
                C,
                dt_bias,
                micro_chunk,
                dt_softplus,
                float(dt_limit[0]),
                float(dt_limit[1]),
            )
        )
    else:
        dt_f = dt.float()
        if dt_bias is not None:
            dt_f = dt_f + dt_bias.float().view(1, 1, nheads)
        if dt_softplus:
            dt_f = F.softplus(dt_f)
        dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])
        dA = dt_f * A.float().view(1, 1, nheads)
        dA_cs = dA.reshape(
            batch, nchunks, micro_chunk, nheads
        ).permute(0, 3, 1, 2).cumsum(-1)
        x_cube = (x.float() * dt_f.unsqueeze(-1)).reshape(
            batch, nchunks, micro_chunk, nheads, headdim
        ).permute(0, 3, 1, 2, 4).contiguous().to(torch.float16)
        B_g_cube = B.float().reshape(
            batch, nchunks, micro_chunk, ngroups, dstate
        ).permute(0, 1, 3, 4, 2).contiguous().to(torch.float16)
        C_g_cube = C.float().reshape(
            batch, nchunks, micro_chunk, ngroups, dstate
        ).permute(0, 1, 3, 2, 4).contiguous().to(torch.float16)

    grouped_state_910 = (
        os.environ.get("MAMBA_ASCENDC_GROUPED_STATE_910", "0").lower()
        in {"1", "true", "yes", "on"}
        and not _is_ascend950(x)
        and micro_chunk == 64
        and dstate == 64
        and heads_per_group == 4
    )
    chunk_mix_op = (
        torch.ops.mamba_ascend.mamba2_ssd_chunk_mix_grouped
        if grouped_state_910
        else torch.ops.mamba_ascend.mamba2_ssd_chunk_mix
    )
    y_diag, chunk_states = chunk_mix_op(
        x_cube, dA_cs, B_g_cube, C_g_cube
    )

    force_state_epilogue = os.environ.get(
        "MAMBA_ASCENDC_FUSED_STATE_EPILOGUE", "0"
    ).lower() in {"1", "true", "yes", "on"}
    # Chunk-128 keeps the recurrent state in Cube-native [N, P] layout so the
    # B^T @ X and C @ state matmuls can share it without a Vector transpose.
    state_np_layout = (
        (micro_chunk == 64 and dstate == 64) or micro_chunk == 128
    )
    use_state_epilogue = (
        D is not None
        and D.ndim == 2
        and z is not None
        and (
            force_state_epilogue
            or state_np_layout
            or batch * nheads * nchunks >= 4096
        )
    )
    if use_state_epilogue:
        if return_states_start:
            train_result = (
                torch.ops.mamba_ascend.mamba2_ssd_state_epilogue_train_states(
                    chunk_states,
                    dA_cs,
                    C_g_cube,
                    y_diag,
                    x,
                    D,
                    z,
                    initial_states,
                )
            )
            result = train_result if return_pre_gate else (
                train_result[0], train_result[1], train_result[3]
            )
        elif return_pre_gate:
            result = torch.ops.mamba_ascend.mamba2_ssd_state_epilogue_train(
                chunk_states,
                dA_cs,
                C_g_cube,
                y_diag,
                x,
                D,
                z,
                initial_states,
            )
        else:
            result = torch.ops.mamba_ascend.mamba2_ssd_state_epilogue(
                chunk_states,
                dA_cs,
                C_g_cube,
                y_diag,
                x,
                D,
                z,
                initial_states,
            )
        if return_preprocess_cache:
            return (*result, x_cube, dA_cs, B_g_cube, C_g_cube)
        return result

    if state_np_layout:
        chunk_states = chunk_states.transpose(-1, -2).contiguous()
    states_start, final_state = torch.ops.mamba_ascend.mamba2_ssd_state_passing(
        chunk_states, dA_cs, initial_states
    )

    force_fused_epilogue = os.environ.get(
        "MAMBA_ASCENDC_FUSED_EPILOGUE", "0"
    ).lower() in {"1", "true", "yes", "on"}
    # Scaling gate on 910B: custom/native speedup is 1.34x at 512 tasks,
    # but drops below one at 1024 because per-task MIX synchronization starts
    # to dominate.  Keep the fast small-shape path without regressing medium.
    use_fused_epilogue = (
        D is not None
        and D.ndim == 2
        and z is not None
        and (force_fused_epilogue or batch * nheads * nchunks <= 512)
    )
    if use_fused_epilogue and not return_pre_gate:
        out = torch.ops.mamba_ascend.mamba2_ssd_off_epilogue(
            C_g_cube, states_start, dA_cs, y_diag, x, D, z
        )
        return out, final_state

    dA_gr = dA_cs.reshape(
        batch, ngroups, heads_per_group, nchunks, micro_chunk
    )
    C_gr = C_g_cube.permute(0, 2, 1, 3, 4).unsqueeze(2)
    state_np = states_start.reshape(
        batch, ngroups, heads_per_group, nchunks, headdim, dstate
    ).transpose(-1, -2)
    y_off = torch.matmul(C_gr, state_np.to(torch.float16)).float()
    y_off = y_off * torch.exp(dA_gr).unsqueeze(-1)
    y_diag = y_diag.reshape(
        batch, ngroups, heads_per_group, nchunks, micro_chunk, headdim
    )

    out = (y_diag + y_off).permute(0, 3, 4, 1, 2, 5)
    out = out.reshape(batch, seqlen, nheads, headdim)
    if D is not None:
        out = out + x.float() * D.float().view(1, 1, nheads, -1)
    pre_gate = out
    if z is not None:
        out = out * F.silu(z.float())
    if return_pre_gate and return_states_start:
        result = (out, final_state, pre_gate, states_start)
        if return_preprocess_cache:
            return (*result, x_cube, dA_cs, B_g_cube, C_g_cube)
        return result
    if return_pre_gate:
        result = (out, final_state, pre_gate)
        if return_preprocess_cache:
            return (*result, x_cube, dA_cs, B_g_cube, C_g_cube)
        return result
    if return_states_start:
        result = (out, final_state, states_start)
        if return_preprocess_cache:
            return (*result, x_cube, dA_cs, B_g_cube, C_g_cube)
        return result
    result = (out, final_state)
    if return_preprocess_cache:
        return (*result, x_cube, dA_cs, B_g_cube, C_g_cube)
    return result


def _aligned_ssd_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    z,
    dt_bias,
    dt_softplus,
    dt_limit,
    initial_states,
    return_pre_gate=False,
    return_states_start=False,
    return_preprocess_cache=False,
):
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2:]
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups

    # The custom preprocessor wins when launch/layout overhead dominates.  For
    # larger tensors its scalar softplus/prefix-sum loop is slower than the
    # vectorized ACLNN path, even though the payload copies use strided DMA.
    use_fused_preprocess = batch * seqlen * nheads <= 8192
    if use_fused_preprocess:
        x_h, dA_cs, B_g_cube, C_g_cube = (
            torch.ops.mamba_ascend.mamba2_ssd_preprocess(
                x,
                dt,
                A,
                B,
                C,
                dt_bias,
                chunk_size,
                dt_softplus,
                float(dt_limit[0]),
                float(dt_limit[1]),
            )
        )
    else:
        dt_f = dt.float()
        if dt_bias is not None:
            dt_f = dt_f + dt_bias.float().view(1, 1, nheads)
        if dt_softplus:
            dt_f = F.softplus(dt_f)
        dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])
        dA = dt_f * A.float().view(1, 1, nheads)
        dA_cs = dA.reshape(
            batch, nchunks, chunk_size, nheads
        ).permute(0, 3, 1, 2).cumsum(-1)
        x_h = (x.float() * dt_f.unsqueeze(-1)).reshape(
            batch, nchunks, chunk_size, nheads, headdim
        ).permute(0, 3, 1, 2, 4).contiguous()
        B_g_cube = B.float().reshape(
            batch, nchunks, chunk_size, ngroups, dstate
        ).permute(0, 1, 3, 4, 2).contiguous().to(torch.float16)
        C_g_cube = C.float().reshape(
            batch, nchunks, chunk_size, ngroups, dstate
        ).permute(0, 1, 3, 2, 4).contiguous().to(torch.float16)

    cb = torch.matmul(C_g_cube, B_g_cube)
    dA_gr = dA_cs.reshape(
        batch, ngroups, heads_per_group, nchunks, chunk_size
    )
    x_cube = x_h.to(torch.float16)
    # The current Vector prepare kernel is correct on 910B3 but its
    # ``weighted_x`` output fails the independent reference gate on 950PR
    # (arch35).  Keep the validated 910 path and use the equivalent ACLNN
    # composition on 950 until an arch35-native prepare kernel is available.
    use_fused_prepare = chunk_size <= 64 and not _is_ascend950(x)
    if use_fused_prepare:
        w_cube, weighted_x_cube = torch.ops.mamba_ascend.mamba2_ssd_prepare(
            cb, dA_cs, x_cube
        )
        y_diag = torch.matmul(w_cube, x_cube).float()
        y_diag = y_diag.reshape(
            batch, ngroups, heads_per_group, nchunks, chunk_size, headdim
        )
        weighted_x_gr = weighted_x_cube.reshape(
            batch, ngroups, heads_per_group, nchunks, chunk_size, headdim
        )
    else:
        x_gr = x_cube.reshape(
            batch, ngroups, heads_per_group, nchunks, chunk_size, headdim
        )
        cb_gr = cb.permute(0, 2, 1, 3, 4).unsqueeze(2)
        causal = torch.exp(_lower_segsum(dA_gr))
        w = cb_gr.float() * causal
        y_diag = torch.matmul(w.to(torch.float16), x_gr).float()
        decay_to_end = torch.exp(dA_gr[..., -1:] - dA_gr)
        weighted_x_gr = (
            x_gr.float() * decay_to_end.unsqueeze(-1)
        ).to(torch.float16)

    B_gr = B_g_cube.permute(0, 2, 1, 3, 4).unsqueeze(2)
    chunk_states = torch.matmul(B_gr, weighted_x_gr).float()
    chunk_states = chunk_states.reshape(
        batch, nheads, nchunks, dstate, headdim
    )
    chunk_states = chunk_states.transpose(-1, -2).contiguous()
    states_start, final_state = torch.ops.mamba_ascend.mamba2_ssd_state_passing(
        chunk_states, dA_cs, initial_states
    )

    C_gr = C_g_cube.permute(0, 2, 1, 3, 4).unsqueeze(2)
    state_np = states_start.reshape(
        batch, ngroups, heads_per_group, nchunks, headdim, dstate
    ).transpose(-1, -2)
    y_off = torch.matmul(C_gr, state_np.to(torch.float16)).float()
    y_off = y_off * torch.exp(dA_gr).unsqueeze(-1)

    out = (y_diag + y_off).permute(0, 3, 4, 1, 2, 5)
    out = out.reshape(batch, seqlen, nheads, headdim)
    if D is not None:
        d_view = D.float().view(1, 1, nheads, -1)
        out = out + x.float() * d_view
    pre_gate = out
    if z is not None:
        out = out * F.silu(z.float())
    result = [out, final_state]
    if return_pre_gate:
        result.append(pre_gate)
    if return_states_start:
        result.append(states_start)
    if return_preprocess_cache:
        result.extend((x_cube, dA_cs, B_g_cube, C_g_cube))
    return tuple(result)


def _generic_final_state_950(
    x,
    dt,
    A,
    B,
    dt_bias,
    dt_softplus,
    dt_limit,
    initial_states,
):
    """Recompute the generic final state for the current arch35 fallback.

    The generic AscendC core produces a correct token output on 950PR but its
    final-state output is zero for non-aligned shapes.  This group-aware
    formulation performs one batched ``B^T @ weighted_x`` without expanding
    shared B across heads.  It is only used when the caller requests the final
    state; the 910B3 core path remains unchanged.
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2:]
    heads_per_group = nheads // ngroups

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float().view(1, 1, nheads)
    if dt_softplus:
        dt_f = F.softplus(dt_f)
    dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])
    d_a_cs = (dt_f * A.float().view(1, 1, nheads)).cumsum(dim=1)
    decay_to_end = torch.exp(d_a_cs[:, -1:].float() - d_a_cs.float())
    weighted_x = x.float() * dt_f.unsqueeze(-1) * decay_to_end.unsqueeze(-1)

    weighted_x_grouped = weighted_x.reshape(
        batch, seqlen, ngroups, heads_per_group, headdim
    ).permute(0, 2, 3, 1, 4)
    b_grouped = B.float().permute(0, 2, 3, 1).unsqueeze(2)
    contribution_np = torch.matmul(b_grouped, weighted_x_grouped)
    final_state = contribution_np.reshape(
        batch, nheads, dstate, headdim
    ).transpose(-1, -2)
    if initial_states is not None:
        total_decay = torch.exp(d_a_cs[:, -1]).unsqueeze(-1).unsqueeze(-1)
        final_state = final_state + initial_states.float() * total_decay
    return final_state.contiguous()


def _mamba2_ssd_fwd_impl(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run the native AscendC Mamba-2 SSD forward operator.

    The signature and tensor layouts match
    ``mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined``.
    The current implementation accepts contiguous FP32 NPU tensors.
    """
    execution_chunk_size = _select_execution_chunk_size(x, B, chunk_size)
    if _can_use_grouped_path(x, B, execution_chunk_size, D, z):
        out, final_state = _grouped_ssd_fwd(
            x,
            dt,
            A,
            B,
            C,
            execution_chunk_size,
            D,
            z,
            dt_bias,
            dt_softplus,
            dt_limit,
            initial_states,
        )
    elif _can_use_chunk_mix_path(x, B, execution_chunk_size):
        out, final_state = _chunk_mix_ssd_fwd(
            x,
            dt,
            A,
            B,
            C,
            execution_chunk_size,
            D,
            z,
            dt_bias,
            dt_softplus,
            dt_limit,
            initial_states,
        )
    elif _can_use_aligned_path(x, B, execution_chunk_size):
        out, final_state = _aligned_ssd_fwd(
            x,
            dt,
            A,
            B,
            C,
            execution_chunk_size,
            D,
            z,
            dt_bias,
            dt_softplus,
            dt_limit,
            initial_states,
        )
    else:
        out, final_state = torch.ops.mamba_ascend.mamba2_ssd_fwd(
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            initial_states,
            chunk_size,
            dt_softplus,
            float(dt_limit[0]),
            float(dt_limit[1]),
        )
        if return_final_state and _is_ascend950(x):
            final_state = _generic_final_state_950(
                x,
                dt,
                A,
                B,
                dt_bias,
                dt_softplus,
                dt_limit,
                initial_states,
            )
    if return_final_state:
        return out, final_state
    return out


def _requires_grad(*values) -> bool:
    return torch.is_grad_enabled() and any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in values
    )


def _can_use_native_bwd_m0(
    x,
    B,
    chunk_size,
    D,
    z,
    dt_bias,
    dt_softplus,
    dt_limit,
    initial_states,
) -> bool:
    return (
        x.ndim == 4
        and B.ndim == 4
        and x.dtype == torch.float32
        and B.dtype == torch.float32
        and x.shape[-1] == 64
        and B.shape[-1] == 64
        and chunk_size == 64
        and x.shape[1] % 64 == 0
        and D is None
        and z is None
        and dt_bias is None
        and initial_states is None
        and not dt_softplus
        and float(dt_limit[0]) == 0.0
        and float(dt_limit[1]) >= 1.0e30
    )


def _can_use_native_bwd_m1(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    z,
    dt_bias,
    initial_states,
) -> bool:
    """Return whether the native/composite M1 chunk backward is supported."""
    tensors = (x, dt, A, B, C)
    optionals = (D, z, dt_bias, initial_states)
    if _is_ascend950(x):
        if not (
            _can_use_grouped_path(x, B, chunk_size, D, z)
            or _can_use_aligned_path(x, B, chunk_size)
        ):
            return False
    elif not _can_use_chunk_mix_path(x, B, chunk_size):
        return False
    if chunk_size != 64 or x.shape[-1] != 64 or B.shape[-1] != 64:
        return False
    if any(value.dtype != torch.float32 for value in tensors):
        return False
    if any(not value.is_contiguous() for value in tensors):
        return False
    if any(
        value is not None
        and (value.dtype != torch.float32 or not value.is_contiguous())
        for value in optionals
    ):
        return False
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2:]
    if (
        dt.shape != (batch, seqlen, nheads)
        or A.shape != (nheads,)
        or B.shape != C.shape
        or B.shape[:2] != (batch, seqlen)
        or nheads % ngroups
        or headdim != 64
        or dstate != 64
        or seqlen % 64
    ):
        return False
    if D is not None and D.shape not in ((nheads,), (nheads, headdim)):
        return False
    if z is not None and z.shape != x.shape:
        return False
    if dt_bias is not None and dt_bias.shape != (nheads,):
        return False
    if initial_states is not None and initial_states.shape != (
        batch,
        nheads,
        headdim,
        dstate,
    ):
        return False
    return True


def _expand_group_tensor(tensor, nheads):
    """Convert [B,K,G,T,N] to head-owned [B,H,K,T,N]."""
    groups = tensor.shape[2]
    return tensor.permute(0, 2, 1, 3, 4).repeat_interleave(
        nheads // groups, dim=1
    )


def _chunk_scan_bwd_diag_prepare_hybrid(
    gy_h,
    x_cube,
    d_a_cs,
    b_cube,
    c_cube,
    precomputed_wr=None,
    group_layout_cache=None,
):
    """Build the DiagState terms that do not depend on state backward.

    Global BatchMatMul dispatches amortize the fixed cost of the seven 64x64
    GEMMs.  Keeping this prefix separate allows it to overlap the independent
    off-diagonal and reverse state-passing chain on a second NPU stream.
    """
    batch, nheads, nchunks, chunk_size, _ = gy_h.shape
    ngroups = b_cube.shape[2]
    if precomputed_wr is None:
        cb_group = torch.matmul(c_cube, b_cube)
        w, r = torch.ops.mamba_ascend.mamba2_ssd_prepare(
            cb_group, d_a_cs, x_cube
        )
    else:
        w, r = precomputed_wr
    gy_half = gy_h.to(torch.float16)
    d_x_diag = torch.matmul(w.transpose(-1, -2), gy_half)
    d_w = torch.matmul(gy_half, x_cube.transpose(-1, -2))

    # dCB needs only the causal decay transform.  The dedicated path avoids
    # allocating and writing the 128 MiB weighted-x output of full prepare.
    d_cb_group = torch.ops.mamba_ascend.mamba2_ssd_prepare_dcb(
        d_w,
        d_a_cs,
        ngroups,
    )
    # dB/dC are group-owned outputs.  Reduce dCB before these two matmuls so
    # their compute and result materialization scale with G instead of H.
    if group_layout_cache is None:
        b_time_group = b_cube.permute(0, 2, 1, 4, 3)
        c_group = c_cube.permute(0, 2, 1, 3, 4)
    else:
        b_time_group, c_group = group_layout_cache
    d_c_diag = torch.matmul(d_cb_group, b_time_group)
    d_b_diag = torch.matmul(
        d_cb_group.transpose(-1, -2), c_group
    )
    return (
        d_x_diag,
        d_b_diag,
        d_c_diag,
        d_w,
        w,
        r,
        b_time_group.unsqueeze(2),
    )


def _chunk_scan_bwd_diag_finish_hybrid(
    prepared,
    d_a_cs,
    d_chunk_states,
    ngroups,
    b_cube=None,
    d_c_off_group=None,
):
    """Finish the state-dependent DiagState terms and fused AIV epilogue."""
    (
        d_x_diag,
        d_b_diag,
        d_c_diag,
        d_w,
        w,
        r,
        b_time_group,
    ) = prepared
    u_half = (
        d_chunk_states
        if d_chunk_states.dtype == torch.float16
        else d_chunk_states.to(torch.float16)
    )
    if u_half.ndim == 6:
        if b_cube is None:
            raise RuntimeError("grouped d_chunk_states requires b_cube")
        (
            batch,
            nchunks,
            grouped_ngroups,
            dstate,
            heads_per_group,
            headdim,
        ) = u_half.shape
        if grouped_ngroups != ngroups:
            raise RuntimeError("grouped d_chunk_states has invalid groups")
        # Cube consumes the producer-native [N,R*P] state directly and emits
        # [T,R,P].  dR remains grouped through GM and is gathered by MTE2 in
        # the fused Diag epilogue.
        d_r = torch.matmul(
            b_cube.transpose(-1, -2),
            u_half.reshape(
                batch,
                nchunks,
                ngroups,
                dstate,
                heads_per_group * headdim,
            ),
        ).reshape(
            batch,
            nchunks,
            ngroups,
            64,
            heads_per_group,
            headdim,
        )
        nheads = ngroups * heads_per_group
        r_group = r.reshape(
            batch,
            ngroups,
            heads_per_group,
            nchunks,
            64,
            headdim,
        )
        u_head = u_half.permute(0, 2, 4, 1, 3, 5)
        d_b_state = torch.matmul(
            r_group, u_head.transpose(-1, -2)
        ).reshape(batch, nheads, nchunks, 64, dstate)
    else:
        batch, nheads, nchunks, headdim, dstate = u_half.shape
        heads_per_group = nheads // ngroups
        u_group = u_half.reshape(
            batch, ngroups, heads_per_group, nchunks, headdim, dstate
        )
        d_r = torch.matmul(
            b_time_group, u_group.transpose(-1, -2)
        ).reshape(batch, nheads, nchunks, -1, headdim)
        d_b_state = torch.matmul(r, u_half)
    return torch.ops.mamba_ascend.mamba2_ssd_bwd_diag_finalize(
        d_x_diag,
        d_r,
        d_b_diag,
        d_b_state,
        d_c_diag,
        d_w,
        w,
        r,
        d_a_cs,
        ngroups,
        d_c_off_group,
    )


def _chunk_scan_bwd_diag_state_hybrid(
    gy_h,
    x_cube,
    d_a_cs,
    b_cube,
    c_cube,
    d_chunk_states,
    precomputed_wr=None,
    group_layout_cache=None,
    d_c_off_group=None,
):
    """Serial compatibility wrapper for the hybrid DiagState path."""
    prepared = _chunk_scan_bwd_diag_prepare_hybrid(
        gy_h,
        x_cube,
        d_a_cs,
        b_cube,
        c_cube,
        precomputed_wr,
        group_layout_cache,
    )
    return _chunk_scan_bwd_diag_finish_hybrid(
        prepared, d_a_cs, d_chunk_states, b_cube.shape[2], b_cube,
        d_c_off_group
    )


def _chunk_scan_bwd_off_hybrid(
    gy_h,
    states_start,
    d_a_cs,
    c_cube,
    c_group_cache=None,
):
    """Batched-Cube off-diagonal backward for high task counts.

    A fused AIV prepare emits the two FP16 Cube operands.  Group-broadcast
    BatchMatMul avoids materializing a head-expanded C tensor, and the final
    AIV pass casts both results to FP32 while forming the dA contribution.
    """
    batch, nheads, nchunks = gy_h.shape[:3]
    ngroups = c_cube.shape[2]
    heads_per_group = nheads // ngroups
    gy_half = gy_h if gy_h.dtype == torch.float16 else gy_h.to(torch.float16)
    q_half, state_half = (
        torch.ops.mamba_ascend.mamba2_ssd_bwd_off_prepare(
            gy_half, states_start, d_a_cs
        )
    )
    q_group = q_half.reshape(
        batch, ngroups, heads_per_group, nchunks, 64, 64
    )
    state_group = state_half.reshape_as(q_group)
    c_group = (
        c_cube.permute(0, 2, 1, 3, 4)
        if c_group_cache is None
        else c_group_cache
    ).unsqueeze(2)
    d_states_half = torch.matmul(q_group.transpose(-1, -2), c_group)
    use_group_reduce = os.environ.get(
        "MAMBA_ASCENDC_BWD_GROUP_REDUCE", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if use_group_reduce:
        d_c_group, g_d_a = (
            torch.ops.mamba_ascend.mamba2_ssd_bwd_off_group_reduce(
                q_group, state_group, c_cube
            )
        )
        return (
            d_states_half.contiguous().view(
                batch, nheads, nchunks, 64, 64
            ),
            d_c_group,
            g_d_a,
        )
    d_c_head_half = torch.matmul(q_group, state_group)
    return torch.ops.mamba_ascend.mamba2_ssd_bwd_off_finalize(
        d_states_half.contiguous(),
        d_c_head_half.contiguous(),
        c_cube,
    )


def _chunk_scan_bwd_off_grouped_hybrid(
    gy_h,
    states_grouped,
    d_a_cs,
    c_cube,
):
    """950PR group-wide Off path with canonical state layout.

    Vector writes Q directly as [B,K,G,T,R,P].  Global Cube BMM keeps the
    sustained-throughput implementation but turns four 64x64 products into
    64x256 products.  dState remains [N,R,P], matching the grouped training
    state cache and avoiding the GiB-scale grouped-to-head transpose.
    """
    batch, nchunks, ngroups, dstate, heads_per_group, headdim = (
        states_grouped.shape
    )
    q_group, _ = torch.ops.mamba_ascend.mamba2_ssd_bwd_off_prepare(
        gy_h,
        states_grouped,
        d_a_cs,
        ngroups,
    )
    q_wide = q_group.reshape(
        batch, nchunks, ngroups, 64, heads_per_group * headdim
    )
    state_wide = states_grouped.reshape(
        batch, nchunks, ngroups, dstate, heads_per_group * headdim
    )
    d_states_wide = torch.matmul(
        c_cube.transpose(-1, -2), q_wide
    )
    d_c_group_half = torch.matmul(
        q_wide, state_wide.transpose(-1, -2)
    )
    y_base_wide = torch.matmul(c_cube, state_wide)
    return torch.ops.mamba_ascend.mamba2_ssd_bwd_off_finalize_grouped(
        d_states_wide.reshape_as(states_grouped).contiguous(),
        d_c_group_half.contiguous(),
        q_group,
        y_base_wide.reshape_as(q_group).contiguous(),
    )


def _chunk_mix_ssd_bwd_m1(
    x,
    dt,
    A,
    B,
    C,
    dout,
    D,
    z,
    dt_bias,
    initial_states,
    dfinal_state,
    y_pre_saved,
    states_start_saved,
    preprocess_cache,
    dt_softplus,
    dt_limit,
):
    """M1 integration gate with native off/state/dt/diag-state kernels."""
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    chunk_size = 64
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups
    logical_head_chunk_tasks = batch * nheads * nchunks
    hybrid_diag_mode = os.environ.get(
        "MAMBA_ASCENDC_BWD_HYBRID_DIAG", "auto"
    ).lower()
    if hybrid_diag_mode not in {
        "auto", "0", "false", "no", "off", "1", "true", "yes", "on"
    }:
        raise ValueError(
            "MAMBA_ASCENDC_BWD_HYBRID_DIAG must be auto, on, or off"
        )
    # The packed group-owned Cube path remains available through explicit
    # ``off`` for development.  Despite removing repeat_interleave traffic,
    # its H/G=4 H256 latency is currently worse than the global-BMM hybrid,
    # so production ``auto`` must keep the measured faster composition.
    use_hybrid_diag = hybrid_diag_mode in {"1", "true", "yes", "on"} or (
        hybrid_diag_mode == "auto"
        and (_is_ascend950(x) or logical_head_chunk_tasks >= 1024)
    )
    hybrid_off_mode = os.environ.get(
        "MAMBA_ASCENDC_BWD_HYBRID_OFF", "auto"
    ).lower()
    if hybrid_off_mode not in {
        "auto", "0", "false", "no", "off", "1", "true", "yes", "on"
    }:
        raise ValueError(
            "MAMBA_ASCENDC_BWD_HYBRID_OFF must be auto, on, or off"
        )
    use_hybrid_off = hybrid_off_mode in {"1", "true", "yes", "on"} or (
        hybrid_off_mode == "auto"
        and (_is_ascend950(x) or logical_head_chunk_tasks >= 1024)
    )
    grouped_off_mode = os.environ.get(
        "MAMBA_ASCENDC_BWD_GROUPED_OFF", "auto"
    ).lower()
    if grouped_off_mode not in {
        "auto", "0", "false", "no", "off", "1", "true", "yes", "on"
    }:
        raise ValueError(
            "MAMBA_ASCENDC_BWD_GROUPED_OFF must be auto, on, or off"
        )
    use_grouped_off = (
        use_hybrid_off
        and states_start_saved is not None
        and states_start_saved.ndim == 6
        and states_start_saved.shape[4] == 4
        # The 950PR group-wide candidate remains an explicit development
        # path.  H256 validation exposed a large-shape dC layout failure and
        # a latency regression, so production auto dispatch must retain the
        # validated per-head hybrid until both gates are fixed.
        and grouped_off_mode in {"1", "true", "yes", "on"}
    )
    if preprocess_cache is None:
        x_cube, d_a_cs, b_cube, c_cube = (
            torch.ops.mamba_ascend.mamba2_ssd_preprocess(
                x,
                dt,
                A,
                B,
                C,
                dt_bias,
                chunk_size,
                dt_softplus,
                float(dt_limit[0]),
                float(dt_limit[1]),
            )
        )
    else:
        x_cube, d_a_cs, b_cube, c_cube = preprocess_cache

    # B_time and C_group are each consumed by several independent global
    # BatchMatMul calls.  Passing the same non-contiguous view to every call
    # makes ACLNN materialize the same 256-MiB transpose repeatedly.  The
    # candidate path creates each canonical group-major tensor once and
    # shares it across states recompute, DiagState and Off.  Auto dispatch is
    # enabled only in the sustained-throughput region validated on both
    # 910B3 and 950PR; explicit on/off values remain available for A/B tests.
    reuse_group_layout_mode = os.environ.get(
        "MAMBA_ASCENDC_BWD_REUSE_GROUP_LAYOUT", "auto"
    ).lower()
    if reuse_group_layout_mode not in {
        "auto", "0", "false", "no", "off", "1", "true", "yes", "on"
    }:
        raise ValueError(
            "MAMBA_ASCENDC_BWD_REUSE_GROUP_LAYOUT must be auto, on, or off"
        )
    reuse_group_layout = (
        reuse_group_layout_mode in {"1", "true", "yes", "on"}
        or (
            reuse_group_layout_mode == "auto"
            and use_hybrid_off
            and use_hybrid_diag
            and logical_head_chunk_tasks >= 16384
        )
    )
    group_layout_cache = None
    if reuse_group_layout:
        group_layout_cache = (
            b_cube.permute(0, 2, 1, 4, 3).contiguous(),
            c_cube.permute(0, 2, 1, 3, 4).contiguous(),
        )
    # Hybrid Diag already needs W and weighted-x.  Compute them once before
    # states recompute so weighted-x can serve both consumers; the old path
    # redundantly rebuilt it with FP32 casts, Exp and Mul, then discarded it.
    diag_wr = None
    if use_hybrid_diag:
        cb_group = torch.matmul(c_cube, b_cube)
        split_prepare_mode = os.environ.get(
            "MAMBA_ASCENDC_BWD_SPLIT_PREPARE", "auto"
        ).lower()
        if split_prepare_mode not in {
            "auto", "0", "false", "no", "off", "1", "true", "yes", "on"
        }:
            raise ValueError(
                "MAMBA_ASCENDC_BWD_SPLIT_PREPARE must be auto, on, or off"
            )
        use_split_prepare = (
            chunk_size == 64
            and headdim == 64
            and (
                split_prepare_mode in {"1", "true", "yes", "on"}
                or (
                    split_prepare_mode == "auto"
                    and _is_ascend950(x_cube)
                )
            )
        )
        prepare_op = (
            torch.ops.mamba_ascend.mamba2_ssd_prepare_split
            if use_split_prepare
            else torch.ops.mamba_ascend.mamba2_ssd_prepare
        )
        diag_wr = prepare_op(cb_group, d_a_cs, x_cube)
    y_diag = None
    if states_start_saved is None:
        chunk_states = None
        use_torch_states_recompute = os.environ.get(
            "MAMBA_ASCENDC_BWD_TORCH_STATES", "1"
        ).lower() in {"1", "true", "yes", "on"}
        if use_torch_states_recompute:
            # Backward consumes only chunk states, so do not recompute the
            # forward-only y_diag branch.  This verified composition is the
            # default until the same states-only algebra is lowered into the
            # persistent native backward kernel.
            x_group = x_cube.reshape(
                batch,
                ngroups,
                heads_per_group,
                nchunks,
                chunk_size,
                headdim,
            )
            d_a_group = d_a_cs.reshape(
                batch,
                ngroups,
                heads_per_group,
                nchunks,
                chunk_size,
            )
            if diag_wr is not None:
                weighted_x = diag_wr[1].reshape_as(x_group)
            else:
                decay_to_end = torch.exp(
                    d_a_group[..., -1:] - d_a_group
                )
                weighted_x = (
                    x_group.float() * decay_to_end.unsqueeze(-1)
                ).to(torch.float16)
            b_time_group = (
                b_cube.permute(0, 2, 1, 4, 3)
                if group_layout_cache is None
                else group_layout_cache[0]
            ).unsqueeze(2)
            if diag_wr is not None:
                # Produce [P,N] directly.  The equivalent B @ weighted_x
                # form emits [N,P] and required a 256-MiB output transpose.
                chunk_states = torch.matmul(
                    weighted_x.transpose(-1, -2),
                    b_time_group,
                ).reshape(
                    batch, nheads, nchunks, headdim, 64
                )
                # Folding the FP16->FP32 conversion into StatePassing saves a
                # full HBM round trip once enough chunk/head tasks are active.
                # For small launches, the in-kernel branch/cast costs more than
                # the standalone cast, so retain the FP32 input fast path.
                if batch * nheads * nchunks < 1024:
                    chunk_states = chunk_states.float()
            else:
                b_state_group = b_cube.permute(
                    0, 2, 1, 3, 4
                ).unsqueeze(2)
                chunk_states_np = torch.matmul(
                    b_state_group, weighted_x
                ).float()
                chunk_states_np = chunk_states_np.reshape(
                    batch, nheads, nchunks, 64, headdim
                )
        else:
            y_diag, chunk_states_np = (
                torch.ops.mamba_ascend.mamba2_ssd_chunk_mix(
                    x_cube, d_a_cs, b_cube, c_cube
                )
            )
        if chunk_states is None:
            chunk_states = chunk_states_np.transpose(-1, -2).contiguous()
        states_start, _ = torch.ops.mamba_ascend.mamba2_ssd_state_passing(
            chunk_states, d_a_cs, initial_states
        )
    else:
        states_start = states_start_saved
        if states_start.ndim == 6 and not use_grouped_off:
            states_start = (
                states_start.permute(0, 2, 4, 1, 5, 3)
                .contiguous()
                .reshape(batch, nheads, nchunks, headdim, 64)
            )

    # The saved training cache is FP16.  The high-task hybrid Off and
    # StatePassing kernels consume it directly, while the legacy small-task
    # native kernels still require FP32.  Materialize at most one compatibility
    # copy, and never create it on the H256 heavy path where both hybrids are
    # selected.
    states_start_compat = states_start
    if states_start.dtype == torch.float16 and (
        not use_hybrid_off or not use_hybrid_diag
    ):
        states_start_compat = states_start.float()

    fused_gate = False
    fused_dd_in_dt = False
    d_d_fused = None
    use_fused_gate = (
        z is not None
        and D is not None
        and D.ndim == 2
        and y_pre_saved is not None
        and os.environ.get(
            "MAMBA_ASCENDC_FUSED_BWD_GATE", "1"
        ).lower() in {"1", "true", "yes", "on"}
    )
    if use_fused_gate:
        fused_dd_in_dt = (
            nheads >= 2
            and chunk_size == 64
            and os.environ.get(
                "MAMBA_ASCENDC_FUSE_DD_IN_DT", "1"
            ).lower() in {"1", "true", "yes", "on"}
        )
        if fused_dd_in_dt:
            gy_h, dz = torch.ops.mamba_ascend.mamba2_ssd_bwd_gate_nodd(
                dout, z, y_pre_saved
            )
        else:
            gy_h, dz, d_d_fused = (
                torch.ops.mamba_ascend.mamba2_ssd_bwd_gate(
                    dout, z, y_pre_saved, x
                )
            )
        fused_gate = True
    elif z is None:
        gy = dout
        dz = None
    else:
        # The training forward stores this value while it is already resident
        # in UB.  Reusing it removes a second C@state projection, decay,
        # layout conversion and D epilogue from backward.  Dividing the gated
        # output by silu(z) is intentionally avoided because it is unstable
        # around z=0.
        if y_pre_saved is None:
            if y_diag is None:
                raise RuntimeError(
                    "saving states_start requires saving y_pre when z is present"
                )
            c_head = _expand_group_tensor(c_cube, nheads).float()
            state_np = states_start.transpose(-1, -2).contiguous().float()
            y_off = torch.matmul(c_head, state_np).float()
            y_off = y_off * torch.exp(d_a_cs).unsqueeze(-1)
            y_pre_h = y_diag.float() + y_off
            y_pre = y_pre_h.permute(0, 2, 3, 1, 4).reshape_as(x)
            if D is not None:
                y_pre = y_pre + x * D.view(1, 1, nheads, -1)
        else:
            y_pre = y_pre_saved
        sigmoid_z = torch.sigmoid(z)
        silu_z = z * sigmoid_z
        gy = dout * silu_z
        dz = dout * y_pre * sigmoid_z * (1.0 + z * (1.0 - sigmoid_z))
    if not fused_gate:
        gy_h = gy.reshape(
            batch, nchunks, chunk_size, nheads, headdim
        ).permute(0, 3, 1, 2, 4).contiguous()

    if use_grouped_off:
        d_states_start, d_c_off_group, g_cs_off = (
            _chunk_scan_bwd_off_grouped_hybrid(
                gy_h,
                states_start,
                d_a_cs,
                c_cube,
            )
        )
    elif use_hybrid_off:
        d_states_start, d_c_off_group, g_cs_off = (
            _chunk_scan_bwd_off_hybrid(
                gy_h,
                states_start,
                d_a_cs,
                c_cube,
                None if group_layout_cache is None else group_layout_cache[1],
            )
        )
    else:
        gy_off = gy_h.float() if gy_h.dtype == torch.float16 else gy_h
        d_states_start, d_c_off_head, g_cs_off = (
            torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_off(
                gy_off, states_start_compat, d_a_cs, c_cube
            )
        )
    state_passing_bwd_op = (
        torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd_half
        if use_hybrid_diag
        else torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd
    )
    d_chunk_states, d_initial, g_cs_chunk_last = (
        state_passing_bwd_op(
            states_start if use_hybrid_diag else states_start_compat,
            d_states_start,
            d_a_cs,
            None if dfinal_state is None else dfinal_state.contiguous(),
        )
    )

    # The MIX operator fuses the remaining diagonal and chunk-state
    # backward branches.  Cube performs all seven 64x64 GEMMs; Vector owns
    # causal decay, mask, reductions and the two branch sums.
    if use_hybrid_diag:
        (
            d_xdt_total,
            d_b_group,
            d_c_diag_group,
            g_cs_diag_state,
        ) = _chunk_scan_bwd_diag_state_hybrid(
            gy_h,
            x_cube,
            d_a_cs,
            b_cube,
            c_cube,
            d_chunk_states,
            diag_wr,
            group_layout_cache,
            d_c_off_group if use_hybrid_off else None,
        )
    else:
        gy_diag = gy_h.float() if gy_h.dtype == torch.float16 else gy_h
        d_xdt_total, d_b_group, d_c_diag_group, g_cs_diag_state = (
            torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_diag_state(
                gy_diag,
                x_cube,
                d_a_cs,
                b_cube,
                c_cube,
                d_chunk_states,
            )
        )
    grouped_dt_mode = os.environ.get(
        "MAMBA_ASCENDC_BWD_GROUPED_DT", "auto"
    ).lower()
    if grouped_dt_mode not in {
        "auto", "0", "false", "no", "off", "1", "true", "yes", "on"
    }:
        raise ValueError(
            "MAMBA_ASCENDC_BWD_GROUPED_DT must be auto, on, or off"
        )
    # Producer-native grouped Off writes failed deterministic stress tests on
    # both 910B3 and 950PR.  Keep the standalone grouped Dt op available for a
    # future producer redesign, but do not dispatch it from public backward.
    use_grouped_dt = False
    # Fusing head-major Diag/Off into Dt was measured at 84.081 ms on the
    # H256 stress case versus 83.289 ms for materializing g_cs_total.  Dt is
    # already MTE2-heavy, so keep that rejected experiment out of dispatch.
    fuse_gcs_in_dt = False
    # A separate per-task chunk-tail MTE2 load regressed H256 from 83.29 ms
    # to 83.74 ms, so the source default stays on the framework tail update.
    fuse_gcs_tail_in_dt = False
    if use_grouped_dt:
        # Keep Off in its producer-native [B,K,G,T,R] layout.  Dt selects the
        # required head column in UB and fuses Diag + Off + chunk-tail, so the
        # old full-tensor transpose, Add and Cat never reach HBM.
        g_cs_total = None
    else:
        g_cs_total = g_cs_diag_state + g_cs_off
        if not fuse_gcs_tail_in_dt:
            # g_cs_total is a fresh contiguous workspace.  Only one value per
            # chunk needs the state-passing contribution; updating that
            # strided tail view avoids copying the other 63/64 values.
            g_cs_total[..., -1].add_(g_cs_chunk_last)
    if fused_gate:
        if fused_dd_in_dt:
            if use_grouped_dt:
                dx, ddt, d_a, d_dt_bias, d_d_fused = (
                    torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_grouped_d_dd(
                        x,
                        d_xdt_total,
                        g_cs_diag_state,
                        g_cs_off,
                        g_cs_chunk_last,
                        dt,
                        A,
                        gy_h,
                        D,
                        dt_bias,
                        dt_softplus,
                        float(dt_limit[0]),
                        float(dt_limit[1]),
                    )
                )
            elif fuse_gcs_in_dt:
                dx, ddt, d_a, d_dt_bias, d_d_fused = (
                    torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_fused_gcs_d_dd(
                        x,
                        d_xdt_total,
                        g_cs_diag_state,
                        g_cs_off,
                        g_cs_chunk_last,
                        dt,
                        A,
                        gy_h,
                        D,
                        dt_bias,
                        dt_softplus,
                        float(dt_limit[0]),
                        float(dt_limit[1]),
                    )
                )
            elif fuse_gcs_tail_in_dt:
                dx, ddt, d_a, d_dt_bias, d_d_fused = (
                    torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_chunk_tail_d_dd(
                        x,
                        d_xdt_total,
                        g_cs_total,
                        g_cs_chunk_last,
                        dt,
                        A,
                        gy_h,
                        D,
                        dt_bias,
                        dt_softplus,
                        float(dt_limit[0]),
                        float(dt_limit[1]),
                    )
                )
            else:
                dx, ddt, d_a, d_dt_bias, d_d_fused = (
                    torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_d_dd(
                        x,
                        d_xdt_total,
                        g_cs_total,
                        dt,
                        A,
                        gy_h,
                        D,
                        dt_bias,
                        dt_softplus,
                        float(dt_limit[0]),
                        float(dt_limit[1]),
                    )
                )
        else:
            dx, ddt, d_a, d_dt_bias = (
                torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_d(
                    x,
                    d_xdt_total,
                    g_cs_total,
                    dt,
                    A,
                    gy_h,
                    D,
                    dt_bias,
                    dt_softplus,
                    float(dt_limit[0]),
                    float(dt_limit[1]),
                )
            )
    else:
        # Hybrid finalize may expose FP16 or a non-contiguous view for some
        # head/group combinations.  Only the compatibility Dt kernel requires
        # contiguous FP32; fused-D routes accept their producer-native input.
        if (
            d_xdt_total.dtype != torch.float32
            or not d_xdt_total.is_contiguous()
        ):
            d_xdt_total = d_xdt_total.float().contiguous()
        dx, ddt, d_a, d_dt_bias = torch.ops.mamba_ascend.mamba2_ssd_dt_bwd(
            x,
            d_xdt_total,
            g_cs_total,
            dt,
            A,
            dt_bias,
            dt_softplus,
            float(dt_limit[0]),
            float(dt_limit[1]),
        )

    if D is None:
        d_d = None
    elif fused_gate:
        d_d = d_d_fused
    else:
        dx = dx + gy * D.view(1, 1, nheads, -1)
        d_d_full = gy * x
        d_d = (
            d_d_full.sum(dim=(0, 1, 3))
            if D.ndim == 1
            else d_d_full.sum(dim=(0, 1))
        )

    if not use_hybrid_off:
        d_c_off_group = d_c_off_head.reshape(
            batch, ngroups, heads_per_group, nchunks, chunk_size, 64
        ).sum(dim=2)
        d_c_off_group = d_c_off_group.permute(0, 2, 3, 1, 4)
    if use_hybrid_diag:
        d_b = d_b_group.reshape_as(B)
    else:
        d_b = d_b_group.permute(0, 1, 3, 2, 4).reshape_as(B).contiguous()
        d_c_diag_group = d_c_diag_group.permute(0, 1, 3, 2, 4)
    if use_hybrid_diag and use_hybrid_off:
        # Diag finalize has already accumulated dC_off in UB before its
        # single public-layout writeback.
        d_c = d_c_diag_group.reshape_as(C).contiguous()
    else:
        d_c = (d_c_diag_group + d_c_off_group).reshape_as(C).contiguous()
    return (
        dx,
        ddt,
        d_a,
        d_b,
        d_c,
        d_d,
        dz,
        d_dt_bias if dt_bias is not None else None,
        d_initial if initial_states is not None else None,
    )


class _Mamba2SsdAutogradM1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        dt_softplus,
        dt_limit_min,
        dt_limit_max,
        initial_states,
    ):
        grouped_path = _can_use_grouped_path(
            x, B, chunk_size, D, z
        )
        save_states_start = grouped_path or os.environ.get(
            "MAMBA_ASCENDC_SAVE_STATES_START", "0"
        ).lower() in {"1", "true", "yes", "on"}
        save_preprocess_mode = os.environ.get(
            "MAMBA_ASCENDC_SAVE_PREPROCESS", "auto"
        ).lower()
        if save_preprocess_mode not in {
            "auto", "0", "false", "no", "off", "1", "true", "yes", "on"
        }:
            raise ValueError(
                "MAMBA_ASCENDC_SAVE_PREPROCESS must be auto, on, or off"
            )
        logical_head_chunk_tasks = x.shape[0] * x.shape[2] * (x.shape[1] // 64)
        save_preprocess = save_preprocess_mode in {"1", "true", "yes", "on"} or (
            save_preprocess_mode == "auto" and logical_head_chunk_tasks >= 65536
        )
        if grouped_path:
            forward_fn = _grouped_ssd_fwd
        elif _is_ascend950(x):
            # The legacy key-9 state layout is not valid on 950PR.  The
            # aligned composition exposes the same training caches without
            # depending on that layout, so it is also the correctness path
            # for small or non-H/G=4 training shapes.
            forward_fn = _aligned_ssd_fwd
        else:
            forward_fn = _chunk_mix_ssd_fwd
        forward_result = forward_fn(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D,
            z,
            dt_bias,
            dt_softplus,
            (dt_limit_min, dt_limit_max),
            initial_states,
            return_pre_gate=True,
            return_states_start=save_states_start,
            return_preprocess_cache=save_preprocess,
        )
        if save_states_start:
            if save_preprocess:
                (
                    out,
                    final_state,
                    y_pre,
                    states_start,
                    x_cube,
                    d_a_cs,
                    b_cube,
                    c_cube,
                ) = forward_result
            else:
                out, final_state, y_pre, states_start = forward_result
        else:
            if save_preprocess:
                (
                    out,
                    final_state,
                    y_pre,
                    x_cube,
                    d_a_cs,
                    b_cube,
                    c_cube,
                ) = forward_result
            else:
                out, final_state, y_pre = forward_result
            states_start = None
        empty = x.new_empty(0)
        ctx.save_for_backward(
            x,
            dt,
            A,
            B,
            C,
            D if D is not None else empty,
            z if z is not None else empty,
            dt_bias if dt_bias is not None else empty,
            initial_states if initial_states is not None else empty,
            y_pre if z is not None else empty,
            states_start if states_start is not None else empty,
            x_cube if save_preprocess else empty,
            d_a_cs if save_preprocess else empty,
            b_cube if save_preprocess else empty,
            c_cube if save_preprocess else empty,
        )
        ctx.has_D = D is not None
        ctx.has_z = z is not None
        ctx.has_dt_bias = dt_bias is not None
        ctx.has_initial = initial_states is not None
        ctx.has_states_start = states_start is not None
        ctx.has_preprocess_cache = save_preprocess
        ctx.grouped_path = grouped_path
        ctx.dt_softplus = dt_softplus
        ctx.dt_limit = (dt_limit_min, dt_limit_max)
        return out, final_state

    @staticmethod
    def backward(ctx, dout, dfinal_state):
        (
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            initial_states,
            y_pre,
            states_start,
            x_cube,
            d_a_cs,
            b_cube,
            c_cube,
        ) = ctx.saved_tensors
        if dout is None:
            dout = torch.zeros_like(x)
        gradients = _chunk_mix_ssd_bwd_m1(
            x,
            dt,
            A,
            B,
            C,
            dout.contiguous(),
            D if ctx.has_D else None,
            z if ctx.has_z else None,
            dt_bias if ctx.has_dt_bias else None,
            initial_states if ctx.has_initial else None,
            None if dfinal_state is None else dfinal_state,
            y_pre if ctx.has_z else None,
            states_start if ctx.has_states_start else None,
            (x_cube, d_a_cs, b_cube, c_cube)
            if ctx.has_preprocess_cache
            else None,
            ctx.dt_softplus,
            ctx.dt_limit,
        )
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial = gradients
        return (
            dx,
            ddt,
            dA,
            dB,
            dC,
            None,
            dD,
            dz,
            ddt_bias,
            None,
            None,
            None,
            dinitial,
        )


class _Mamba2SsdAutogradM0(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        dt_softplus,
        dt_limit_min,
        dt_limit_max,
        initial_states,
    ):
        out, final_state = _mamba2_ssd_fwd_impl(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=(dt_limit_min, dt_limit_max),
            initial_states=initial_states,
            return_final_state=True,
        )
        ctx.save_for_backward(x, dt, A, B, C, final_state)
        ctx.dt_softplus = dt_softplus
        ctx.dt_limit = (dt_limit_min, dt_limit_max)
        return out, final_state

    @staticmethod
    def backward(ctx, dout, dfinal_state):
        x, dt, A, B, C, final_state = ctx.saved_tensors
        # PyTorch passes ``None`` when the first returned value does not
        # participate in the loss (for example, a final-state-only loss).
        # The native operator always accepts a dense dout tensor.
        if dout is None:
            dout = torch.zeros_like(x)
        result = torch.ops.mamba_ascend.mamba2_ssd_bwd(
            x,
            dt,
            A,
            B,
            C,
            dout.contiguous(),
            final_state,
            None,
            None,
            None,
            None,
            None if dfinal_state is None else dfinal_state.contiguous(),
            ctx.dt_softplus,
            float(ctx.dt_limit[0]),
            float(ctx.dt_limit[1]),
        )
        dx, ddt, dA_partial, dB, dC, _, _, _ = result
        dA = dA_partial.sum(dim=0)
        return (
            dx,
            ddt,
            dA,
            dB,
            dC,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def mamba2_ssd_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    initial_states: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run Mamba-2 SSD forward with M1 native-core or M0 fallback autograd."""
    if _requires_grad(x, dt, A, B, C, D, z, dt_bias, initial_states):
        if _can_use_native_bwd_m1(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D,
            z,
            dt_bias,
            initial_states,
        ):
            out, final_state = _Mamba2SsdAutogradM1.apply(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D,
                z,
                dt_bias,
                dt_softplus,
                float(dt_limit[0]),
                float(dt_limit[1]),
                initial_states,
            )
        elif not _can_use_native_bwd_m0(
            x,
            B,
            chunk_size,
            D,
            z,
            dt_bias,
            dt_softplus,
            dt_limit,
            initial_states,
        ):
            raise NotImplementedError(
                "AscendC backward M0 requires contiguous FP32 P=N=chunk=64, "
                "L divisible by 64, default dt_limit, and no optional "
                "D/z/dt_bias/initial_states or dt_softplus"
            )
        else:
            out, final_state = _Mamba2SsdAutogradM0.apply(
                x,
                dt,
                A,
                B,
                C,
                chunk_size,
                D,
                z,
                dt_bias,
                dt_softplus,
                float(dt_limit[0]),
                float(dt_limit[1]),
                initial_states,
            )
    else:
        out, final_state = _mamba2_ssd_fwd_impl(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            initial_states=initial_states,
            return_final_state=True,
        )
    if return_final_state:
        return out, final_state
    return out


mamba_chunk_scan_combined = mamba2_ssd_fwd
