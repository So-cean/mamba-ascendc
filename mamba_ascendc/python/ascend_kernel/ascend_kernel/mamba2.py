"""PyTorch-facing Mamba-2 SSD forward API for the AscendC extension."""

import os

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


_ADDITIVE_CAUSAL_MASKS = {}


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


def _can_use_chunk_mix_path(x, B, chunk_size):
    if not _chunk_mix_enabled() or not _can_use_aligned_path(x, B, chunk_size):
        return False
    return (
        x.shape[-1] == 64
        and B.shape[-1] in (64, 128)
        and x.shape[1] % 64 == 0
        and chunk_size in (64, 128)
    )


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

    y_diag, chunk_states = torch.ops.mamba_ascend.mamba2_ssd_chunk_mix(
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
        and z is not None
        and (
            force_state_epilogue
            or state_np_layout
            or batch * nheads * nchunks >= 4096
        )
    )
    if use_state_epilogue:
        return torch.ops.mamba_ascend.mamba2_ssd_state_epilogue(
            chunk_states,
            dA_cs,
            C_g_cube,
            y_diag,
            x,
            D,
            z,
            initial_states,
        )

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
        and z is not None
        and (force_fused_epilogue or batch * nheads * nchunks <= 512)
    )
    if use_fused_epilogue:
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
    if z is not None:
        out = out * F.silu(z.float())
    return out, final_state


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
    use_fused_prepare = chunk_size <= 64
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
    if z is not None:
        out = out * F.silu(z.float())
    return out, final_state


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
    """Run the native AscendC Mamba-2 SSD forward operator.

    The signature and tensor layouts match
    ``mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined``.
    V1 accepts contiguous FP32 NPU tensors.
    """
    if _can_use_chunk_mix_path(x, B, chunk_size):
        out, final_state = _chunk_mix_ssd_fwd(
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
        )
    elif _can_use_aligned_path(x, B, chunk_size):
        out, final_state = _aligned_ssd_fwd(
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
    if return_final_state:
        return out, final_state
    return out


mamba_chunk_scan_combined = mamba2_ssd_fwd
