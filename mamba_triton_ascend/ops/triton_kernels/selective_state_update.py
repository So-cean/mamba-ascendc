# Copyright (c) 2024, mamba-triton-ascend authors.
# Selective state update (single-step inference) - Triton Ascend migration.
# Based on mamba_ssm/ops/triton/selective_state_update.py
# Removed @triton.heuristics decorators for Triton-Ascend compatibility.

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, repeat


@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr, state_batch_indices_ptr,
    # Matrix dimensions
    batch, nheads, dim, dstate, nheads_ngroups_ratio,
    # Strides
    stride_state_batch, stride_state_head, stride_state_dim, stride_state_dstate,
    stride_x_batch, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_dim,
    stride_dt_bias_head, stride_dt_bias_dim,
    stride_A_head, stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate,
    stride_C_batch, stride_C_group, stride_C_dstate,
    stride_D_head, stride_D_dim,
    stride_z_batch, stride_z_head, stride_z_dim,
    stride_out_batch, stride_out_head, stride_out_dim,
    # Meta-parameters as tl.constexpr
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    out_ptrs = out_ptr + offs_m * stride_out_dim

    if HAS_STATE_BATCH_INDICES == 1:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr)
        # Skip padding tokens
        if state_batch_idx < 0:
            tl.store(out_ptrs, 0.0, mask=offs_m < dim)
            return
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS == 1:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z == 1:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head

    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS == 1:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D == 1:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D == 1:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z == 1:
        z_ptrs = z_ptr + offs_m * stride_z_dim

    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if TIE_HDIM == 0:
        dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS == 1:
            dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if DT_SOFTPLUS == 1:
            dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA = tl.exp(A * dt[:, None])
    else:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS == 1:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS == 1:
            dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A * dt)  # scalar, not a matrix

    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D == 1:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z == 1:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if TIE_HDIM == 0:
        dB = B[None, :] * dt[:, None]
    else:
        dB = B * dt  # vector of size (dstate,)
    state = state * dA + dB * x[:, None]
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D == 1:
        out += x * D
    if HAS_Z == 1:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)


def selective_state_update(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False,
    state_batch_indices=None
):
    """
    Selective state update for single-step inference.
    Compatible with mamba_ssm.ops.selective_state_update.

    Arguments:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)

    out = torch.empty_like(x)

    # Grid configuration for NPU
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)

    z_strides = ((z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0))

    # Tune by hand to avoid autotune overwriting state
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))

    tie_hdim = (A.stride(-1) == 0 and A.stride(-2) == 0 and
                dt.stride(-1) == 0 and
                (dt_bias is None or dt_bias.stride(-1) == 0))

    # Explicit constexpr flags for Triton-Ascend compatibility
    _has_dt_bias = 1 if dt_bias is not None else 0
    _has_d = 1 if D is not None else 0
    _has_z = 1 if z is not None else 0
    _has_state_batch_indices = 1 if state_batch_indices is not None else 0
    _dt_softplus = 1 if dt_softplus else 0
    block_size_dstate = triton.next_power_of_2(dstate)

    _selective_scan_update_kernel[grid](
        state, x, dt, dt_bias, A, B, C, D, z, out, state_batch_indices,
        batch, nheads, dim, dstate, nheads // ngroups,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else (0, 0),
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
        z_strides[0], z_strides[1], z_strides[2],
        out.stride(0), out.stride(1), out.stride(2),
        _dt_softplus,
        tie_hdim,
        BLOCK_SIZE_M,
        _has_dt_bias,
        _has_d,
        _has_z,
        _has_state_batch_indices,
        block_size_dstate,
        num_warps=num_warps,
    )

    if not has_heads:
        out = out.squeeze(1)
    return out
