# Copyright (c) 2026, mamba-triton-ascend authors.
# PyTorch reference for Mamba-2 SSD chunk scan.
# Interface compatible with mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined.
# No einops dependency; all operations use vanilla torch.

import torch
import torch.nn.functional as F


def ssd_sequential_ref(
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    initial_states=None,
    return_final_state=False,
):
    """Dtype-preserving sequential SSD oracle for gradcheck-sized inputs.

    Unlike :func:`ssd_chunk_scan_ref`, this helper never forces FP32.  It is
    intentionally sequential and is used for FP64 directional derivatives,
    not performance measurements.
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[2:]
    if nheads % ngroups:
        raise ValueError("nheads must be divisible by ngroups")
    heads_per_group = nheads // ngroups

    q = dt
    if dt_bias is not None:
        q = q + dt_bias.view(1, 1, nheads)
    if dt_softplus:
        q = F.softplus(q)
    q = q.clamp(min=dt_limit[0], max=dt_limit[1])
    b_heads = B.repeat_interleave(heads_per_group, dim=2)
    c_heads = C.repeat_interleave(heads_per_group, dim=2)
    state = (
        torch.zeros(
            batch,
            nheads,
            headdim,
            dstate,
            dtype=x.dtype,
            device=x.device,
        )
        if initial_states is None
        else initial_states
    )
    outputs = []
    for index in range(seqlen):
        q_t = q[:, index]
        decay = torch.exp(q_t * A.view(1, nheads))
        v = x[:, index] * q_t.unsqueeze(-1)
        state = (
            decay.unsqueeze(-1).unsqueeze(-1) * state
            + v.unsqueeze(-1) * b_heads[:, index].unsqueeze(-2)
        )
        y = (state * c_heads[:, index].unsqueeze(-2)).sum(dim=-1)
        if D is not None:
            if D.ndim == 1:
                y = y + x[:, index] * D.view(1, nheads, 1)
            else:
                y = y + x[:, index] * D.unsqueeze(0)
        if z is not None:
            y = y * F.silu(z[:, index])
        outputs.append(y)
    out = torch.stack(outputs, dim=1)
    if return_final_state:
        return out, state
    return out


def segsum(x, device=None):
    """Stable segment sum: segsum(x)[i,j] = sum_{k=j+1}^{i} x[k] for i >= j.

    The diagonal (i=j) evaluates to 0, so exp(segsum(x))[i,i] = 1.
    Upper-triangular entries (i<j) are set to -inf.

    Args:
        x: (..., T)
        device: torch device for mask creation (defaults to x.device)
    Returns:
        (..., T, T)
    """
    if device is None:
        device = x.device
    T = x.size(-1)

    # Expand last dim: (..., T) -> (..., T, T)  [replaces einops repeat]
    x_exp = x.unsqueeze(-1).expand(*x.shape, T)

    # Lower-triangular mask (excluding diagonal) for cumsum
    mask_lo = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x_exp = x_exp.masked_fill(~mask_lo, 0)

    # Cumsum along the expanded (-2) axis
    x_segsum = torch.cumsum(x_exp, dim=-2)

    # Lower-triangular mask (including diagonal) for valid entries
    mask_lo_diag = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask_lo_diag, -torch.inf)

    return x_segsum


def _ssd_core(x, A, B, C, chunk_size, initial_states=None):
    """Core SSD computation on pre-processed per-head inputs.

    This is equivalent to `ssd_minimal_discrete` from mamba_ssm,
    rewritten without einops.

    Args:
        x: (batch, seqlen, nheads, headdim) -- already multiplied by dt
        A: (batch, seqlen, nheads) -- dA = dt * A_real, discretized
        B: (batch, seqlen, nheads, dstate) -- already expanded to per-head
        C: (batch, seqlen, nheads, dstate) -- already expanded to per-head
        chunk_size: int
        initial_states: (batch, 1, nheads, headdim, dstate) or None

    Returns:
        y:          (batch, seqlen, nheads, headdim)
        final_state: (batch, 1, nheads, headdim, dstate)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    assert seqlen % chunk_size == 0, "seqlen must be divisible by chunk_size"
    num_chunks = seqlen // chunk_size

    # Rearrange into chunks: (b, l, ...) -> (b, c, l, ...)
    # Equivalent to einops: rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size)
    x_c = x.reshape(batch, num_chunks, chunk_size, nheads, headdim)
    A_c = A.reshape(batch, num_chunks, chunk_size, nheads)
    B_c = B.reshape(batch, num_chunks, chunk_size, nheads, dstate)
    C_c = C.reshape(batch, num_chunks, chunk_size, nheads, dstate)

    # A: (b, c, l, h) -> (b, h, c, l) for inter-chunk cumsum
    # Equivalent to einops: rearrange(A, "b c l h -> b h c l")
    A_c = A_c.permute(0, 3, 1, 2)  # (batch, nheads, num_chunks, chunk_size)
    A_cumsum = torch.cumsum(A_c, dim=-1)

    # === 1. Intra-chunk (diagonal blocks) ===
    # L: (batch, nheads, num_chunks, chunk_size, chunk_size)
    L = torch.exp(segsum(A_c, device=x.device))
    # Y_diag: (batch, num_chunks, chunk_size, nheads, headdim)
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C_c, B_c, L, x_c)

    # === 2. Chunk final states ===
    # decay_states: (batch, nheads, num_chunks, chunk_size)
    #  = exp(A_cumsum[:, -1] - A_cumsum[:, t]) for each step t
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # states: (batch, num_chunks, nheads, headdim, dstate)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B_c, decay_states, x_c)

    # === 3. Inter-chunk recurrence (state passing) ===
    if initial_states is None:
        initial_states = torch.zeros(batch, 1, nheads, headdim, dstate,
                                     dtype=states.dtype, device=states.device)
    # states_ext: (batch, num_chunks+1, nheads, headdim, dstate)
    states_ext = torch.cat([initial_states, states], dim=1)

    # decay_chunk_last: (batch, nheads, num_chunks)
    decay_chunk_last = A_cumsum[:, :, :, -1]
    # Pad a leading 0: (batch, nheads, num_chunks+1)
    decay_chunk_pad = F.pad(decay_chunk_last, (1, 0))
    # decay_chunk: (batch, nheads, num_chunks+1, num_chunks+1)
    decay_chunk = torch.exp(segsum(decay_chunk_pad, device=x.device))

    # new_states: (batch, num_chunks+1, nheads, headdim, dstate)
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states_ext)
    states_new = new_states[:, :-1]       # (batch, num_chunks, nheads, headdim, dstate)
    final_state = new_states[:, -1:]      # (batch, 1, nheads, headdim, dstate)

    # === 4. State -> output (off-diagonal blocks) ===
    # state_decay_out: (batch, nheads, num_chunks, chunk_size)
    state_decay_out = torch.exp(A_cumsum)
    # Y_off: (batch, num_chunks, chunk_size, nheads, headdim)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C_c, states_new, state_decay_out)

    # Combine and reshape: (b, c, l, h, p) -> (b, seqlen, h, p)
    Y = (Y_diag + Y_off).reshape(batch, seqlen, nheads, headdim)

    return Y, final_state


def ssd_chunk_scan_ref(
    x, dt, A, B, C, chunk_size,
    D=None, z=None, dt_bias=None,
    dt_softplus=False, dt_limit=(0.0, float("inf")),
    initial_states=None,
    return_final_state=False,
):
    """Mamba-2 SSD chunk scan – pure PyTorch reference.

    Interface compatible with:
        mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined

    Args:
        x:           (batch, seqlen, nheads, headdim)
        dt:          (batch, seqlen, nheads)
        A:           (nheads,) – negative decay value (e.g. -exp(A_log))
        B:           (batch, seqlen, ngroups, dstate)
        C:           (batch, seqlen, ngroups, dstate)
        chunk_size:  int  (e.g., 64 or 128)
        D:           (nheads, headdim) or (nheads,) or None
        z:           (batch, seqlen, nheads, headdim) or None
        dt_bias:     (nheads,) or None
        dt_softplus: bool
        initial_states: (batch, nheads, headdim, dstate) or None
        return_final_state: bool

    Returns:
        out: (batch, seqlen, nheads, headdim)
        [final_state]: (batch, nheads, headdim, dstate) if return_final_state
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0, f"nheads={nheads} not divisible by ngroups={ngroups}"
    nheads_per_group = nheads // ngroups

    # Save input dtype for output cast
    dtype_in = x.dtype
    x_f = x.float()
    dt_f = dt.float()
    B_f = B.float()
    C_f = C.float()
    z_f = z.float() if z is not None else None
    initial_states_f = initial_states.float() if initial_states is not None else None

    # ---- Process dt ----
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float().view(1, 1, -1)
    if dt_softplus:
        dt_f = F.softplus(dt_f)
    dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])

    # Discretize: dA = dt * A, x = x * dt
    # A is expected as a negative decay value (e.g. -exp(A_log)),
    # matching mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined interface
    A_f = A.float().view(1, 1, -1)                   # (1, 1, nheads)
    dA = dt_f * A_f                                  # (batch, seqlen, nheads)
    x_dt = x_f * dt_f.unsqueeze(-1)                  # (batch, seqlen, nheads, headdim)

    # ---- Pad seqlen to chunk_size multiple ----
    if seqlen % chunk_size != 0:
        pad_len = chunk_size - seqlen % chunk_size
        x_dt = F.pad(x_dt, (0, 0, 0, 0, 0, pad_len))
        dA   = F.pad(dA,   (0, 0, 0, pad_len))
        B_p  = F.pad(B_f,  (0, 0, 0, 0, 0, pad_len))
        C_p  = F.pad(C_f,  (0, 0, 0, 0, 0, pad_len))
        if z_f is not None:
            z_f = F.pad(z_f, (0, 0, 0, 0, 0, pad_len))
    else:
        B_p = B_f
        C_p = C_f

    padded_len = x_dt.shape[1]
    out = torch.zeros(batch, padded_len, nheads, headdim,
                      device=x.device, dtype=torch.float32)

    all_final_states = []

    # Process each group independently (B/C shared within a group)
    for g in range(ngroups):
        h_start = g * nheads_per_group
        h_end   = h_start + nheads_per_group

        # Select heads for this group
        x_g = x_dt[:, :, h_start:h_end, :]                    # (B, L, H_g, P)
        A_g = dA[:, :, h_start:h_end]                          # (B, L, H_g)

        # Expand B/C from per-group to per-head
        # B_p: (B, L, ngroups, N) -> B_g_raw: (B, L, N) -> (B, L, 1, N) -> (B, L, H_g, N)
        B_g_raw = B_p[:, :, g, :]                               # (B, L, N)
        B_g = B_g_raw.unsqueeze(2).expand(-1, -1, nheads_per_group, -1)
        C_g_raw = C_p[:, :, g, :]                               # (B, L, N)
        C_g = C_g_raw.unsqueeze(2).expand(-1, -1, nheads_per_group, -1)

        initial_g = None
        if initial_states_f is not None:
            expected = (batch, nheads, headdim, dstate)
            assert initial_states_f.shape == expected, (
                f"initial_states.shape={initial_states_f.shape}, expected={expected}"
            )
            initial_g = initial_states_f[:, h_start:h_end].unsqueeze(1)

        y_g, final_g = _ssd_core(
            x_g, A_g, B_g, C_g, chunk_size, initial_states=initial_g
        )
        out[:, :, h_start:h_end, :] = y_g
        # final_g: (batch, 1, H_g, headdim, dstate)
        all_final_states.append(final_g)

    # Trim padding
    if seqlen % chunk_size != 0:
        out = out[:, :seqlen]
        if z_f is not None:
            z_f = z_f[:, :seqlen]

    # ---- D skip connection ----
    if D is not None:
        D_f = D.float()
        if D_f.dim() == 2:
            # (nheads, headdim) – broadcast over batch, seqlen
            out = out + x_f * D_f.unsqueeze(0).unsqueeze(0)
        else:
            # (nheads,) – broadcast over batch, seqlen, headdim
            out = out + x_f * D_f.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    # ---- z gating ----
    if z_f is not None:
        out = out * F.silu(z_f)

    out = out.to(dtype=dtype_in)

    if return_final_state:
        final_state = torch.cat(all_final_states, dim=2)  # (batch, 1, nheads, headdim, dstate)
        return out, final_state.squeeze(1)                 # (batch, nheads, headdim, dstate)

    return out
