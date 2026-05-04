# Copyright (c) 2024, mamba-triton-ascend authors.
# PyTorch reference implementations for correctness verification.

import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_scan_ref(
    u, delta, A, B, C, D=None, z=None,
    delta_bias=None, delta_softplus=False,
    return_last_state=False,
):
    """
    PyTorch reference implementation of selective scan.
    Compatible with mamba-ssm selective_scan_ref interface.

    u: (batch, dim, seqlen)
    delta: (batch, dim, seqlen)
    A: (dim, dstate) or (nheads, dim, dstate) - fixed params
    B: (batch, dstate, seqlen) or (batch, ngroups, dstate, seqlen) - variable
    C: same shape as B
    D: (dim,) or (nheads, dim)
    z: (batch, dim, seqlen)
    delta_bias: (dim,) or (nheads, dim)

    Returns:
        out: (batch, dim, seqlen)
        last_state (optional): (batch, dim, dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    if D is not None:
        D = D.float()
    if z is not None:
        z = z.float()

    # Handle delta_bias and softplus
    # Use new variable name to preserve autograd graph for original inputs
    delta_used = delta
    if delta_bias is not None:
        delta_used = delta_used + delta_bias[..., None].float()
    if delta_softplus:
        delta_used = F.softplus(delta_used)

    batch, dim, seqlen = u.shape
    dstate = A.shape[-1]

    # Determine if B/C are variable (per-step)
    # Variable B/C: shape has seqlen dimension at last axis
    is_variable_B = B.dim() >= 3 and B.shape[-1] == seqlen
    is_variable_C = C.dim() >= 3 and C.shape[-1] == seqlen

    # Handle head dimension if present
    has_heads = A.dim() == 3
    if has_heads:
        nheads = A.shape[0]
        assert nheads == 1, "Current version only supports nheads=1"
        A_used = A.squeeze(0)  # (dim, dstate)
        if D is not None:
            D_used = D.squeeze(0)  # (dim,)
        else:
            D_used = None
        if delta_bias is not None:
            delta_bias_used = delta_bias.squeeze(0)
        else:
            delta_bias_used = None
    else:
        A_used = A
        D_used = D
        delta_bias_used = delta_bias

    # Handle group dimension if present
    if is_variable_B and B.dim() == 4:
        ngroups = B.shape[1]
        assert ngroups == 1, "Current version only supports ngroups=1"
        B_used = B.squeeze(1)  # (batch, dstate, seqlen)
    else:
        B_used = B
    if is_variable_C and C.dim() == 4:
        ngroups = C.shape[1]
        assert ngroups == 1, "Current version only supports ngroups=1"
        C_used = C.squeeze(1)  # (batch, dstate, seqlen)
    else:
        C_used = C

    # Initialize hidden state
    x = A.new_zeros((batch, dim, dstate))
    ys = []

    # Precompute deltaA = exp(delta_used * A_used)
    # A_used: (dim, dstate), delta_used: (batch, dim, seqlen)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta_used, A_used))  # (batch, dim, seqlen, dstate)

    # Compute deltaB_u = delta_used * B_used * u
    if not is_variable_B:
        # B_used: (dim, dstate)
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta_used, B_used, u)
    else:
        # B_used: (batch, dstate, seqlen)
        deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta_used, B_used, u)

    # Pre-process C_used if variable and has groups
    if is_variable_C and C_used.dim() == 4:
        C_used = repeat(C_used, "B G N L -> B (G H) N L", H=dim // C_used.shape[1])

    last_state = None
    for i in range(seqlen):
        # State update: x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]

        # Output: y = sum(x * C_used)
        if not is_variable_C:
            # C_used: (dim, dstate)
            y = torch.einsum('bdn,dn->bd', x, C_used)
        else:
            if C_used.dim() == 3:
                # C_used: (batch, dstate, seqlen)
                y = torch.einsum('bdn,bn->bd', x, C_used[:, :, i])
            else:
                # C_used: (batch, dim, dstate, seqlen) after repeat
                y = torch.einsum('bdn,bdn->bd', x, C_used[:, :, :, i])

        if i == seqlen - 1:
            last_state = x

        ys.append(y)

    y = torch.stack(ys, dim=2)  # (batch, dim, seqlen)

    # D skip connection
    if D_used is not None:
        out = y + u * rearrange(D_used, "d -> d 1")
    else:
        out = y

    # Z gating
    if z is not None:
        out = out * F.silu(z)

    out = out.to(dtype=dtype_in)

    if not return_last_state:
        return out
    else:
        return out, last_state


def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None,
    dt_bias=None, dt_softplus=False,
):
    """
    PyTorch reference for single-step selective state update (inference).

    state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
    x: (batch, dim) or (batch, nheads, dim)
    dt: (batch, dim) or (batch, nheads, dim)
    A: (dim, dstate) or (nheads, dim, dstate)
    B: (batch, dstate) or (batch, ngroups, dstate)
    C: same as B
    D: (dim,) or (nheads, dim)
    z: (batch, dim) or (batch, nheads, dim)
    dt_bias: (dim,) or (nheads, dim)
    """
    has_heads = state.dim() > 3

    # Unsqueeze to 4D/3D if needed
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
    assert nheads % ngroups == 0
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape

    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)

    # Compute dA = exp(dt * A)
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)

    # Expand B, C to match nheads
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)

    # Compute dB = dt * B
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")

    # State update (in-place)
    state.copy_(state * dA + dB * rearrange(x, "b h d -> b h d 1"))

    # Output
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)

    if D is not None:
        out += (x * D).to(out.dtype)

    if z is not None:
        out = out * F.silu(z)

    out = out.to(x.dtype)

    if not has_heads:
        out = out.squeeze(1)

    return out
