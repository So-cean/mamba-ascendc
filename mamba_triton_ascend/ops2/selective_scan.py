# Copyright (c) 2024, mamba-triton-ascend authors.
# Selective scan top-level interface (training forward + backward).
# Compatible with mamba_ssm.ops.selective_scan_fn.
# Based on sustcsonglin/mamba-triton with associative_scan.

import torch
import torch.nn.functional as F

from mamba_triton_ascend.ops2.triton_kernels.selective_scan_fwd import selective_scan_fwd
from mamba_triton_ascend.ops.reference import selective_scan_ref


class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None,
                delta_bias=None, delta_softplus=False,
                return_last_state=False):
        # Ensure contiguous
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()

        # Handle B/C dimensions for mamba-ssm compatibility.
        if B.dim() == 4:
            if B.shape[1] != 1:
                raise ValueError(f"B has ngroups={B.shape[1]} but only ngroups=1 is supported")
            B = B.squeeze(1)
            ctx.squeeze_B = True
        else:
            ctx.squeeze_B = False

        if C.dim() == 4:
            if C.shape[1] != 1:
                raise ValueError(f"C has ngroups={C.shape[1]} but only ngroups=1 is supported")
            C = C.squeeze(1)
            ctx.squeeze_C = True
        else:
            ctx.squeeze_C = False

        # Transpose from (B, D, L) to (B, L, D) for kernel
        u_t = u.transpose(-1, -2).contiguous()
        delta_t = delta.transpose(-1, -2).contiguous()
        B_t = B.transpose(-1, -2).contiguous()
        C_t = C.transpose(-1, -2).contiguous()

        # Call forward kernel (delta_bias/delta_softplus handled inside)
        scan_out = selective_scan_fwd(
            u_t, delta_t, A, B_t, C_t,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
        )

        # Transpose back to (B, D, L)
        scan_out = scan_out.transpose(-1, -2).contiguous()

        # Apply D skip connection on host
        if D is not None:
            out = scan_out + D.unsqueeze(-1) * u
        else:
            out = scan_out

        # Apply z gating on host
        if z is not None:
            out_z = out * F.silu(z)
        else:
            out_z = out

        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.has_D = D is not None

        # Save for backward
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias)

        # Compute last_state from reference (simpler than adding to kernel)
        if return_last_state:
            with torch.no_grad():
                _, last_state = selective_scan_ref(
                    u, delta, A, B, C,
                    D=None, z=None,
                    delta_bias=delta_bias,
                    delta_softplus=delta_softplus,
                    return_last_state=True,
                )
            return (out_z, last_state)
        else:
            return out_z

    @staticmethod
    def backward(ctx, dout, *args):
        # TODO: implement backward
        raise NotImplementedError("Backward for ops2 not yet implemented")


def selective_scan_fn(
    u, delta, A, B, C, D=None, z=None,
    delta_bias=None, delta_softplus=False,
    return_last_state=False,
):
    """
    Selective scan function with autograd support.
    Compatible with mamba_ssm.ops.selective_scan_fn.

    Args:
        u: (batch, dim, seqlen)
        delta: (batch, dim, seqlen)
        A: (dim, dstate)
        B: (batch, dstate, seqlen) or (batch, 1, dstate, seqlen)
        C: same as B
        D: (dim,)
        z: (batch, dim, seqlen)
        delta_bias: (dim,)
        delta_softplus: bool
        return_last_state: bool

    Returns:
        out: (batch, dim, seqlen)
        last_state (optional): (batch, dim, dstate)
    """
    return SelectiveScanFn.apply(
        u, delta, A, B, C, D, z,
        delta_bias, delta_softplus, return_last_state
    )
