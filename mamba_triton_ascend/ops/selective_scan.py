# Copyright (c) 2024, mamba-triton-ascend authors.
# Selective scan top-level interface (training forward + backward).
# Compatible with mamba_ssm.ops.selective_scan_fn.

import torch
import torch.nn.functional as F

from mamba_triton_ascend.ops.triton_kernels.selective_scan_fwd import selective_scan_fwd
from mamba_triton_ascend.ops.triton_kernels.selective_scan_bwd import selective_scan_bwd
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
        # Our kernel expects 3D (batch, dstate, seqlen).
        # If 4D (batch, ngroups, dstate, seqlen) with ngroups==1, squeeze.
        if B.dim() == 4:
            if B.shape[1] != 1:
                raise ValueError(f"B has ngroups={B.shape[1]} but only ngroups=1 is supported")
            B = B.squeeze(1)
            ctx.unsqueeze_B = True
        else:
            ctx.unsqueeze_B = False

        if C.dim() == 4:
            if C.shape[1] != 1:
                raise ValueError(f"C has ngroups={C.shape[1]} but only ngroups=1 is supported")
            C = C.squeeze(1)
            ctx.unsqueeze_C = True
        else:
            ctx.unsqueeze_C = False

        # Call forward kernel (computes scan output before D and z)
        scan_out, carry = selective_scan_fwd(
            u, delta, A, B, C,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            BLOCK_L=32,
        )

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
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, carry)

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
        u, delta, A, B, C, D, z, delta_bias, carry = ctx.saved_tensors

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        batch, dim, seqlen = u.shape

        # Step 1: Handle z gating and D skip connection
        if ctx.has_z:
            z_sigmoid = torch.sigmoid(z)
            z_silu = z * z_sigmoid
            dsilu = z_sigmoid * (1 + z * (1 - z_sigmoid))

            # Recompute scan_out (before D and z)
            with torch.no_grad():
                scan_out, _ = selective_scan_fwd(
                    u, delta, A, B, C,
                    delta_bias=delta_bias,
                    delta_softplus=ctx.delta_softplus,
                    BLOCK_L=32,
                )

            # y = scan_out + D*u  (the value before z gating)
            y = scan_out
            if ctx.has_D:
                y = y + D.unsqueeze(-1) * u

            grad_y = dout * z_silu
            dz = dout * y * dsilu
        else:
            grad_y = dout
            dz = None

        # Handle D skip connection
        if ctx.has_D:
            dD = torch.einsum("bdl,bdl->d", grad_y, u)
        else:
            dD = None

        # Step 2: Call backward kernel (grad_y is gradient w.r.t. scan output before D and z)
        du, ddelta, dA, dB, dC, ddelta_bias = selective_scan_bwd(
            u, delta, A, B, C,
            grad_y,
            carry,
            delta_bias=delta_bias,
            delta_softplus=ctx.delta_softplus,
            BLOCK_L=16,
        )

        # Add D contribution to du
        if ctx.has_D:
            du += grad_y * D.unsqueeze(-1)

        # Unsqueeze B/C if needed
        dB = dB.unsqueeze(1) if getattr(ctx, "unsqueeze_B", False) else dB
        dC = dC.unsqueeze(1) if getattr(ctx, "unsqueeze_C", False) else dC

        return (
            du, ddelta, dA, dB, dC,
            dD if D is not None else None,
            dz,
            ddelta_bias if delta_bias is not None else None,
            None,  # delta_softplus
            None,  # return_last_state
        )


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
