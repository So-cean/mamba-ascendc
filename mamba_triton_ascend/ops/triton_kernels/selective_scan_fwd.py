# Copyright (c) 2024, mamba-triton-ascend authors.
# Forward selective scan Triton kernel for Ascend NPU.
# ALL loops are static_range to avoid BiShengIR scf.for iterarg bugs.
# Grid = (batch, dim), each program handles all blocks and dstate sequentially.

import torch
import triton
import triton.language as tl


@triton.jit
def selective_scan_fwd_kernel(
    u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr,
    y_partial_ptr, delta_bias_ptr, carry_ptr,
    batch, dim, dstate, seqlen, num_blocks,
    stride_u_batch, stride_u_dim, stride_u_seqlen,
    stride_delta_batch, stride_delta_dim, stride_delta_seqlen,
    stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_dstate, stride_B_seqlen,
    stride_C_batch, stride_C_dstate, stride_C_seqlen,
    stride_carry_bd, stride_carry_n, stride_carry_block,
    BLOCK_L: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_DSTATE: tl.constexpr,
    HAS_DELTA_BIAS: tl.constexpr,
    DELTA_SOFTPLUS: tl.constexpr,
):
    """
    Grid: (batch, dim)
    All loops are static_range to avoid scf.for codegen bugs on Ascend.
    """
    b = tl.program_id(0)
    d = tl.program_id(1)
    if b >= batch or d >= dim:
        return

    bd = b * dim + d

    delta_base = delta_ptr + b * stride_delta_batch + d * stride_delta_dim
    u_base = u_ptr + b * stride_u_batch + d * stride_u_dim

    delta_bias_val = 0.0
    if HAS_DELTA_BIAS == 1:
        delta_bias_val = tl.load(delta_bias_ptr + d).to(tl.float32)

    for block_id in tl.static_range(MAX_BLOCKS):
        if block_id < num_blocks:
            t_start = block_id * BLOCK_L

            for n in tl.static_range(MAX_DSTATE):
                if n < dstate:
                    A_n = tl.load(A_ptr + d * stride_A_dim + n * stride_A_dstate).to(tl.float32)

                    B_base_n = B_ptr + b * stride_B_batch + n * stride_B_dstate
                    C_base_n = C_ptr + b * stride_C_batch + n * stride_C_dstate

                    if block_id > 0:
                        carry_off = bd * stride_carry_bd + n * stride_carry_n + (block_id - 1) * stride_carry_block
                        h_prev = tl.load(carry_ptr + carry_off).to(tl.float32)
                    else:
                        h_prev = 0.0

                    for i in tl.static_range(BLOCK_L):
                        pos = t_start + i
                        if pos < seqlen:
                            delta_val = tl.load(delta_base + pos * stride_delta_seqlen).to(tl.float32)
                            if HAS_DELTA_BIAS == 1:
                                delta_val += delta_bias_val
                            if DELTA_SOFTPLUS == 1:
                                delta_val = tl.where(
                                    delta_val <= 20.0,
                                    tl.math.log1p(tl.exp(delta_val)),
                                    delta_val
                                )

                            u_val = tl.load(u_base + pos * stride_u_seqlen).to(tl.float32)
                            B_val = tl.load(B_base_n + pos * stride_B_seqlen).to(tl.float32)
                            C_val = tl.load(C_base_n + pos * stride_C_seqlen).to(tl.float32)

                            a_i = tl.exp(delta_val * A_n)
                            x_i = delta_val * B_val * u_val
                            h_i = a_i * h_prev + x_i

                            y_partial_off = ((bd) * dstate + n) * seqlen + pos
                            tl.store(y_partial_ptr + y_partial_off, h_i * C_val)

                            h_prev = h_i

                    carry_off = bd * stride_carry_bd + n * stride_carry_n + block_id * stride_carry_block
                    tl.store(carry_ptr + carry_off, h_prev)


def selective_scan_fwd(
    u, delta, A, B, C,
    delta_bias=None, delta_softplus=False,
    BLOCK_L=32,
):
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    u = u.contiguous()
    delta = delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    y_partial = torch.zeros(batch, dim, dstate, seqlen, dtype=torch.float32, device=u.device)
    num_blocks = (seqlen + BLOCK_L - 1) // BLOCK_L
    carry = torch.empty(batch * dim, dstate, num_blocks, dtype=torch.float32, device=u.device)

    grid = (batch, dim)

    _has_delta_bias = 1 if delta_bias is not None else 0
    _delta_softplus = 1 if delta_softplus else 0
    selective_scan_fwd_kernel[grid](
        u, delta, A, B, C,
        y_partial, delta_bias, carry,
        batch, dim, dstate, seqlen, num_blocks,
        u.stride(0), u.stride(1), u.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        carry.stride(0), carry.stride(1), carry.stride(2),
        BLOCK_L=BLOCK_L, MAX_BLOCKS=num_blocks, MAX_DSTATE=dstate,
        HAS_DELTA_BIAS=_has_delta_bias, DELTA_SOFTPLUS=_delta_softplus,
    )

    out = y_partial.sum(dim=2)
    return out, carry
