# Copyright (c) 2024, mamba-triton-ascend authors.
# Backward selective scan Triton kernel for Ascend NPU.
# Grid=(batch, dim), each program handles all dstate sequentially.
# No atomic_add anywhere.

import torch
import triton
import triton.language as tl


@triton.jit
def selective_scan_bwd_kernel(
    u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, grad_y_ptr,
    du_partial_ptr, ddelta_partial_ptr, dB_partial_ptr, dC_partial_ptr,
    dA_partial_ptr,
    h_temp_ptr, a_temp_ptr, carry_ptr, delta_bias_ptr,
    batch, dim, dstate, seqlen, num_blocks,
    stride_u_batch, stride_u_dim, stride_u_seqlen,
    stride_delta_batch, stride_delta_dim, stride_delta_seqlen,
    stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_dstate, stride_B_seqlen,
    stride_C_batch, stride_C_dstate, stride_C_seqlen,
    stride_gy_batch, stride_gy_dim, stride_gy_seqlen,
    # carry: (batch*dim, dstate, num_blocks)
    stride_carry_bd, stride_carry_n, stride_carry_block,
    # temp: (batch, dim, num_blocks, BLOCK_L, dstate)
    stride_temp_b, stride_temp_d, stride_temp_block, stride_temp_l, stride_temp_n,
    # partial: (batch, dim, seqlen, dstate)
    stride_p_b, stride_p_d, stride_p_seq, stride_p_n,
    # dA_partial: (batch, dim, dstate)
    stride_dA_b, stride_dA_d, stride_dA_n,
    BLOCK_L: tl.constexpr,
    HAS_DELTA_BIAS: tl.constexpr,
    DELTA_SOFTPLUS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    if pid_b >= batch or pid_d >= dim:
        return

    b = pid_b
    d = pid_d
    bd = b * dim + d

    delta_base = delta_ptr + b * stride_delta_batch + d * stride_delta_dim
    u_base = u_ptr + b * stride_u_batch + d * stride_u_dim
    grad_y_base = grad_y_ptr + b * stride_gy_batch + d * stride_gy_dim

    delta_bias_val = 0.0
    if HAS_DELTA_BIAS == 1:
        delta_bias_val = tl.load(delta_bias_ptr + d).to(tl.float32)

    temp_base_bd = b * stride_temp_b + d * stride_temp_d

    # =====================================================================
    # Phase 1: Forward recompute h and a for all blocks, store to temp buffer
    # =====================================================================
    for n in range(dstate):
        A_n = tl.load(A_ptr + d * stride_A_dim + n * stride_A_dstate).to(tl.float32)
        B_base_n = B_ptr + b * stride_B_batch + n * stride_B_dstate

        for block_id in range(num_blocks):
            t_start = block_id * BLOCK_L

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

                    a_i = tl.exp(delta_val * A_n)
                    x_i = delta_val * B_val * u_val
                    h_i = a_i * h_prev + x_i

                    temp_off = temp_base_bd + block_id * stride_temp_block + i * stride_temp_l + n * stride_temp_n
                    tl.store(h_temp_ptr + temp_off, h_i)
                    tl.store(a_temp_ptr + temp_off, a_i)

                    h_prev = h_i

    # =====================================================================
    # Phase 2: Backward scan block by block (reverse order)
    # =====================================================================
    for n in range(dstate):
        A_n = tl.load(A_ptr + d * stride_A_dim + n * stride_A_dstate).to(tl.float32)
        B_base_n = B_ptr + b * stride_B_batch + n * stride_B_dstate
        C_base_n = C_ptr + b * stride_C_batch + n * stride_C_dstate

        dA_n = 0.0
        dH_next = 0.0

        for block_id_rev in range(num_blocks):
            block_id = num_blocks - 1 - block_id_rev
            t_start = block_id * BLOCK_L

            if block_id > 0:
                carry_off = bd * stride_carry_bd + n * stride_carry_n + (block_id - 1) * stride_carry_block
                h_start = tl.load(carry_ptr + carry_off).to(tl.float32)
            else:
                h_start = 0.0

            for i in tl.static_range(BLOCK_L):
                pos = t_start + (BLOCK_L - 1 - i)
                if pos >= t_start and pos < seqlen:
                    idx = pos - t_start

                    temp_off = temp_base_bd + block_id * stride_temp_block + idx * stride_temp_l + n * stride_temp_n
                    h_i = tl.load(h_temp_ptr + temp_off).to(tl.float32)
                    a_i = tl.load(a_temp_ptr + temp_off).to(tl.float32)

                    if idx > 0:
                        h_prev_i = tl.load(h_temp_ptr + temp_off - stride_temp_l).to(tl.float32)
                    else:
                        h_prev_i = h_start

                    # a_next
                    a_next = 0.0
                    if pos < seqlen - 1:
                        if idx < BLOCK_L - 1:
                            a_next = tl.load(a_temp_ptr + temp_off + stride_temp_l).to(tl.float32)
                        else:
                            next_block = block_id + 1
                            if next_block < num_blocks:
                                next_temp_off = temp_base_bd + next_block * stride_temp_block + 0 * stride_temp_l + n * stride_temp_n
                                a_next = tl.load(a_temp_ptr + next_temp_off).to(tl.float32)

                    delta_raw = tl.load(delta_base + pos * stride_delta_seqlen).to(tl.float32)
                    delta_val = delta_raw
                    delta_sigmoid = 1.0
                    if HAS_DELTA_BIAS == 1:
                        delta_val += delta_bias_val
                    if DELTA_SOFTPLUS == 1:
                        delta_sigmoid = tl.sigmoid(delta_val)
                        delta_val = tl.where(
                            delta_val <= 20.0,
                            tl.math.log1p(tl.exp(delta_val)),
                            delta_val
                        )

                    u_val = tl.load(u_base + pos * stride_u_seqlen).to(tl.float32)
                    B_val = tl.load(B_base_n + pos * stride_B_seqlen).to(tl.float32)
                    C_val = tl.load(C_base_n + pos * stride_C_seqlen).to(tl.float32)
                    grad_y_val = tl.load(grad_y_base + pos * stride_gy_seqlen).to(tl.float32)

                    dH_i = grad_y_val * C_val + dH_next * a_next
                    du_val = dH_i * B_val * delta_val
                    ddelta_val = dH_i * B_val * u_val + dH_i * h_prev_i * A_n * a_i
                    ddelta_val *= delta_sigmoid
                    dA_n += dH_i * h_prev_i * delta_val * a_i
                    dB_val = dH_i * u_val * delta_val
                    dC_val = grad_y_val * h_i

                    partial_off = b * stride_p_b + d * stride_p_d + pos * stride_p_seq + n * stride_p_n
                    tl.store(du_partial_ptr + partial_off, du_val)
                    tl.store(ddelta_partial_ptr + partial_off, ddelta_val)
                    tl.store(dB_partial_ptr + partial_off, dB_val)
                    tl.store(dC_partial_ptr + partial_off, dC_val)

                    dH_next = dH_i

        dA_off = b * stride_dA_b + d * stride_dA_d + n * stride_dA_n
        tl.store(dA_partial_ptr + dA_off, dA_n)


def selective_scan_bwd(
    u, delta, A, B, C,
    grad_y, carry,
    delta_bias=None, delta_softplus=False,
    BLOCK_L=16,
):
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    u = u.contiguous()
    delta = delta.contiguous()
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    grad_y = grad_y.contiguous()

    num_blocks = (seqlen + BLOCK_L - 1) // BLOCK_L

    # Partial buffers: (batch, dim, seqlen, dstate)
    du_partial = torch.zeros(batch, dim, seqlen, dstate, dtype=torch.float32, device=u.device)
    ddelta_partial = torch.zeros(batch, dim, seqlen, dstate, dtype=torch.float32, device=u.device)
    dB_partial = torch.zeros(batch, dim, seqlen, dstate, dtype=torch.float32, device=u.device)
    dC_partial = torch.zeros(batch, dim, seqlen, dstate, dtype=torch.float32, device=u.device)
    dA_partial = torch.zeros(batch, dim, dstate, dtype=torch.float32, device=u.device)

    # Temp buffers: (batch, dim, num_blocks, BLOCK_L, dstate)
    h_temp = torch.empty(batch, dim, num_blocks, BLOCK_L, dstate, dtype=torch.float32, device=u.device)
    a_temp = torch.empty(batch, dim, num_blocks, BLOCK_L, dstate, dtype=torch.float32, device=u.device)

    grid = (batch, dim)

    _has_delta_bias = 1 if delta_bias is not None else 0
    _delta_softplus = 1 if delta_softplus else 0

    selective_scan_bwd_kernel[grid](
        u, delta, A, B, C, grad_y,
        du_partial, ddelta_partial, dB_partial, dC_partial,
        dA_partial,
        h_temp, a_temp,
        carry, delta_bias,
        batch, dim, dstate, seqlen, num_blocks,
        u.stride(0), u.stride(1), u.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        grad_y.stride(0), grad_y.stride(1), grad_y.stride(2),
        carry.stride(0), carry.stride(1), carry.stride(2),
        h_temp.stride(0), h_temp.stride(1), h_temp.stride(2), h_temp.stride(3), h_temp.stride(4),
        du_partial.stride(0), du_partial.stride(1), du_partial.stride(2), du_partial.stride(3),
        dA_partial.stride(0), dA_partial.stride(1), dA_partial.stride(2),
        BLOCK_L=BLOCK_L,
        HAS_DELTA_BIAS=_has_delta_bias,
        DELTA_SOFTPLUS=_delta_softplus,
    )

    # Sum partial buffers to get final gradients
    du = du_partial.sum(dim=3)       # sum over dstate (last dim)
    ddelta = ddelta_partial.sum(dim=3)
    dB = dB_partial.sum(dim=1)       # sum over dim -> (batch, seqlen, dstate)
    dC = dC_partial.sum(dim=1)
    dA = dA_partial.sum(dim=0)       # sum over batch -> (dim, dstate)

    # dB/dC need shape (batch, dstate, seqlen)
    dB = dB.permute(0, 2, 1)
    dC = dC.permute(0, 2, 1)

    if delta_bias is not None:
        ddelta_bias = ddelta_partial.sum(dim=(0, 2, 3))  # (dim,)
    else:
        ddelta_bias = None

    return du, ddelta, dA, dB, dC, ddelta_bias
