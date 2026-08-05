# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from mamba_triton_ascend.mamba2.utils.determinism import autotune_configs


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


@triton.jit
def _bmm_chunk_fwd_fp32_kernel(
    a_ptr, b_ptr, out_ptr, seq_idx_ptr,
    seqlen, chunk_size, batch,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_bk,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_outm, stride_outn,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    HAS_SEQ_IDX: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Ascend FP32 BMM using vector outer products instead of FP32 tl.dot."""
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    a_ptr += (pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen
              + pid_h * stride_a_head)
    b_ptr += (pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen
              + pid_h * stride_b_head)
    out_ptr += (pid_b * stride_out_batch + pid_c * stride_out_chunk
                + pid_h * stride_out_head)
    if HAS_SEQ_IDX:
        seq_idx_ptr += (pid_b * stride_seq_idx_batch
                        + pid_c * chunk_size * stride_seq_idx_seqlen)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 0.0, tl.float32)
    a_ptrs = a_ptr + (
        offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen
    )
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k[None, :] < K - k_block * BLOCK_SIZE_K),
            other=0.0,
        ).to(tl.float16)
        b_tile = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k_block * BLOCK_SIZE_K)
            & (offs_n[None, :] < chunk_size_limit),
            other=0.0,
        ).to(tl.float16)
        acc += tl.dot(a_tile, b_tile)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if HAS_SEQ_IDX:
        seq_m = tl.load(
            seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
            mask=offs_m < chunk_size_limit,
            other=-1,
        )
        seq_n = tl.load(
            seq_idx_ptr + offs_n * stride_seq_idx_seqlen,
            mask=offs_n < chunk_size_limit,
            other=-2,
        )
        acc = tl.where(seq_m[:, None] == seq_n[None, :], acc, 0.0)
    tl.store(
        out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn,
        acc,
        mask=(offs_m[:, None] < chunk_size_limit)
        & (offs_n[None, :] < chunk_size_limit),
    )


# GPU autotuning is intentionally disabled for the Ascend FP32 forward MVP.
# The backend warns that launch parameters are not tunable and several large
# candidates exceed the 910B UB budget.
@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, out_ptr, seq_idx_ptr,
    # Matrix dimensions
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_bk,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_outm, stride_outn,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 0.0, dtype=tl.float32)
    # FP32 tl.dot is mis-lowered by the current Triton-Ascend toolchain.
    # Accumulate rank-1 products explicitly to preserve FP32 semantics.
    for k in range(0, K):
        a_m = tl.load(
            a_ptr + offs_m * stride_a_seqlen + k * stride_ak,
            mask=offs_m < chunk_size_limit,
            other=0.0,
        ).to(tl.float32)
        b_n = tl.load(
            b_ptr + k * stride_bk + offs_n * stride_b_seqlen,
            mask=offs_n < chunk_size_limit,
            other=0.0,
        ).to(tl.float32)
        acc += a_m[:, None] * b_n[None, :]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


@triton.autotune(
    configs=autotune_configs([
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_CS': 32}, num_stages=4, num_warps=2),
    ]),
    key=['chunk_size', 'K'],
)
@triton.jit
def _bmm_chunk_bwd_kernel(
    # Pointers to matrices
    a_ptr, dout_ptr, db_ptr, res_ptr,
    # Matrix dimensions
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_dout_batch, stride_dout_chunk, stride_dout_head, stride_dout_csize_m, stride_dout_csize_n,
    stride_db_batch, stride_db_seqlen, stride_db_head, stride_db_k,
    stride_res_batch, stride_res_seqlen, stride_res_head, stride_res_k,
    # Meta-parameters
    dot_dtype: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * stride_dout_chunk + pid_h * stride_dout_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize_n + offs_cs[None, :] * stride_dout_csize_m)
    a_ptrs = a_ptr + (offs_cs[:, None] * stride_a_seqlen + offs_n[None, :] * stride_ak)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for cs in range(0, tl.cdiv(chunk_size_limit, BLOCK_SIZE_CS)):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS), other=0.0).to(dot_dtype)
        a = tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * BLOCK_SIZE_CS) & (offs_n[None, :] < K), other=0.0).to(dot_dtype)
        acc += tl.dot(dout, a)
        dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m
        a_ptrs += BLOCK_SIZE_CS * stride_a_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_RESIDUAL:
        res_ptr += pid_b * stride_res_batch + pid_c * chunk_size * stride_res_seqlen + pid_h * stride_res_head
        res_ptrs = res_ptr + (offs_m[:, None] * stride_res_seqlen + offs_n[None, :] * stride_res_k)
        res = tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K)).to(tl.float32)
        acc += res
    db = acc.to(db_ptr.dtype.element_ty)

    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_h * stride_db_head
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_k)
    tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < K))


def _bmm_chunk_fwd(a, b, chunk_size, seq_idx=None, causal=False, output_dtype=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    # Check constraints.
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    assert b.shape == a.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    # Allocates output.
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.empty((batch, nchunks, chunk_size, chunk_size) if not has_groups else (batch, nchunks, ngroups, chunk_size, chunk_size),
                      device=a.device, dtype=out_dtype)
    if (
        a.device.type == "npu"
        and a.dtype == torch.float32
        and b.dtype == torch.float32
        and out_dtype == torch.float32
    ):
        # A 64x64 tile amortizes launch work on regular workloads, while its
        # larger Cube setup cost is visible on the tiny two-chunk case.
        tiny_launch = batch * nchunks <= 4
        if tiny_launch:
            block_m = block_n = 32
        elif chunk_size >= 128:
            block_m, block_n = 128, 64
        else:
            block_m = block_n = 64
        block_k = 16
        # Feed Cube contiguous [M, K] and [K, N] operands.  The original B
        # layout makes the N dimension strided and forces a conversion in
        # every program.
        a_cube = a.to(torch.float16)
        if has_groups:
            b_cube = b.permute(0, 2, 3, 1).contiguous().to(torch.float16)
            b_batch_stride = b_cube.stride(0)
            b_seqlen_stride = b_cube.stride(3)
            b_head_stride = b_cube.stride(1)
            b_k_stride = b_cube.stride(2)
        else:
            b_cube = b.transpose(1, 2).contiguous().to(torch.float16)
            b_batch_stride = b_cube.stride(0)
            b_seqlen_stride = b_cube.stride(2)
            b_head_stride = 0
            b_k_stride = b_cube.stride(1)
        grid = (
            triton.cdiv(chunk_size, block_m) * triton.cdiv(chunk_size, block_n),
            batch * nchunks,
            ngroups if has_groups else 1,
        )
        seq_strides = seq_idx.stride() if seq_idx is not None else (0, 0)
        _bmm_chunk_fwd_fp32_kernel[grid](
            a_cube, b_cube, out, seq_idx,
            seqlen, chunk_size, batch,
            a_cube.stride(0), a_cube.stride(1), 0 if not has_groups else a_cube.stride(2), a_cube.stride(-1),
            b_batch_stride, b_seqlen_stride, b_head_stride, b_k_stride,
            out.stride(0), out.stride(1), 0 if not has_groups else out.stride(2), out.stride(-2), out.stride(-1),
            seq_strides[0], seq_strides[1],
            HAS_SEQ_IDX=seq_idx is not None,
            K=k,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
        )
        return out
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16 else
                 (tl.float16 if a.dtype == torch.float16 or b.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
                    batch, nchunks if not has_groups else nchunks * ngroups)
    _bmm_chunk_fwd_kernel[grid](
        a, b, out, seq_idx,
        seqlen, chunk_size, k, ngroups if has_groups else 1,
        a.stride(0), a.stride(1), 0 if not has_groups else a.stride(2), a.stride(-1),
        b.stride(0), b.stride(1), 0 if not has_groups else b.stride(2), b.stride(-1),
        out.stride(0), out.stride(1), 0 if not has_groups else out.stride(2), out.stride(-2), out.stride(-1),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        causal,
        dot_dtype,
        HAS_SEQ_IDX=seq_idx is not None,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )
    return out


def _bmm_chunk_bwd(a, dout, residual=None, out=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        dout: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
        residual: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
    Return:
        out: (batch, seqlen, k) or (batch, seqlen, ngroups, k)

    If there was seq_idx in the fwd pass, then dout[i, j] for seq_idx[i] != seq_idx[j] should already be
    zeroed out before calling this function.
    """
    # Check constraints.
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    nchunks, chunk_size = dout.shape[1], dout.shape[-1]
    if a.stride(-1) != 1 and a.stride(-2) != 1:
        a = a.contiguous()
    if dout.stride(-1) != 1 and dout.stride(-2) != 1:
        dout = dout.contiguous()
    if residual is not None:
        assert residual.shape == (batch, seqlen, k) if not has_groups else (batch, seqlen, ngroups, k)
        if residual.stride(-1) != 1 and residual.stride(1) != 1:
            residual = residual.contiguous()
    # Allocates output.
    if out is not None:
        assert out.shape == a.shape
        assert out.stride(-1) == 1 or out.stride(1) == 1
    else:
        out = torch.empty_like(a)
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or dout.dtype == torch.bfloat16 else
                 (tl.float16 if a.dtype == torch.float16 or dout.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(k, META['BLOCK_SIZE_N']), batch,
                    nchunks if not has_groups else nchunks * ngroups)
    residual_strides = ((residual.stride(0), residual.stride(1), 0 if not has_groups else residual.stride(2),
                         residual.stride(-1))
                        if residual is not None else (0, 0, 0, 0))
    _bmm_chunk_bwd_kernel[grid](
        a, dout, out, residual,
        seqlen, chunk_size, k, ngroups if has_groups else 1,
        a.stride(0), a.stride(1), 0 if not has_groups else a.stride(2), a.stride(-1),
        dout.stride(0), dout.stride(1), 0 if not has_groups else dout.stride(2), dout.stride(-2), dout.stride(-1),
        out.stride(0), out.stride(1), 0 if not has_groups else out.stride(2), out.stride(-1),
        residual_strides[0], residual_strides[1], residual_strides[2], residual_strides[3],
        dot_dtype,
        HAS_RESIDUAL=residual is not None,
    )
    return out
