#!/usr/bin/env python3
"""Direct precision gate for the experimental T=128 ChunkMix key."""

import ctypes
import os

import pytest
import torch
import torch_npu  # noqa: F401


library_path = os.environ.get("MAMBA_CHUNK_MIX_OP_API_LIB")
if library_path:
    ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
import ascend_kernel  # noqa: E402,F401


def _metrics(actual, expected):
    error = actual.float() - expected.float()
    return (
        error.abs().max().item(),
        (torch.linalg.vector_norm(error) /
         torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)).item(),
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item(),
    )


@pytest.mark.parametrize("batch,heads,chunks,groups", [(1, 2, 1, 1), (1, 4, 2, 2)])
def test_mamba2_ssd_chunk_mix_128(batch, heads, chunks, groups):
    torch.manual_seed(20260802 + heads + chunks)
    tile, head_dim, state_dim = 128, 64, 128
    x = (0.1 * torch.randn(batch, heads, chunks, tile, head_dim)).half()
    da_step = -(0.005 + 0.045 * torch.rand(batch, heads, chunks, tile))
    da = torch.cumsum(da_step, dim=-1)
    b = (0.1 * torch.randn(batch, chunks, groups, tile, state_dim)).half()
    c = (0.1 * torch.randn(batch, chunks, groups, tile, state_dim)).half()

    heads_per_group = heads // groups
    expected_y = torch.empty_like(x, dtype=torch.float32)
    expected_state = torch.empty(
        batch, heads, chunks, state_dim, head_dim, dtype=torch.float32
    )
    causal = torch.tril(torch.ones(tile, tile, dtype=torch.bool))
    for bi in range(batch):
        for hi in range(heads):
            gi = hi // heads_per_group
            for ki in range(chunks):
                xf = x[bi, hi, ki].float()
                bf = b[bi, ki, gi].float()
                cf = c[bi, ki, gi].float()
                d = da[bi, hi, ki]
                cb = cf @ bf.transpose(0, 1)
                decay = torch.exp(d[:, None] - d[None, :])
                expected_y[bi, hi, ki] = torch.where(
                    causal, cb * decay, 0.0
                ) @ xf
                state_decay = torch.exp(d[-1] - d)
                # T=128 keeps state in Cube-native [N,P] layout.
                expected_state[bi, hi, ki] = (
                    bf.transpose(0, 1) @ (xf * state_decay[:, None])
                )

    actual_y, actual_state = torch.ops.mamba_ascend.mamba2_ssd_chunk_mix(
        x.npu(), da.npu(), b.transpose(-1, -2).contiguous().npu(), c.npu()
    )
    actual_y = actual_y.cpu()
    actual_state = actual_state.cpu()
    for name, actual, expected in (
        ("y", actual_y, expected_y),
        ("state", actual_state, expected_state),
    ):
        values = _metrics(actual, expected)
        print(name, values, flush=True)
        assert torch.isfinite(actual).all()
        assert values[0] <= 5e-2
        assert values[1] <= 5e-3
        assert values[2] >= 0.999
