#!/usr/bin/env python3
"""Basic functional and mixed-precision gate for Mamba2SsdChunkMix key 2."""

from __future__ import annotations

import ctypes
import os

import pytest
import torch
import torch_npu  # noqa: F401


def _load_custom_operator() -> None:
    library_path = os.environ.get("MAMBA_CHUNK_MIX_OP_API_LIB")
    if library_path:
        library = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
        assert getattr(library, "aclnnMamba2SsdChunkMix")
    import ascend_kernel  # noqa: F401,E402


_load_custom_operator()


def _make_inputs(batch: int, heads: int, chunks: int, groups: int,
                 state_dim: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (0.1 * torch.randn(
        (batch, heads, chunks, 64, 64), generator=generator
    )).half()
    # A*dt is negative in Mamba, so its cumulative sum is monotone decreasing.
    d_a = -(0.005 + 0.045 * torch.rand(
        (batch, heads, chunks, 64), generator=generator
    ))
    d_a_cumsum = torch.cumsum(d_a, dim=-1).float()
    b = (0.1 * torch.randn(
        (batch, chunks, groups, 64, state_dim), generator=generator
    )).half()
    c = (0.1 * torch.randn(
        (batch, chunks, groups, 64, state_dim), generator=generator
    )).half()
    return x, d_a_cumsum, b, c


def _reference(x: torch.Tensor, d_a: torch.Tensor,
               b_tensor: torch.Tensor, c_tensor: torch.Tensor):
    batch, heads, chunks, _, _ = x.shape
    groups = b_tensor.shape[2]
    heads_per_group = heads // groups
    y = torch.empty_like(x, dtype=torch.float32)
    state_dim = b_tensor.shape[-1]
    states = torch.empty(
        (batch, heads, chunks, 64, state_dim), dtype=torch.float32
    )
    causal = torch.tril(torch.ones((64, 64), dtype=torch.bool))
    for batch_idx in range(batch):
        for head_idx in range(heads):
            group_idx = head_idx // heads_per_group
            for chunk_idx in range(chunks):
                x_float = x[batch_idx, head_idx, chunk_idx].float()
                d_a_row = d_a[batch_idx, head_idx, chunk_idx]
                b_float = b_tensor[batch_idx, chunk_idx, group_idx].float()
                c_float = c_tensor[batch_idx, chunk_idx, group_idx].float()
                cb = c_float @ b_float.transpose(0, 1)
                decay = torch.exp(d_a_row[:, None] - d_a_row[None, :])
                weights = torch.where(causal, cb * decay, 0.0)
                y[batch_idx, head_idx, chunk_idx] = weights @ x_float
                state_decay = torch.exp(d_a_row[-1] - d_a_row)
                weighted_x = x_float * state_decay[:, None]
                states[batch_idx, head_idx, chunk_idx] = (
                    weighted_x.transpose(0, 1) @ b_float
                )
    return y, states


def _metrics(actual: torch.Tensor, expected: torch.Tensor):
    diff = (actual.float() - expected.float()).reshape(-1)
    expected_flat = expected.float().reshape(-1)
    max_abs = diff.abs().max().item()
    nrmse = (
        torch.sqrt(torch.mean(diff * diff)) /
        torch.sqrt(torch.mean(expected_flat * expected_flat)).clamp_min(1.0e-12)
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1), expected_flat.reshape(1, -1)
    ).item()
    return max_abs, nrmse, cosine


@pytest.mark.parametrize(
    "batch,heads,chunks,groups,state_dim",
    [
        # N=64 regression plus N=128 at one, one-wave and multi-wave loads.
        (1, 1, 1, 1, 64),
        (2, 8, 2, 2, 64),
        (1, 1, 1, 1, 128),
        (1, 10, 2, 2, 128),
        (2, 8, 4, 2, 128),
        # 32 (B,G,K) tasks trigger grouped-CB reuse across four heads/group.
        (2, 16, 4, 4, 128),
    ],
)
def test_mamba2_ssd_chunk_mix_basic(batch, heads, chunks, groups, state_dim):
    cpu_inputs = _make_inputs(
        batch, heads, chunks, groups, state_dim,
        seed=(20260802 + batch * 1000 + heads * 100 + chunks * 10 + groups
              + state_dim),
    )
    expected_y, expected_state = _reference(*cpu_inputs)
    x, da, b, c = cpu_inputs
    npu_inputs = (
        x.npu(), da.npu(), b.transpose(-1, -2).contiguous().npu(), c.npu()
    )
    actual_y, actual_state = torch.ops.mamba_ascend.mamba2_ssd_chunk_mix(*npu_inputs)
    actual_y = actual_y.cpu()
    actual_state = actual_state.cpu()
    if actual_state.shape[-2:] == (state_dim, 64):
        expected_state = expected_state.transpose(-1, -2).contiguous()

    for name, actual, expected in (
        ("y_diag", actual_y, expected_y),
        ("chunk_states", actual_state, expected_state),
    ):
        assert torch.isfinite(actual).all(), f"{name} contains NaN/Inf"
        max_abs, nrmse, cosine = _metrics(actual, expected)
        print(
            f"case={(batch, heads, chunks, groups, state_dim)} output={name} "
            f"max_abs={max_abs:.6e} nrmse={nrmse:.6e} cosine={cosine:.9f}"
        )
        assert max_abs <= 5.0e-2
        assert nrmse <= 5.0e-3
        assert cosine >= 0.999


if __name__ == "__main__":
    # A device exception poisons the current NPU context, so stop immediately
    # and let the next test process start from a clean device context.
    raise SystemExit(pytest.main([__file__, "-v", "-s", "-x"]))
