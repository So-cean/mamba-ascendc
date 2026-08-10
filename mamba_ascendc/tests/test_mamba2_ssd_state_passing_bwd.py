"""Correctness gate for the native reverse chunk-state recurrence."""

from __future__ import annotations

import pytest
import torch

import ascend_kernel  # noqa: F401  -- loads the torch custom-op library


def _reference(states_start, d_states_start, d_a_cumsum, dfinal_state):
    gstate = (
        torch.zeros_like(states_start[:, :, 0])
        if dfinal_state is None
        else dfinal_state
    )
    d_chunk_states = torch.empty_like(states_start)
    d_a_chunk_last = torch.empty(states_start.shape[:3], dtype=torch.float32)
    for chunk in range(states_start.shape[2] - 1, -1, -1):
        alpha = torch.exp(d_a_cumsum[:, :, chunk, -1])
        d_chunk_states[:, :, chunk] = gstate
        d_a_chunk_last[:, :, chunk] = (
            gstate * states_start[:, :, chunk]
        ).sum(dim=(-2, -1)) * alpha
        gstate = (
            d_states_start[:, :, chunk]
            + alpha[..., None, None] * gstate
        )
    return d_chunk_states, gstate, d_a_chunk_last


@pytest.mark.parametrize(
    "shape,use_dfinal",
    [
        ((1, 1, 1, 64, 64, 64), False),
        ((1, 1, 4, 64, 64, 64), True),
        ((1, 2, 2, 64, 128, 64), True),
        ((2, 2, 2, 64, 128, 128), False),
    ],
)
def test_mamba2_ssd_state_passing_bwd(shape, use_dfinal):
    batch, heads, chunks, headdim, dstate, chunk_size = shape
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    states_start = 0.1 * torch.randn(
        batch, heads, chunks, headdim, dstate, generator=generator
    )
    d_states_start = 0.1 * torch.randn(
        batch, heads, chunks, headdim, dstate, generator=generator
    )
    d_a_cumsum = -0.01 - 0.2 * torch.rand(
        batch, heads, chunks, chunk_size, generator=generator
    )
    dfinal_state = (
        0.1
        * torch.randn(batch, heads, headdim, dstate, generator=generator)
        if use_dfinal
        else None
    )
    expected = _reference(
        states_start, d_states_start, d_a_cumsum, dfinal_state
    )
    actual = torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd(
        states_start.npu(),
        d_states_start.npu(),
        d_a_cumsum.npu(),
        None if dfinal_state is None else dfinal_state.npu(),
    )
    torch.npu.synchronize()

    for name, got, want in zip(
        ("d_chunk_states", "dinitial_state", "dA_chunk_last"),
        actual,
        expected,
    ):
        torch.testing.assert_close(
            got.cpu(), want, rtol=2.0e-4, atol=2.0e-4, msg=name
        )
