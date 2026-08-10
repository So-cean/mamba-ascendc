"""Precision gate for the heavy FP16-state reverse recurrence path."""

from __future__ import annotations

import pytest
import torch

import ascend_kernel  # noqa: F401 -- register torch.ops.mamba_ascend


def _reference(states_start, d_states_start, d_a_cumsum, dfinal_state):
    states_float = states_start.float()
    d_states_float = d_states_start.float()
    gstate = (
        torch.zeros_like(d_states_float[:, :, 0])
        if dfinal_state is None
        else dfinal_state
    )
    d_chunk_states = torch.empty_like(states_start)
    d_a_chunk_last = torch.empty(states_start.shape[:3], dtype=torch.float32)
    for chunk in range(states_start.shape[2] - 1, -1, -1):
        alpha = torch.exp(d_a_cumsum[:, :, chunk, -1])
        d_chunk_states[:, :, chunk] = gstate.half()
        d_a_chunk_last[:, :, chunk] = (
            gstate * states_float[:, :, chunk]
        ).sum(dim=(-2, -1)) * alpha
        gstate = d_states_float[:, :, chunk] + alpha[..., None, None] * gstate
    return d_chunk_states, gstate, d_a_chunk_last


@pytest.mark.parametrize(
    "shape,use_dfinal",
    [
        ((1, 1, 1, 64, 64, 64), False),
        ((1, 4, 4, 64, 64, 64), True),
        ((2, 8, 8, 64, 64, 64), True),
        ((2, 16, 16, 64, 64, 64), False),
    ],
)
@pytest.mark.parametrize("d_state_dtype", (torch.float32, torch.float16))
def test_mamba2_ssd_state_passing_bwd_half(
    shape, use_dfinal, d_state_dtype
):
    batch, heads, chunks, headdim, dstate, chunk_size = shape
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    states_start = (
        0.1
        * torch.randn(
            batch, heads, chunks, headdim, dstate, generator=generator
        )
    ).half()
    d_states_start = 0.1 * torch.randn(
        batch, heads, chunks, headdim, dstate, generator=generator
    ).to(d_state_dtype)
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
    actual = torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd_half(
        states_start.npu(),
        d_states_start.npu(),
        d_a_cumsum.npu(),
        None if dfinal_state is None else dfinal_state.npu(),
    )
    repeated = torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd_half(
        states_start.npu(),
        d_states_start.npu(),
        d_a_cumsum.npu(),
        None if dfinal_state is None else dfinal_state.npu(),
    )
    torch.npu.synchronize()

    assert actual[0].dtype == torch.float16
    assert actual[1].dtype == torch.float32
    assert actual[2].dtype == torch.float32

    for name, lhs, rhs in zip(
        ("d_chunk_states", "dinitial_state", "dA_chunk_last"),
        actual,
        repeated,
    ):
        assert torch.equal(lhs.cpu(), rhs.cpu()), f"{name} is not deterministic"

    for name, got, want in zip(
        ("d_chunk_states", "dinitial_state", "dA_chunk_last"),
        actual,
        expected,
    ):
        got_cpu = got.cpu()
        abs_err = (got_cpu.float() - want.float()).abs()
        rel_err = abs_err / (want.float().abs() + 1.0e-7)
        # The stored recurrence is FP16.  One representable FP16 step near
        # unit scale is 4.8828125e-4; allow exactly that storage rounding plus
        # the existing ScalarExp approximation, while keeping FP32 outputs at
        # the tighter native-state threshold.
        tolerance = 5.0e-4 if got_cpu.dtype == torch.float16 else 3.0e-4
        torch.testing.assert_close(
            got_cpu,
            want,
            rtol=tolerance,
            atol=tolerance,
            msg=(
                f"{name}: finite={bool(torch.isfinite(got_cpu).all())}, "
                f"max_abs={abs_err.max().item():.8g}, "
                f"mean_abs={abs_err.mean().item():.8g}, "
                f"max_rel={rel_err.max().item():.8g}"
            ),
        )


def _head_to_grouped_state(value: torch.Tensor, groups: int) -> torch.Tensor:
    batch, heads, chunks, headdim, dstate = value.shape
    heads_per_group = heads // groups
    return (
        value.reshape(
            batch, groups, heads_per_group, chunks, headdim, dstate
        )
        .permute(0, 3, 1, 5, 2, 4)
        .contiguous()
    )


def test_mamba2_ssd_state_passing_bwd_half_grouped_output():
    """Keep the reverse recurrence producer-native through its GM output."""
    batch, heads, chunks, groups = 2, 32, 4, 8
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    shape = (batch, heads, chunks, 64, 64)
    states_head = (0.1 * torch.randn(shape, generator=generator)).half()
    d_states_head = (0.1 * torch.randn(shape, generator=generator)).half()
    d_a = -0.01 - 0.2 * torch.rand(
        batch, heads, chunks, 64, generator=generator
    )
    dfinal = 0.1 * torch.randn(
        batch, heads, 64, 64, generator=generator
    )
    expected = _reference(states_head, d_states_head, d_a, dfinal)
    expected_grouped = _head_to_grouped_state(expected[0], groups)
    states_grouped = _head_to_grouped_state(states_head, groups)
    d_states_grouped = _head_to_grouped_state(d_states_head, groups)

    args = (
        states_grouped.npu(),
        d_states_grouped.npu(),
        d_a.npu(),
        dfinal.npu(),
    )
    first = torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd_half(*args)
    second = torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd_half(*args)
    torch.npu.synchronize()

    assert first[0].shape == expected_grouped.shape
    torch.testing.assert_close(
        first[0].cpu(), expected_grouped, rtol=5.0e-4, atol=5.0e-4
    )
    torch.testing.assert_close(
        first[1].cpu(), expected[1], rtol=3.0e-4, atol=3.0e-4
    )
    torch.testing.assert_close(
        first[2].cpu(), expected[2], rtol=3.0e-4, atol=3.0e-4
    )
    for actual, repeated in zip(first, second):
        assert torch.equal(actual.cpu(), repeated.cpu())
