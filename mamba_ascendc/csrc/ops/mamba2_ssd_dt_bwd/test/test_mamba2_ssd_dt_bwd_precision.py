"""40-case FP32 precision evaluation for Mamba2SsdDtBwd."""

from __future__ import annotations

import pytest
import torch

import ascend_kernel  # noqa: F401

from dt_bwd_precision_common import THRESHOLD, all_cases, evaluate, make_inputs, npu_call


@pytest.mark.parametrize("case", all_cases(), ids=lambda case: f"{case.case_id:02d}-{case.feature}")
def test_precision(case):
    output_metrics, passed = evaluate(case)
    summary = ", ".join(
        f"{name}:MERE={values['MERE']:.3e},MARE={values['MARE']:.3e}"
        for name, values in output_metrics.items()
    )
    assert passed, (
        f"FP32 threshold MERE<{THRESHOLD:.3e}, "
        f"MARE<{10 * THRESHOLD:.3e}; {summary}"
    )


def test_reduction_is_bitwise_deterministic():
    case = all_cases()[27]
    inputs = make_inputs(case)
    outputs = [npu_call(case, inputs)[2:] for _ in range(5)]
    for repeat in outputs[1:]:
        assert torch.equal(repeat[0], outputs[0][0])
        assert torch.equal(repeat[1], outputs[0][1])


def test_fp64_directional_derivative():
    case = all_cases()[10]
    x, d_xdt, g_cs, dt, a, bias = make_inputs(case)
    native = npu_call(case, (x, d_xdt, g_cs, dt, a, bias))
    generator = torch.Generator(device="cpu").manual_seed(20260899)
    directions = (
        0.1 * torch.randn(x.shape, generator=generator),
        0.1 * torch.randn(dt.shape, generator=generator),
        0.1 * torch.randn(a.shape, generator=generator),
        0.1 * torch.randn(bias.shape, generator=generator),
    )
    analytic = sum(
        (grad.double() * direction.double()).sum()
        for grad, direction in zip(native, directions)
    )

    x64, dt64, a64, bias64 = (
        value.double() for value in (x, dt, a, bias)
    )
    d_xdt64, g_cs64 = d_xdt.double(), g_cs.double()
    batch, heads, chunks, chunk_size, _ = d_xdt.shape

    def loss_at(sign, epsilon):
        x_i = x64 + sign * epsilon * directions[0].double()
        dt_i = dt64 + sign * epsilon * directions[1].double()
        a_i = a64 + sign * epsilon * directions[2].double()
        bias_i = bias64 + sign * epsilon * directions[3].double()
        q = torch.nn.functional.softplus(dt_i + bias_i)
        q_h = q.permute(0, 2, 1).reshape(batch, heads, chunks, chunk_size)
        x_h = x_i.permute(0, 2, 1, 3).reshape_as(d_xdt64)
        xdt = x_h * q_h[..., None]
        cs = torch.cumsum(a_i[None, :, None, None] * q_h, dim=-1)
        return (xdt * d_xdt64).sum() + (cs * g_cs64).sum()

    epsilon = 1.0e-5
    finite_difference = (loss_at(1.0, epsilon) - loss_at(-1.0, epsilon)) / (
        2.0 * epsilon
    )
    relative_error = (
        (analytic - finite_difference).abs()
        / (finite_difference.abs() + 1.0e-12)
    ).item()
    assert relative_error < 2.0e-3, (
        f"directional derivative relative error={relative_error:.3e}; "
        f"native={analytic.item():.8f}, fp64_fd={finite_difference.item():.8f}"
    )
