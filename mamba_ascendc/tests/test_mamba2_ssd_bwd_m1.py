"""End-to-end gradient gates for the M1 chunk backward composition."""

from __future__ import annotations

import pytest
import torch

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


def _make_case(feature: str):
    shape = {
        "basic": (1, 64, 1, 1),
        "shared_final": (1, 128, 4, 2),
        "full": (1, 64, 2, 1),
        "final_only": (1, 128, 2, 1),
    }[feature]
    batch, seqlen, heads, groups = shape
    generator = torch.Generator(device="cpu").manual_seed(
        20260808 + sum(shape)
    )

    def positive(*dims, scale=0.1):
        return scale + scale * torch.rand(*dims, generator=generator)

    values = {
        "x": positive(batch, seqlen, heads, 64),
        "dt": positive(batch, seqlen, heads, scale=0.03),
        "A": -(0.1 + 0.2 * torch.rand(heads, generator=generator)),
        "B": positive(batch, seqlen, groups, 64, scale=0.05),
        "C": positive(batch, seqlen, groups, 64, scale=0.05),
        "D": None,
        "z": None,
        "dt_bias": None,
        "initial_states": None,
        "dt_softplus": False,
        "dt_limit": (0.0, float("inf")),
    }
    if feature == "full":
        values.update(
            D=positive(heads, 64, scale=0.03),
            z=positive(batch, seqlen, heads, 64, scale=0.2),
            dt_bias=0.02 * torch.randn(heads, generator=generator),
            initial_states=positive(batch, heads, 64, 64, scale=0.02),
            dt_softplus=True,
            # softplus(dt+bias) is around 0.7 here, so this finite clamp
            # exercises the active (non-zero-gradient) interval.
            dt_limit=(0.02, 1.0),
        )
    return values


def _leaf_values(case, device):
    names = ["x", "dt", "A", "B", "C"]
    if case["D"] is not None:
        names.append("D")
    if case["z"] is not None:
        names.append("z")
    if case["dt_bias"] is not None:
        names.append("dt_bias")
    if case["initial_states"] is not None:
        names.append("initial_states")
    values = {
        name: case[name].to(device).contiguous().detach().requires_grad_(True)
        for name in names
    }
    return names, values


def _forward(fn, values, case):
    return fn(
        values["x"],
        values["dt"],
        values["A"],
        values["B"],
        values["C"],
        64,
        D=values.get("D"),
        z=values.get("z"),
        dt_bias=values.get("dt_bias"),
        dt_softplus=case["dt_softplus"],
        dt_limit=case["dt_limit"],
        initial_states=values.get("initial_states"),
        return_final_state=True,
    )


def _metric(actual, expected):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    error = actual - expected
    norm = torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
    both_zero = (
        torch.count_nonzero(actual).item() == 0
        and torch.count_nonzero(expected).item() == 0
    )
    return {
        "max_abs": error.abs().max().item(),
        "nrmse": (torch.linalg.vector_norm(error) / norm).item(),
        "cosine": 1.0
        if both_zero
        else torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
    }


@pytest.mark.parametrize(
    "feature",
    ("basic", "shared_final", "full", "final_only"),
)
def test_mamba2_ssd_bwd_m1(monkeypatch, feature):
    monkeypatch.setenv("MAMBA_ASCENDC_CHUNK_MIX", "1")
    case = _make_case(feature)
    cpu_names, cpu_values = _leaf_values(case, "cpu")
    ref_out, ref_final = _forward(ssd_chunk_scan_ref, cpu_values, case)

    generator = torch.Generator(device="cpu").manual_seed(20260809)
    dout = 0.1 + 0.1 * torch.rand(ref_out.shape, generator=generator)
    use_output = feature != "final_only"
    use_final = feature in ("shared_final", "full", "final_only")
    dfinal = (
        0.1 + 0.1 * torch.rand(ref_final.shape, generator=generator)
        if use_final
        else None
    )
    ref_outputs = (ref_out, ref_final) if use_output and use_final else (
        (ref_out,) if use_output else (ref_final,)
    )
    ref_upstream = (dout, dfinal) if use_output and use_final else (
        (dout,) if use_output else (dfinal,)
    )
    expected_raw = torch.autograd.grad(
        ref_outputs,
        tuple(cpu_values[name] for name in cpu_names),
        ref_upstream,
        allow_unused=True,
    )
    expected = tuple(
        torch.zeros_like(cpu_values[name]) if grad is None else grad
        for name, grad in zip(cpu_names, expected_raw)
    )

    npu_names, npu_values = _leaf_values(case, "npu")
    assert npu_names == cpu_names
    out, final_state = _forward(ascend_kernel.mamba2_ssd_fwd, npu_values, case)
    outputs = (out, final_state) if use_output and use_final else (
        (out,) if use_output else (final_state,)
    )
    upstream = (dout.npu(), dfinal.npu()) if use_output and use_final else (
        (dout.npu(),) if use_output else (dfinal.npu(),)
    )
    actual = torch.autograd.grad(
        outputs,
        tuple(npu_values[name] for name in npu_names),
        upstream,
    )
    torch.npu.synchronize()

    for name, got, want in zip(npu_names, actual, expected):
        metric = _metric(got, want)
        assert torch.isfinite(got).all(), f"{name}: non-finite"
        assert metric["nrmse"] <= 8.0e-3, f"{name}: {metric}"
        assert metric["cosine"] >= 0.999, f"{name}: {metric}"
