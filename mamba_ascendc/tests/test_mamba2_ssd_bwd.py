"""Basic correctness gate for the native Mamba-2 SSD backward M0 kernel."""

from __future__ import annotations

import pytest
import torch

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


def _make_inputs(shape, seed=20260806):
    batch, seqlen, nheads, headdim, dstate, ngroups = shape
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*dims):
        return torch.randn(*dims, generator=generator, dtype=torch.float32)

    return (
        randn(batch, seqlen, nheads, headdim),
        0.005
        + 0.01
        * torch.rand(batch, seqlen, nheads, generator=generator),
        -(0.1 + 0.2 * torch.rand(nheads, generator=generator)),
        0.1 * randn(batch, seqlen, ngroups, dstate),
        0.1 * randn(batch, seqlen, ngroups, dstate),
    )


def _metrics(actual, expected):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    error = actual - expected
    both_zero = (
        torch.count_nonzero(actual).item() == 0
        and torch.count_nonzero(expected).item() == 0
    )
    return {
        "max_abs": error.abs().max().item(),
        "nrmse": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
        ).item(),
        "cosine": 1.0
        if both_zero
        else torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
    }


@pytest.mark.parametrize(
    "shape,use_final_grad",
    [
        ((1, 64, 1, 64, 64, 1), False),
        ((1, 64, 1, 64, 64, 1), True),
        ((1, 128, 2, 64, 64, 1), False),
        ((1, 64, 4, 64, 64, 2), True),
    ],
)
def test_mamba2_ssd_bwd_m0(shape, use_final_grad):
    cpu_values = _make_inputs(shape)
    ref_values = tuple(value.detach().requires_grad_(True) for value in cpu_values)
    x, dt, A, B, C = ref_values
    ref_out, ref_final = ssd_chunk_scan_ref(
        x, dt, A, B, C, 64, return_final_state=True
    )
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    dout = torch.randn(ref_out.shape, generator=generator)
    dfinal = (
        torch.randn(ref_final.shape, generator=generator)
        if use_final_grad
        else None
    )
    outputs = (ref_out, ref_final) if use_final_grad else (ref_out,)
    grad_outputs = (dout, dfinal) if use_final_grad else (dout,)
    reference = torch.autograd.grad(
        outputs, ref_values, grad_outputs, allow_unused=False
    )

    npu_values = tuple(value.npu().contiguous() for value in cpu_values)
    npu_out, npu_final = ascend_kernel.mamba2_ssd_fwd(
        *npu_values, 64, return_final_state=True
    )
    torch.npu.synchronize()
    result = torch.ops.mamba_ascend.mamba2_ssd_bwd(
        *npu_values,
        dout.npu(),
        npu_final,
        None,
        None,
        None,
        None,
        None if dfinal is None else dfinal.npu(),
        False,
        0.0,
        torch.finfo(torch.float32).max,
    )
    dx, ddt, dA_partial, dB, dC, dD, dz, dinitial = result
    torch.npu.synchronize()
    actual = (dx, ddt, dA_partial.sum(dim=0), dB, dC)

    for name, got, expected in zip(
        ("dx", "ddt", "dA", "dB", "dC"), actual, reference
    ):
        metric = _metrics(got, expected)
        assert torch.isfinite(got).all(), f"{name}: non-finite values"
        assert metric["nrmse"] <= 1.0e-2, f"{name}: {metric}"
        assert metric["cosine"] >= 0.999, f"{name}: {metric}"

    assert dD.numel() == 0
    assert dz.numel() == 0
    assert dinitial.numel() == 0
    assert npu_out.shape == x.shape


@pytest.mark.parametrize("use_final_grad", [False, True])
def test_mamba2_ssd_bwd_public_autograd(use_final_grad):
    shape = (1, 64, 1, 64, 64, 1)
    cpu_values = _make_inputs(shape, seed=20260808)
    ref_values = tuple(value.detach().requires_grad_(True) for value in cpu_values)
    ref_out, ref_final = ssd_chunk_scan_ref(
        *ref_values, 64, return_final_state=True
    )
    generator = torch.Generator(device="cpu").manual_seed(20260809)
    dout = torch.randn(ref_out.shape, generator=generator)
    dfinal = (
        torch.randn(ref_final.shape, generator=generator)
        if use_final_grad
        else None
    )
    reference = torch.autograd.grad(
        (ref_out, ref_final) if use_final_grad else (ref_out,),
        ref_values,
        (dout, dfinal) if use_final_grad else (dout,),
    )

    npu_values = tuple(
        value.npu().contiguous().detach().requires_grad_(True)
        for value in cpu_values
    )
    out, final_state = ascend_kernel.mamba2_ssd_fwd(
        *npu_values, 64, return_final_state=True
    )
    actual = torch.autograd.grad(
        (out, final_state) if use_final_grad else (out,),
        npu_values,
        (dout.npu(), dfinal.npu()) if use_final_grad else (dout.npu(),),
    )
    torch.npu.synchronize()
    for name, got, expected in zip(
        ("dx", "ddt", "dA", "dB", "dC"), actual, reference
    ):
        metric = _metrics(got, expected)
        assert metric["nrmse"] <= 1.0e-2, f"{name}: {metric}"
        assert metric["cosine"] >= 0.999, f"{name}: {metric}"


def test_mamba2_ssd_bwd_public_autograd_final_only():
    shape = (1, 64, 1, 64, 64, 1)
    cpu_values = _make_inputs(shape, seed=20260810)
    ref_values = tuple(value.detach().requires_grad_(True) for value in cpu_values)
    _, ref_final = ssd_chunk_scan_ref(
        *ref_values, 64, return_final_state=True
    )
    generator = torch.Generator(device="cpu").manual_seed(20260811)
    dfinal = torch.randn(ref_final.shape, generator=generator)
    reference_raw = torch.autograd.grad(
        ref_final, ref_values, dfinal, allow_unused=True
    )
    reference = tuple(
        torch.zeros_like(value) if grad is None else grad
        for value, grad in zip(ref_values, reference_raw)
    )

    npu_values = tuple(
        value.npu().contiguous().detach().requires_grad_(True)
        for value in cpu_values
    )
    _, final_state = ascend_kernel.mamba2_ssd_fwd(
        *npu_values, 64, return_final_state=True
    )
    actual = torch.autograd.grad(final_state, npu_values, dfinal.npu())
    torch.npu.synchronize()
    for name, got, expected in zip(
        ("dx", "ddt", "dA", "dB", "dC"), actual, reference
    ):
        metric = _metrics(got, expected)
        assert metric["nrmse"] <= 1.0e-2, f"{name}: {metric}"
        assert metric["cosine"] >= 0.999, f"{name}: {metric}"
