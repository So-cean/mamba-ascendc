"""Shared 32-case precision evaluation for Mamba2 SSD backward M0."""

from __future__ import annotations

import math

import torch

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


THRESHOLD = 2**-13
GRAD_NAMES = ("dx", "ddt", "dA", "dB", "dC")

# M0 intentionally supports P=N=chunk=64 only.  The matrix varies batch,
# sequence length, head/group sharing and all three public-output grad paths.
SHAPES = (
    ("single_head", (1, 64, 1, 64, 64, 64, 1)),
    ("shared_group_2h", (1, 64, 2, 64, 64, 64, 1)),
    ("one_group_per_head_2h", (1, 64, 2, 64, 64, 64, 2)),
    ("shared_group_4h", (1, 64, 4, 64, 64, 64, 1)),
    ("two_groups_4h", (1, 64, 4, 64, 64, 64, 2)),
    ("one_group_per_head_4h", (1, 64, 4, 64, 64, 64, 4)),
    ("batch2_shared_group", (2, 64, 2, 64, 64, 64, 1)),
    ("two_chunks", (1, 128, 2, 64, 64, 64, 2)),
)

GRAD_MODES = (
    "output_only",
    "output_and_final",
    "final_only",
    "scaled_output_and_final",
)


def iter_case_specs():
    case_id = 0
    for category, shape in SHAPES:
        for grad_mode in GRAD_MODES:
            case_id += 1
            yield case_id, category, shape, grad_mode


def make_inputs(case_id, shape):
    batch, seqlen, nheads, headdim, dstate, _, ngroups = shape
    generator = torch.Generator(device="cpu").manual_seed(20260806 + case_id)

    # A positive, non-cancelling domain makes MERE/MARE meaningful for every
    # derivative instead of allowing random near-zero denominators to dominate.
    x = 0.10 + 0.15 * torch.rand(
        batch, seqlen, nheads, headdim, generator=generator
    )
    dt = 0.002 + 0.004 * torch.rand(
        batch, seqlen, nheads, generator=generator
    )
    A = -(0.005 + 0.005 * torch.rand(nheads, generator=generator))
    B = 0.05 + 0.10 * torch.rand(
        batch, seqlen, ngroups, dstate, generator=generator
    )
    C = 0.05 + 0.10 * torch.rand(
        batch, seqlen, ngroups, dstate, generator=generator
    )
    return x, dt, A, B, C


def make_grad_outputs(case_id, ref_out, ref_final, grad_mode):
    generator = torch.Generator(device="cpu").manual_seed(20260906 + case_id)
    dout = 0.10 + 0.20 * torch.rand(ref_out.shape, generator=generator)
    dfinal = 0.05 + 0.10 * torch.rand(ref_final.shape, generator=generator)
    if grad_mode == "output_only":
        return (ref_out,), (dout,), (dout.npu(),)
    if grad_mode == "final_only":
        return (ref_final,), (dfinal,), (dfinal.npu(),)
    if grad_mode == "scaled_output_and_final":
        dout = dout * 0.125
        dfinal = dfinal * 2.0
    return (
        (ref_out, ref_final),
        (dout, dfinal),
        (dout.npu(), dfinal.npu()),
    )


def compute_metrics(actual, reference):
    actual = actual.detach().cpu().float()
    reference = reference.detach().cpu().float()
    abs_err = (actual - reference).abs()
    rel_err = abs_err / (reference.abs() + 1.0e-7)
    ref_rms = torch.sqrt(torch.mean(reference.square())).clamp_min(1.0e-12)
    nrmse = torch.sqrt(torch.mean(abs_err.square())) / ref_rms
    if torch.count_nonzero(actual).item() == 0 and torch.count_nonzero(reference).item() == 0:
        cosine = 1.0
    else:
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        ).item()
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "MERE": rel_err.mean().item(),
        "MARE": rel_err.max().item(),
        "NRMSE": nrmse.item(),
        "cosine_sim": cosine,
        "nan_inf": int((~torch.isfinite(actual)).sum().item()),
    }


def evaluate_case(spec):
    case_id, category, shape, grad_mode = spec
    values = make_inputs(case_id, shape)
    ref_values = tuple(value.detach().requires_grad_(True) for value in values)
    ref_out, ref_final = ssd_chunk_scan_ref(
        *ref_values, shape[5], return_final_state=True
    )
    ref_outputs, ref_grad_outputs, npu_grad_outputs = make_grad_outputs(
        case_id, ref_out, ref_final, grad_mode
    )
    reference_raw = torch.autograd.grad(
        ref_outputs, ref_values, ref_grad_outputs, allow_unused=True
    )
    reference = tuple(
        torch.zeros_like(value) if grad is None else grad
        for value, grad in zip(ref_values, reference_raw)
    )

    npu_values = tuple(
        value.npu().contiguous().detach().requires_grad_(True)
        for value in values
    )
    out, final_state = ascend_kernel.mamba2_ssd_fwd(
        *npu_values, shape[5], return_final_state=True
    )
    actual_outputs = {
        "output_only": (out,),
        "output_and_final": (out, final_state),
        "final_only": (final_state,),
        "scaled_output_and_final": (out, final_state),
    }[grad_mode]
    actual = torch.autograd.grad(
        actual_outputs, npu_values, npu_grad_outputs, allow_unused=False
    )
    torch.npu.synchronize()

    metrics = {
        name: compute_metrics(got, expected)
        for name, got, expected in zip(GRAD_NAMES, actual, reference)
    }
    passed = all(
        value["MERE"] < THRESHOLD
        and value["MARE"] < 10 * THRESHOLD
        and value["nan_inf"] == 0
        for value in metrics.values()
    )
    return {
        "case_id": case_id,
        "category": category,
        "grad_mode": grad_mode,
        "shape": list(shape),
        "dtype": "float32",
        "metrics": metrics,
        "threshold": THRESHOLD,
        "passed": passed,
    }
