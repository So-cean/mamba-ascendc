"""Shared end-to-end precision matrix for the M1 backward."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


SHAPES = (
    ("single", (1, 64, 1, 1)),
    ("heads", (1, 64, 2, 1)),
    ("groups", (1, 128, 2, 2)),
    ("shared", (1, 128, 4, 1)),
    ("multi_group", (1, 128, 4, 2)),
    ("batch", (2, 64, 4, 4)),
    ("chunks", (1, 256, 8, 2)),
    ("occupancy", (2, 128, 8, 8)),
)

MODES = (
    "output_only",
    "output_final",
    "final_only",
    "full_channel_D",
    "full_head_D",
)


@dataclass(frozen=True)
class Case:
    case_id: int
    category: str
    shape: tuple[int, int, int, int]
    mode: str


def all_cases():
    result = []
    case_id = 0
    for category, shape in SHAPES:
        for mode in MODES:
            case_id += 1
            result.append(Case(case_id, category, shape, mode))
    # Exercise the production hybrid Diag/Off route.  The cartesian cases
    # above top out at 32 logical head-chunk tasks, while auto routing switches
    # at 1024.  One focused full-feature case covers the optimized Prepare and
    # PrepareDcb kernels without multiplying the CPU reference cost by every
    # output mode.
    case_id += 1
    result.append(
        Case(case_id, "hybrid_threshold", (1, 1024, 64, 16), "full_head_D")
    )
    return result


def make_case(case: Case):
    batch, seqlen, heads, groups = case.shape
    generator = torch.Generator(device="cpu").manual_seed(
        20260810 + case.case_id
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
    if case.mode.startswith("full"):
        values.update(
            D=(
                positive(heads, scale=0.03)
                if case.mode == "full_head_D"
                else positive(heads, 64, scale=0.03)
            ),
            z=positive(batch, seqlen, heads, 64, scale=0.2),
            dt_bias=0.02 * torch.randn(heads, generator=generator),
            initial_states=positive(batch, heads, 64, 64, scale=0.02),
            dt_softplus=True,
            dt_limit=(0.02, 1.0),
        )
    return values


def _leaf_values(values, device):
    names = ["x", "dt", "A", "B", "C"]
    for name in ("D", "z", "dt_bias", "initial_states"):
        if values[name] is not None:
            names.append(name)
    leaves = {
        name: values[name].to(device).contiguous().detach().requires_grad_(True)
        for name in names
    }
    return names, leaves


def _forward(fn, leaves, values):
    return fn(
        leaves["x"],
        leaves["dt"],
        leaves["A"],
        leaves["B"],
        leaves["C"],
        64,
        D=leaves.get("D"),
        z=leaves.get("z"),
        dt_bias=leaves.get("dt_bias"),
        dt_softplus=values["dt_softplus"],
        dt_limit=values["dt_limit"],
        initial_states=leaves.get("initial_states"),
        return_final_state=True,
    )


def _selected_outputs(case, out, final_state, dout, dfinal):
    if case.mode == "output_only":
        return (out,), (dout,)
    if case.mode == "final_only":
        return (final_state,), (dfinal,)
    return (out, final_state), (dout, dfinal)


def _metrics(actual, expected):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    error = actual - expected
    both_zero = (
        torch.count_nonzero(actual).item() == 0
        and torch.count_nonzero(expected).item() == 0
    )
    return {
        "max_abs_err": error.abs().max().item(),
        "nrmse": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
        ).item(),
        "cosine_sim": 1.0
        if both_zero
        else F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
    }


def evaluate(case: Case):
    os.environ["MAMBA_ASCENDC_CHUNK_MIX"] = "1"
    values = make_case(case)
    cpu_names, cpu_leaves = _leaf_values(values, "cpu")
    ref_out, ref_final = _forward(ssd_chunk_scan_ref, cpu_leaves, values)
    generator = torch.Generator(device="cpu").manual_seed(20260811 + case.case_id)
    dout = 0.1 + 0.1 * torch.rand(ref_out.shape, generator=generator)
    dfinal = 0.1 + 0.1 * torch.rand(ref_final.shape, generator=generator)
    ref_outputs, ref_upstream = _selected_outputs(
        case, ref_out, ref_final, dout, dfinal
    )
    expected_raw = torch.autograd.grad(
        ref_outputs,
        tuple(cpu_leaves[name] for name in cpu_names),
        ref_upstream,
        allow_unused=True,
    )
    expected = tuple(
        torch.zeros_like(cpu_leaves[name]) if grad is None else grad
        for name, grad in zip(cpu_names, expected_raw)
    )

    npu_names, npu_leaves = _leaf_values(values, "npu")
    out, final_state = _forward(ascend_kernel.mamba2_ssd_fwd, npu_leaves, values)
    outputs, upstream = _selected_outputs(
        case, out, final_state, dout.npu(), dfinal.npu()
    )
    actual = torch.autograd.grad(
        outputs,
        tuple(npu_leaves[name] for name in npu_names),
        upstream,
    )
    torch.npu.synchronize()
    output_metrics = {
        name: _metrics(got, want)
        for name, got, want in zip(npu_names, actual, expected)
    }
    passed = all(
        metric["nrmse"] <= 8.0e-3 and metric["cosine_sim"] >= 0.999
        for metric in output_metrics.values()
    )
    return output_metrics, passed
