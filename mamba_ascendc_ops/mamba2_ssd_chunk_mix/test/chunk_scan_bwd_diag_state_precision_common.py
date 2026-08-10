"""Deterministic precision cases for the fused diagonal/state backward op."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import ascend_kernel  # noqa: F401 -- registers torch.ops.mamba_ascend


SHAPES = (
    ("single", "one head and one chunk", (1, 1, 1, 1)),
    ("heads", "two heads sharing one group", (1, 2, 1, 1)),
    ("chunks", "two heads and two chunks", (1, 2, 2, 1)),
    ("groups", "four heads over two groups", (1, 4, 2, 2)),
    ("shared", "four heads sharing one group", (1, 4, 4, 1)),
    ("batch", "two batches and two heads", (2, 2, 2, 1)),
    ("batch_group", "two batches and two groups", (2, 4, 1, 2)),
    ("group_per_head", "one group per head", (2, 4, 4, 4)),
    ("scale", "eight heads and eight chunks", (1, 8, 8, 2)),
    ("occupancy", "two batches and eight groups", (2, 8, 4, 8)),
)

MODES = ("nominal", "zero_decay", "strong_decay", "small_signal")


@dataclass(frozen=True)
class Case:
    case_id: int
    category: str
    description: str
    shape: tuple[int, int, int, int]
    mode: str


def all_cases() -> list[Case]:
    cases = []
    case_id = 0
    for category, description, shape in SHAPES:
        for mode in MODES:
            case_id += 1
            cases.append(Case(case_id, category, description, shape, mode))
    return cases


def make_inputs(case: Case):
    batch, heads, chunks, groups = case.shape
    generator = torch.Generator(device="cpu").manual_seed(
        20260820 + case.case_id
    )
    scale = 0.01 if case.mode == "small_signal" else 0.1
    gy = scale + scale * torch.rand(
        batch, heads, chunks, 64, 64, generator=generator
    )
    x = (
        scale
        + scale
        * torch.rand(batch, heads, chunks, 64, 64, generator=generator)
    ).half()
    b_cube = (
        scale
        + scale
        * torch.rand(batch, chunks, groups, 64, 64, generator=generator)
    ).half()
    c_cube = (
        scale
        + scale
        * torch.rand(batch, chunks, groups, 64, 64, generator=generator)
    ).half()
    d_chunk_states = scale + scale * torch.rand(
        batch, heads, chunks, 64, 64, generator=generator
    )
    if case.mode == "zero_decay":
        d_a = torch.zeros(batch, heads, chunks, 64)
    else:
        step_scale = 0.05 if case.mode == "strong_decay" else 0.002
        steps = step_scale + step_scale * torch.rand(
            batch, heads, chunks, 64, generator=generator
        )
        d_a = -steps.cumsum(dim=-1)
    return gy, x, d_a, b_cube, c_cube, d_chunk_states


def _expand_group(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    groups = tensor.shape[2]
    return tensor.permute(0, 2, 1, 3, 4).repeat_interleave(
        heads // groups, dim=1
    )


def reference(gy, x, d_a, b_cube, c_cube, d_chunk_states):
    heads = gy.shape[1]
    b_nt = _expand_group(b_cube, heads).float()
    b_tn = b_nt.transpose(-1, -2).contiguous()
    c_tn = _expand_group(c_cube, heads).float()
    x_float = x.float()
    gy_half = gy.half().float()
    u_half = d_chunk_states.half().float()

    values = d_a.unsqueeze(-1) - d_a.unsqueeze(-2)
    causal = torch.tril(torch.ones(64, 64, dtype=torch.bool))
    decay = torch.where(causal, torch.exp(values), torch.zeros_like(values))

    cb = torch.matmul(c_tn, b_nt)
    w = (cb * decay).half().float()
    d_x_diag = torch.matmul(w.transpose(-1, -2), gy_half)
    d_w = torch.matmul(gy_half, x_float.transpose(-1, -2))
    d_cb = (d_w * decay).half().float()
    d_c = torch.matmul(d_cb, b_tn)
    d_b_diag = torch.matmul(d_cb.transpose(-1, -2), c_tn)
    g_diag_product = d_w * w
    g_diag = g_diag_product.sum(-1) - g_diag_product.sum(-2)

    decay_to_end = torch.exp(d_a[..., -1:] - d_a)
    r = (x_float * decay_to_end.half().float().unsqueeze(-1)).half().float()
    d_r = torch.matmul(b_tn, u_half.transpose(-1, -2))
    d_b_state = torch.matmul(r, u_half)
    d_x_state = d_r * decay_to_end.unsqueeze(-1)
    state_rows = (d_r * r).sum(-1)
    g_state = -state_rows
    g_state[..., -1] += state_rows.sum(-1)

    batch, heads, chunks, _, _ = gy.shape
    groups = b_cube.shape[2]
    heads_per_group = heads // groups
    d_b_group = (d_b_diag + d_b_state).reshape(
        batch, groups, heads_per_group, chunks, 64, 64
    ).sum(dim=2).permute(0, 2, 1, 3, 4).contiguous()
    d_c_group = d_c.reshape(
        batch, groups, heads_per_group, chunks, 64, 64
    ).sum(dim=2).permute(0, 2, 1, 3, 4).contiguous()
    return (
        d_x_diag + d_x_state,
        d_b_group,
        d_c_group,
        g_diag + g_state,
    )


def npu_call(inputs):
    npu_inputs = tuple(tensor.npu().contiguous() for tensor in inputs)
    actual = torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_diag_state(
        *npu_inputs
    )
    torch.npu.synchronize()
    return tuple(tensor.cpu() for tensor in actual)


def metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    abs_err = error.abs()
    rel_err = abs_err / (expected.abs() + 1.0e-6)
    rmse = error.square().mean().sqrt()
    nrmse = (rmse / (expected.square().mean().sqrt() + 1.0e-6)).item()
    both_zero = not torch.count_nonzero(actual) and not torch.count_nonzero(
        expected
    )
    cosine = (
        1.0
        if both_zero
        else F.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item()
    )
    return {
        "MERE": rel_err.mean().item(),
        "MARE": rel_err.max().item(),
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "NRMSE": nrmse,
        "cosine_sim": cosine,
    }


def evaluate(case: Case):
    inputs = make_inputs(case)
    expected = reference(*inputs)
    actual = npu_call(inputs)
    names = ("d_xdt", "d_b_group", "d_c_diag_group", "g_dA_cs")
    output_metrics = {
        name: metrics(got, want)
        for name, got, want in zip(names, actual, expected)
    }
    # MARE is reported but is not a stable gate for the cancellation-heavy
    # cumsum gradient.  NRMSE and cosine measure the complete output tensor.
    passed = all(
        value["NRMSE"] < 2.0e-3 and value["cosine_sim"] > 0.99999
        for value in output_metrics.values()
    )
    return output_metrics, passed
