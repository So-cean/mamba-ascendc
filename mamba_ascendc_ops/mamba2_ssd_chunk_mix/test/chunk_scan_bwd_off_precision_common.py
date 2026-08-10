"""Deterministic precision cases for Mamba2SsdChunkScanBwdOff."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import ascend_kernel  # noqa: F401 -- registers torch.ops.mamba_ascend


THRESHOLD = 2**-13

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
        20260808 + case.case_id
    )
    scale = 0.01 if case.mode == "small_signal" else 0.1
    gy = scale + scale * torch.rand(
        batch, heads, chunks, 64, 64, generator=generator
    )
    states = scale + scale * torch.rand(
        batch, heads, chunks, 64, 64, generator=generator
    )
    c_cube = (
        scale
        + scale
        * torch.rand(batch, chunks, groups, 64, 64, generator=generator)
    ).half()
    if case.mode == "zero_decay":
        d_a = torch.zeros(batch, heads, chunks, 64)
    elif case.mode == "strong_decay":
        d_a = -1.0 - torch.rand(
            batch, heads, chunks, 64, generator=generator
        )
    else:
        d_a = -0.01 - 0.2 * torch.rand(
            batch, heads, chunks, 64, generator=generator
        )
    return gy, states, d_a, c_cube


def reference(gy, states, d_a, c_cube):
    heads = gy.shape[1]
    groups = c_cube.shape[2]
    c_head = c_cube.permute(0, 2, 1, 3, 4).repeat_interleave(
        heads // groups, dim=1
    )
    # Match the native mixed-precision contract exactly: the two Cube
    # operands Q/state are rounded to FP16 and accumulated into FP32.
    q = (gy * torch.exp(d_a)[..., None]).half().float()
    state_half = states.half().float()
    c_float = c_head.float()
    d_states = torch.matmul(q.transpose(-1, -2), c_float)
    d_c_head = torch.matmul(q, state_half)
    g_d_a = (d_c_head * c_float).sum(dim=-1)
    return d_states, d_c_head, g_d_a


def npu_call(inputs):
    npu_inputs = tuple(tensor.npu().contiguous() for tensor in inputs)
    actual = torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_off(
        *npu_inputs
    )
    torch.npu.synchronize()
    return tuple(tensor.cpu() for tensor in actual)


def metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    abs_err = (actual - expected).abs()
    rel_err = abs_err / (expected.abs() + 1.0e-7)
    both_zero = not torch.count_nonzero(actual) and not torch.count_nonzero(
        expected
    )
    cosine = (
        1.0
        if both_zero
        else F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    )
    return {
        "MERE": rel_err.mean().item(),
        "MARE": rel_err.max().item(),
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "cosine_sim": cosine,
    }


def evaluate(case: Case):
    inputs = make_inputs(case)
    expected = reference(*inputs)
    actual = npu_call(inputs)
    names = ("d_states_start", "d_c_head", "g_dA_cs_off")
    output_metrics = {
        name: metrics(got, want)
        for name, got, want in zip(names, actual, expected)
    }
    passed = all(
        value["MERE"] < THRESHOLD
        and value["MARE"] < 10 * THRESHOLD
        for value in output_metrics.values()
    )
    return output_metrics, passed
