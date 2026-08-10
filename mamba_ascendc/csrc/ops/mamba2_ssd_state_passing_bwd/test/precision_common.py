"""Shared cases and metrics for state-passing backward precision tests."""

from __future__ import annotations

import torch

import ascend_kernel  # noqa: F401  -- loads the custom-op library


THRESHOLD = 2**-13

TEST_SHAPES = [
    ("single", "one stream one chunk T64/N64", (1, 1, 1, 64, 64, 64)),
    ("single", "one stream four chunks T64/N64", (1, 1, 4, 64, 64, 64)),
    ("heads", "two heads two chunks T64/N64", (1, 2, 2, 64, 64, 64)),
    ("batch", "two batches two heads T64/N64", (2, 2, 2, 64, 64, 64)),
    ("n128", "N128 with T64", (1, 2, 2, 64, 128, 64)),
    ("t128", "N128 with T128", (1, 2, 2, 64, 128, 128)),
    ("small", "minimum K with four heads", (1, 4, 1, 64, 64, 64)),
    ("large", "eight chunks four heads N64", (1, 4, 8, 64, 64, 64)),
    ("large", "four chunks four heads N128", (1, 4, 4, 64, 128, 128)),
    ("large", "two batches eight heads N128", (2, 8, 2, 64, 128, 128)),
]

FINAL_MODES = ("dfinal_none", "dfinal_zero", "dfinal_random")


def make_inputs(shape, mode, seed):
    batch, heads, chunks, headdim, dstate, chunk_size = shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Positive values avoid ill-conditioned relative error caused by a dot
    # product whose exact reference is arbitrarily close to zero.
    states_start = 0.02 + 0.08 * torch.rand(
        batch, heads, chunks, headdim, dstate, generator=generator
    )
    d_states_start = 0.02 + 0.08 * torch.rand(
        batch, heads, chunks, headdim, dstate, generator=generator
    )
    d_a_cumsum = -0.01 - 0.2 * torch.rand(
        batch, heads, chunks, chunk_size, generator=generator
    )
    if mode == "dfinal_none":
        dfinal_state = None
    elif mode == "dfinal_zero":
        dfinal_state = torch.zeros(batch, heads, headdim, dstate)
    else:
        dfinal_state = 0.02 + 0.08 * torch.rand(
            batch, heads, headdim, dstate, generator=generator
        )
    return states_start, d_states_start, d_a_cumsum, dfinal_state


def reference(states_start, d_states_start, d_a_cumsum, dfinal_state):
    gstate = (
        torch.zeros_like(states_start[:, :, 0])
        if dfinal_state is None
        else dfinal_state.clone()
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


def run_case(shape, mode, seed):
    cpu_inputs = make_inputs(shape, mode, seed)
    expected = reference(*cpu_inputs)
    states_start, d_states_start, d_a_cumsum, dfinal_state = cpu_inputs
    actual = torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd(
        states_start.npu(),
        d_states_start.npu(),
        d_a_cumsum.npu(),
        None if dfinal_state is None else dfinal_state.npu(),
    )
    torch.npu.synchronize()
    output_metrics = {}
    for name, got, want in zip(
        ("d_chunk_states", "dinitial_state", "dA_chunk_last"),
        actual,
        expected,
    ):
        got_cpu = got.cpu().float()
        want_cpu = want.float()
        abs_err = (got_cpu - want_cpu).abs()
        rel_err = abs_err / (want_cpu.abs() + 1.0e-7)
        both_zero = not torch.count_nonzero(got_cpu) and not torch.count_nonzero(
            want_cpu
        )
        cosine = (
            1.0
            if both_zero
            else torch.nn.functional.cosine_similarity(
                got_cpu.flatten(), want_cpu.flatten(), dim=0
            ).item()
        )
        output_metrics[name] = {
            "MERE": rel_err.mean().item(),
            "MARE": rel_err.max().item(),
            "max_abs_err": abs_err.max().item(),
            "mean_abs_err": abs_err.mean().item(),
            "cosine_sim": cosine,
        }
    passed = all(
        metric["MERE"] < THRESHOLD
        and metric["MARE"] < 10 * THRESHOLD
        for metric in output_metrics.values()
    )
    return output_metrics, passed
