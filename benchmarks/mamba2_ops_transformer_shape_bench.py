"""Same-shape benchmark matrix for the ops-transformer Mamba-2 comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


CASES = {
    "s1": (1, 256, 128, 64, 128, 8),
    "s2": (1, 512, 128, 64, 128, 8),
    "s3": (1, 1024, 128, 64, 128, 8),
    "s4": (1, 2048, 128, 64, 128, 8),
    "s5": (2, 2048, 128, 64, 128, 8),
}
LOGICAL_CHUNK_SIZE = 256
EXECUTION_MICRO_CHUNK = 128


def make_inputs(case: str):
    batch, seqlen, nheads, headdim, dstate, ngroups = CASES[case]
    generator = torch.Generator().manual_seed(20260811)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    values = {
        "x": randn(batch, seqlen, nheads, headdim),
        "dt": 0.01
        + 0.1
        * torch.rand(batch, seqlen, nheads, generator=generator),
        "A": -(0.1 + 0.4 * torch.rand(nheads, generator=generator)),
        "B": randn(batch, seqlen, ngroups, dstate) / 5,
        "C": randn(batch, seqlen, ngroups, dstate) / 5,
        "D": randn(nheads),
        "dt_bias": randn(nheads) * 0.1,
        "initial_states": randn(batch, nheads, headdim, dstate) / 5,
    }
    return {name: value.npu() for name, value in values.items()}


def invoke_ascendc(values):
    return ascend_kernel.mamba2_ssd_fwd(
        values["x"],
        values["dt"],
        values["A"],
        values["B"],
        values["C"],
        LOGICAL_CHUNK_SIZE,
        D=values["D"],
        dt_bias=values["dt_bias"],
        dt_softplus=True,
        initial_states=values["initial_states"],
        return_final_state=True,
    )


def invoke_pytorch(values):
    return ssd_chunk_scan_ref(
        values["x"],
        values["dt"],
        values["A"],
        values["B"],
        values["C"],
        LOGICAL_CHUNK_SIZE,
        D=values["D"],
        dt_bias=values["dt_bias"],
        dt_softplus=True,
        initial_states=values["initial_states"],
        return_final_state=True,
    )


def error_metrics(actual, expected):
    error = actual.float() - expected.float()
    return {
        "max_abs": error.abs().max().item(),
        "nrmse": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item(),
        "cosine": F.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def time_device(call, warmup: int, repeat: int):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends):
        start.record()
        call()
        end.record()
    torch.npu.synchronize()
    times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
        "p90_ms": sorted(times)[int(0.9 * (repeat - 1))],
    }


def run_case(case: str, warmup: int, repeat: int, time_reference: bool):
    values = make_inputs(case)
    with torch.no_grad():
        expected_out, expected_state = invoke_pytorch(values)
        actual_out, actual_state = invoke_ascendc(values)
        torch.npu.synchronize()
        result = {
            "case": case,
            "tensor_shape_b_l_h_p_n_g": list(CASES[case]),
            "logical_chunk_size": LOGICAL_CHUNK_SIZE,
            "execution_micro_chunk": EXECUTION_MICRO_CHUNK,
            "dtype": "float32",
            "features": "D+dt_bias+softplus+initial+final,no_z",
            "ascendc": time_device(
                lambda: invoke_ascendc(values), warmup, repeat
            ),
            "output": error_metrics(actual_out, expected_out),
            "final_state": error_metrics(actual_state, expected_state),
        }
        if time_reference:
            result["pytorch"] = time_device(
                lambda: invoke_pytorch(values), warmup, repeat
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=list(CASES)
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--time-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [
        run_case(case, args.warmup, args.repeat, args.time_reference)
        for case in args.cases
    ]
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
