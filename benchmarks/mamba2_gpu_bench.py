"""A100 steady-state benchmark for the official mamba_ssm Mamba-2 forward."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from mamba2_shape_matrix import (
    CASES,
    SHAPE_CASES,
    SUITES,
    select_case_names,
)


def make_inputs(case):
    batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size = CASES[case]
    generator = torch.Generator(device="cpu").manual_seed(20260801)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    x = randn(batch, seqlen, nheads, headdim)
    dt = 0.01 + 0.1 * torch.rand(
        batch, seqlen, nheads, generator=generator, dtype=torch.float32
    )
    A = -0.1 - 0.4 * torch.rand(nheads, generator=generator)
    B = randn(batch, seqlen, ngroups, dstate) / 5
    C = randn(batch, seqlen, ngroups, dstate) / 5
    D = randn(nheads, headdim)
    z = randn(batch, seqlen, nheads, headdim)
    dt_bias = randn(nheads) * 0.1
    initial_states = randn(batch, nheads, headdim, dstate) / 5
    values = tuple(
        value.cuda()
        for value in (x, dt, A, B, C, D, z, dt_bias, initial_states)
    )
    return values, chunk_size


def benchmark(case, warmup, repeat):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    inputs, chunk_size = make_inputs(case)
    x, dt, A, B, C, D, z, dt_bias, initial_states = inputs

    def run():
        return mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            dt_softplus=True,
            return_final_states=True,
        )

    with torch.no_grad():
        for _ in range(warmup):
            output = run()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        for start, end in zip(starts, ends):
            start.record()
            output = run()
            end.record()
        torch.cuda.synchronize()
    times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size = CASES[case]
    return {
        "status": "ok",
        "backend": "mamba_ssm",
        "device": torch.cuda.get_device_name(0),
        "case": case,
        "suite_axis": SHAPE_CASES[case].axis,
        "expected_910b3_path": SHAPE_CASES[case].expected_910b3_path,
        "full_feature_input_mib": SHAPE_CASES[case].full_feature_input_mib,
        "shape": CASES[case],
        "shape_b_l_h_p_n_c_g": [
            batch, seqlen, nheads, headdim, dstate, chunk_size, ngroups
        ],
        "dtype": "float32",
        "warmup": warmup,
        "repeat": repeat,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "p90_ms": sorted(times_ms)[int(0.9 * (len(times_ms) - 1))],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "checksum": output[0].float().sum().item(),
        "final_state_checksum": output[1].float().sum().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES))
    parser.add_argument("--suite", choices=tuple(SUITES), default="standard")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--output",
        help="Optional JSONL output path; parent directories are created.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed/OOM case and continue the selected shape suite.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available")
    case_names = select_case_names(args.cases, args.suite)
    output_handle = None
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output_handle = destination.open("w", encoding="utf-8")
    try:
        for case in case_names:
            try:
                result = benchmark(case, args.warmup, args.repeat)
            except (OSError, RuntimeError) as error:
                if not args.continue_on_error:
                    raise
                torch.cuda.empty_cache()
                result = {
                    "status": "error",
                    "backend": "mamba_ssm",
                    "device": torch.cuda.get_device_name(0),
                    "case": case,
                    "shape_b_l_h_p_n_c_g": list(SHAPE_CASES[case].public_tuple),
                    "error": str(error),
                }
            line = json.dumps(result, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            if output_handle is not None:
                output_handle.write(line + "\n")
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()


if __name__ == "__main__":
    main()
