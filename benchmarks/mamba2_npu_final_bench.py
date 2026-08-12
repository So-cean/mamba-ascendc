"""Formal end-to-end benchmark of the public AscendC Mamba2 wrapper."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

import ascend_kernel
from mamba2_shape_matrix import (
    BASE_CASES,
    CASES,
    SEQUENCE_CASES,
    SHAPE_CASES,
    SUITES,
    select_case_names,
)
from mamba_torch.ssd_reference import ssd_chunk_scan_ref

__all__ = ["BASE_CASES", "CASES", "SEQUENCE_CASES", "make_inputs"]


def make_inputs(case):
    batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size = CASES[case]
    generator = torch.Generator().manual_seed(20260801)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    values = (
        randn(batch, seqlen, nheads, headdim),
        0.01 + 0.1 * torch.rand(batch, seqlen, nheads, generator=generator),
        -(0.1 + 0.4 * torch.rand(nheads, generator=generator)),
        randn(batch, seqlen, ngroups, dstate) / 5,
        randn(batch, seqlen, ngroups, dstate) / 5,
        randn(nheads, headdim),
        randn(batch, seqlen, nheads, headdim),
        randn(nheads) * 0.1,
        randn(batch, nheads, headdim, dstate) / 5,
    )
    return tuple(value.npu() for value in values), chunk_size


def metrics(actual, expected):
    error = actual.float() - expected.float()
    return {
        "status": "ok",
        "max_abs": error.abs().max().item(),
        "mean_abs": error.abs().mean().item(),
        "nrmse": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item(),
        "cosine": F.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item(),
        "allclose": bool(torch.allclose(actual, expected, rtol=3e-2, atol=1e-2)),
    }


def invoke(values, chunk_size):
    x, dt, A, B, C, D, z, dt_bias, initial_states = values
    return ascend_kernel.mamba2_ssd_fwd(
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
        return_final_state=True,
    )


def run_case(case, warmup, repeat, skip_precision=False):
    values, chunk_size = make_inputs(case)
    batch, seqlen, nheads, headdim, dstate, ngroups, _ = CASES[case]
    x, dt, A, B, C, D, z, dt_bias, initial_states = values
    with torch.no_grad():
        if skip_precision:
            out, state = invoke(values, chunk_size)
            torch.npu.synchronize()
            out_metrics = None
            state_metrics = None
        else:
            ref_out, ref_state = ssd_chunk_scan_ref(
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
                return_final_state=True,
            )
            out, state = invoke(values, chunk_size)
            torch.npu.synchronize()
            out_metrics = metrics(out, ref_out)
            state_metrics = metrics(state, ref_state)
            if not (out_metrics["allclose"] and state_metrics["allclose"]):
                raise AssertionError(
                    f"precision gate failed before timing case={case}: "
                    f"out={out_metrics}, final_state={state_metrics}"
                )
        for _ in range(warmup):
            out, state = invoke(values, chunk_size)
        torch.npu.synchronize()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
        for start, end in zip(starts, ends):
            start.record()
            out, state = invoke(values, chunk_size)
            end.record()
        torch.npu.synchronize()
        times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "backend": "ascendc",
        "device": torch.npu.get_device_name(0),
        "case": case,
        "suite_axis": SHAPE_CASES[case].axis,
        "expected_910b3_path": SHAPE_CASES[case].expected_910b3_path,
        "full_feature_input_mib": SHAPE_CASES[case].full_feature_input_mib,
        "shape_b_l_h_p_n_c_g": [
            batch, seqlen, nheads, headdim, dstate, chunk_size, ngroups
        ],
        "warmup": warmup,
        "repeat": repeat,
        "dtype": "float32",
        "precision_skipped": skip_precision,
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "p90_ms": sorted(times)[int(0.9 * (len(times) - 1))],
        "min_ms": min(times),
        "out": out_metrics,
        "final_state": state_metrics,
    }


def profile(case, output_dir):
    values, chunk_size = make_inputs(case)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    schedule = torch_npu.profiler.schedule(wait=0, warmup=5, active=5, repeat=1)
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )
    with torch.no_grad(), torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
        experimental_config=config,
    ) as profiler:
        for _ in range(10):
            invoke(values, chunk_size)
            profiler.step()
    torch.npu.synchronize()
    print(f"profile_dir={output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASES)
    )
    parser.add_argument("--suite", choices=tuple(SUITES), default="standard")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--output")
    parser.add_argument("--profile-root")
    parser.add_argument(
        "--skip-precision",
        action="store_true",
        help="Skip the PyTorch reference for memory-heavy performance-only cases.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed/OOM case and continue the selected shape suite.",
    )
    args = parser.parse_args()
    case_names = select_case_names(args.cases, args.suite)
    output_handle = None
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output.open("w", encoding="utf-8")
    try:
        for case in case_names:
            try:
                result = run_case(
                    case, args.warmup, args.repeat, args.skip_precision
                )
            except (AssertionError, RuntimeError) as error:
                if not args.continue_on_error:
                    raise
                torch.npu.empty_cache()
                result = {
                    "status": "error",
                    "backend": "ascendc",
                    "device": torch.npu.get_device_name(0),
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
    if args.profile_root:
        for case in case_names:
            profile(case, str(Path(args.profile_root) / case))


if __name__ == "__main__":
    main()
