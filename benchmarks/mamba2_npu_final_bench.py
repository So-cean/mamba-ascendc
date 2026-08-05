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
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


CASES = {
    "tiny": (1, 128, 2, 64, 64, 1, 64),
    "small": (2, 512, 8, 64, 64, 1, 64),
    "medium": (4, 2048, 16, 64, 128, 4, 128),
    "extreme": (8, 4096, 32, 64, 128, 8, 128),
    "super_extreme": (8, 8192, 32, 64, 128, 8, 128),
    "ultra_extreme": (8, 16384, 32, 64, 128, 8, 128),
}


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
    )
    return tuple(value.npu() for value in values), chunk_size


def metrics(actual, expected):
    error = actual.float() - expected.float()
    return {
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
    x, dt, A, B, C, D, z, dt_bias = values
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
        dt_softplus=True,
        return_final_state=True,
    )


def run_case(case, warmup, repeat, skip_precision=False):
    values, chunk_size = make_inputs(case)
    batch, seqlen, nheads, headdim, dstate, ngroups, _ = CASES[case]
    x, dt, A, B, C, D, z, dt_bias = values
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
                dt_softplus=True,
                return_final_state=True,
            )
            out, state = invoke(values, chunk_size)
            torch.npu.synchronize()
            out_metrics = metrics(out, ref_out)
            state_metrics = metrics(state, ref_state)
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
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=tuple(CASES))
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--output")
    parser.add_argument("--profile-root")
    parser.add_argument(
        "--skip-precision",
        action="store_true",
        help="Skip the PyTorch reference for memory-heavy performance-only cases.",
    )
    args = parser.parse_args()
    results = [
        run_case(case, args.warmup, args.repeat, args.skip_precision)
        for case in args.cases
    ]
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    if args.profile_root:
        for case in args.cases:
            profile(case, str(Path(args.profile_root) / case))


if __name__ == "__main__":
    main()
