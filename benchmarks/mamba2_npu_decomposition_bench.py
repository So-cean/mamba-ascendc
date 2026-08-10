"""Compare NPU PyTorch composition, Triton-Ascend, and AscendC forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch_npu

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref
from mamba_triton_ascend.mamba2 import mamba_chunk_scan_combined

from mamba2_npu_final_bench import BASE_CASES, CASES, make_inputs


BACKENDS = ("pytorch", "triton-ascend", "ascendc")


def invoke(backend: str, values, chunk_size: int):
    x, dt, A, B, C, D, z, dt_bias = values
    common = {
        "D": D,
        "z": z,
        "dt_bias": dt_bias,
        "dt_softplus": True,
    }
    if backend == "pytorch":
        return ssd_chunk_scan_ref(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            return_final_state=True,
            **common,
        )
    if backend == "triton-ascend":
        return mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            return_final_states=True,
            backend="triton",
            **common,
        )
    if backend == "ascendc":
        return ascend_kernel.mamba2_ssd_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            return_final_state=True,
            **common,
        )
    raise ValueError(f"unsupported backend: {backend}")


def benchmark(backend: str, case: str, warmup: int, repeat: int) -> dict:
    values, chunk_size = make_inputs(case)
    with torch.no_grad():
        for _ in range(warmup):
            output = invoke(backend, values, chunk_size)
        torch.npu.synchronize()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
        for start, end in zip(starts, ends):
            start.record()
            output = invoke(backend, values, chunk_size)
            end.record()
        torch.npu.synchronize()
    times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    out = output[0] if isinstance(output, tuple) else output
    return {
        "benchmark": "npu_framework_decomposition",
        "backend": backend,
        "device": torch.npu.get_device_name(0),
        "case": case,
        "shape_b_l_h_p_n_c_g": list(CASES[case][0:5])
        + [CASES[case][6], CASES[case][5]],
        "dtype": "float32",
        "warmup": warmup,
        "repeat": repeat,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "p90_ms": sorted(times_ms)[int(0.9 * (len(times_ms) - 1))],
        "checksum": out.float().sum().item(),
    }


def profile(backend: str, case: str, root: Path) -> None:
    values, chunk_size = make_inputs(case)
    output = root / backend / case
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
            invoke(backend, values, chunk_size)
            profiler.step()
    torch.npu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASES), default=("medium",)
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--profile-root")
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is available")
    results = []
    for case in args.cases:
        if case not in BASE_CASES and not case.startswith("seq_"):
            raise ValueError(f"unsupported case: {case}")
        for backend in args.backends:
            result = benchmark(backend, case, args.warmup, args.repeat)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if args.profile_root:
                profile(backend, case, Path(args.profile_root))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
