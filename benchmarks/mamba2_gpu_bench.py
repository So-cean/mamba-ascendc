"""A100 steady-state benchmark for the official mamba_ssm Mamba-2 forward."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


BASE_CASES = {
    "tiny": (1, 128, 2, 64, 64, 1, 64),
    "small": (2, 512, 8, 64, 64, 1, 64),
    "medium": (4, 2048, 16, 64, 128, 4, 128),
    "extreme": (8, 4096, 32, 64, 128, 8, 128),
    "super_extreme": (8, 8192, 32, 64, 128, 8, 128),
    "ultra_extreme": (8, 16384, 32, 64, 128, 8, 128),
}
SEQUENCE_CASES = {
    f"seq_{length}": (8, length, 32, 64, 128, 8, 128)
    for length in (512, 1024, 2048, 4096, 8192, 16384)
}
CASES = {**BASE_CASES, **SEQUENCE_CASES}


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
    values = tuple(value.cuda() for value in (x, dt, A, B, C, D, z, dt_bias))
    return values, chunk_size


def benchmark(case, warmup, repeat):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    inputs, chunk_size = make_inputs(case)
    x, dt, A, B, C, D, z, dt_bias = inputs

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
            dt_softplus=True,
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
        "backend": "mamba_ssm",
        "device": torch.cuda.get_device_name(0),
        "case": case,
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
        "checksum": output.float().sum().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(BASE_CASES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--output",
        help="Optional JSONL output path; parent directories are created.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available")
    results = [benchmark(case, args.warmup, args.repeat) for case in args.cases]
    rendered = [json.dumps(result, ensure_ascii=False) for result in results]
    for line in rendered:
        print(line)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(rendered) + "\n")


if __name__ == "__main__":
    main()
