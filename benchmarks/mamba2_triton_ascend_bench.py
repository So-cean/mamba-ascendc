"""Steady-state Mamba-2 SSD forward benchmark for Triton-Ascend."""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401  # Register the NPU backend.

from mamba_triton_ascend.mamba2 import mamba_chunk_scan_combined


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


def make_inputs(case: str):
    batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size = CASES[case]
    generator = torch.Generator(device="cpu").manual_seed(20260801)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    values = (
        randn(batch, seqlen, nheads, headdim),
        0.01 + 0.1 * torch.rand(
            batch, seqlen, nheads, generator=generator, dtype=torch.float32
        ),
        -(0.1 + 0.4 * torch.rand(nheads, generator=generator)),
        randn(batch, seqlen, ngroups, dstate) / 5,
        randn(batch, seqlen, ngroups, dstate) / 5,
        randn(nheads, headdim),
        randn(batch, seqlen, nheads, headdim),
        randn(nheads) * 0.1,
    )
    return tuple(value.npu() for value in values), chunk_size


def benchmark(case: str, warmup: int, repeat: int) -> dict:
    values, chunk_size = make_inputs(case)
    x, dt, A, B, C, D, z, dt_bias = values

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
            backend="triton",
        )

    with torch.no_grad():
        for _ in range(warmup):
            output = run()
        torch.npu.synchronize()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
        for start, end in zip(starts, ends):
            start.record()
            output = run()
            end.record()
        torch.npu.synchronize()

    times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    batch, seqlen, nheads, headdim, dstate, ngroups, _ = CASES[case]
    return {
        "backend": "triton-ascend",
        "device": torch.npu.get_device_name(0),
        "case": case,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASES), default=tuple(BASE_CASES)
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is available")
    for case in args.cases:
        print(json.dumps(benchmark(case, args.warmup, args.repeat), ensure_ascii=False))


if __name__ == "__main__":
    main()
