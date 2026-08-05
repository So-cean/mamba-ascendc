#!/usr/bin/env python3
"""Scaling benchmark for the T=P=64, N in {64,128} chunk MIX gate."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa: F401

WARMUP = 30
REPEAT = 100
BATCHED_REPEAT = 200

CASES = (
    ("n064_task_001", 1, 1, 1, 1, 64),
    ("n064_task_020", 1, 10, 2, 1, 64),
    ("n064_task_064", 2, 8, 4, 2, 64),
    ("n128_task_001", 1, 1, 1, 1, 128),
    ("n128_task_020", 1, 10, 2, 2, 128),
    ("n128_task_064", 2, 8, 4, 2, 128),
    ("n128_grouped_headtask_256", 2, 16, 8, 4, 128),
)


def load_operator() -> None:
    library = ctypes.CDLL(
        os.environ["MAMBA_CHUNK_MIX_OP_API_LIB"], mode=ctypes.RTLD_GLOBAL
    )
    assert getattr(library, "aclnnMamba2SsdChunkMix")
    import ascend_kernel  # noqa: F401,E402


def make_inputs(batch: int, heads: int, chunks: int, groups: int,
                state_dim: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (0.1 * torch.randn(
        (batch, heads, chunks, 64, 64), generator=generator
    )).half().npu()
    d_a = -(0.005 + 0.045 * torch.rand(
        (batch, heads, chunks, 64), generator=generator
    ))
    d_a = torch.cumsum(d_a, dim=-1).float().npu()
    b = (0.1 * torch.randn(
        (batch, chunks, groups, 64, state_dim), generator=generator
    )).half().npu()
    c = (0.1 * torch.randn(
        (batch, chunks, groups, 64, state_dim), generator=generator
    )).half().npu()
    return x, d_a, b, c


def native_forward(x, d_a, b_tensor, c_tensor):
    heads = x.shape[1]
    groups = b_tensor.shape[2]
    repeat = heads // groups
    b_head = b_tensor.repeat_interleave(repeat, dim=2).permute(0, 2, 1, 3, 4)
    c_head = c_tensor.repeat_interleave(repeat, dim=2).permute(0, 2, 1, 3, 4)
    cb = torch.matmul(c_head, b_head.transpose(-1, -2)).float()
    decay = torch.exp(d_a.unsqueeze(-1) - d_a.unsqueeze(-2))
    weights = torch.tril(cb * decay).half()
    y = torch.matmul(weights, x).float()
    state_decay = torch.exp(d_a[..., -1:] - d_a)
    weighted_x = (x.float() * state_decay.unsqueeze(-1)).half()
    state = torch.matmul(weighted_x.transpose(-1, -2), b_head).float()
    return y, state


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def benchmark(fn) -> dict[str, float]:
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()

    times = []
    for _ in range(REPEAT):
        torch.npu.synchronize()
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1.0e3)

    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(BATCHED_REPEAT):
        fn()
    torch.npu.synchronize()
    batched_ms = (time.perf_counter() - start) * 1.0e3 / BATCHED_REPEAT
    return {
        "p50_ms": statistics.median(times),
        "p90_ms": percentile(times, 0.90),
        "p99_ms": percentile(times, 0.99),
        "batched_avg_ms": batched_ms,
    }


def main() -> int:
    load_operator()
    rows = []
    for index, (tag, batch, heads, chunks, groups, state_dim) in enumerate(CASES):
        inputs = make_inputs(
            batch, heads, chunks, groups, state_dim, 20260802 + index
        )
        custom_fn = lambda: torch.ops.mamba_ascend.mamba2_ssd_chunk_mix(*inputs)
        native_fn = lambda: native_forward(*inputs)
        custom = benchmark(custom_fn)
        native = benchmark(native_fn)
        tasks = batch * heads * chunks
        row = {
            "tag": tag,
            "shape": [batch, heads, chunks, groups, 64, 64, state_dim],
            "tasks": tasks,
            "used_mix_groups": min(tasks, 20),
            "custom": custom,
            "native": native,
            "native_over_custom_p50": native["p50_ms"] / custom["p50_ms"],
            "native_over_custom_batched": (
                native["batched_avg_ms"] / custom["batched_avg_ms"]
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    report_path = Path(os.environ.get(
        "MAMBA_CHUNK_MIX_BENCH_OUTPUT",
        Path(__file__).with_name("mamba2_ssd_chunk_mix_scaling_report.json"),
    ))
    report_path.write_text(json.dumps({
        "warmup": WARMUP,
        "repeat": REPEAT,
        "batched_repeat": BATCHED_REPEAT,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"report={report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
