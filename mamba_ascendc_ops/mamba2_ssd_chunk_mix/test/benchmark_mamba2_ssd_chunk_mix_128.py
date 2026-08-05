#!/usr/bin/env python3
"""Equal-token Event benchmark for T=64 versus experimental T=128."""

import ctypes
import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu  # noqa: F401


CASES = {
    "medium": (4, 2048, 16, 4),
    "extreme": (8, 4096, 32, 8),
    "super_extreme": (8, 8192, 32, 8),
}
WARMUP = int(os.environ.get("MAMBA_CHUNK128_WARMUP", "30"))
REPEAT = int(os.environ.get("MAMBA_CHUNK128_REPEAT", "200"))


def make_inputs(batch, seqlen, heads, groups, tile):
    chunks = seqlen // tile
    x = (0.1 * torch.randn(batch, heads, chunks, tile, 64,
                           device="npu")).half()
    step = -(0.005 + 0.045 * torch.rand(
        batch, heads, chunks, tile, device="npu"))
    da = torch.cumsum(step, dim=-1)
    b = (0.1 * torch.randn(batch, chunks, groups, tile, 128,
                           device="npu")).half()
    c = (0.1 * torch.randn(batch, chunks, groups, tile, 128,
                           device="npu")).half()
    return x, da, b, c


def bench(inputs):
    fn = lambda: torch.ops.mamba_ascend.mamba2_ssd_chunk_mix(*inputs)
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.npu.synchronize()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(REPEAT)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(REPEAT)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.npu.synchronize()
    values = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p90_ms": sorted(values)[int(0.9 * (len(values) - 1))],
        "min_ms": min(values),
    }


def main():
    ctypes.CDLL(os.environ["MAMBA_CHUNK_MIX_OP_API_LIB"],
                mode=ctypes.RTLD_GLOBAL)
    import ascend_kernel  # noqa: F401,E402

    selected = os.environ.get(
        "MAMBA_CHUNK128_CASES", "medium extreme super_extreme"
    ).split()
    rows = []
    for name in selected:
        batch, seqlen, heads, groups = CASES[name]
        timings = {}
        for tile in (64, 128):
            inputs = make_inputs(batch, seqlen, heads, groups, tile)
            timings[f"t{tile}"] = bench(inputs)
            del inputs
            torch.npu.empty_cache()
        row = {
            "case": name,
            "shape_b_l_h_p_n_g": [batch, seqlen, heads, 64, 128, groups],
            "t64": timings["t64"],
            "t128": timings["t128"],
            "t64_over_t128": (
                timings["t64"]["median_ms"] /
                timings["t128"]["median_ms"]
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = Path(os.environ.get(
        "MAMBA_CHUNK128_BENCH_OUTPUT",
        "docs/report/mamba2_ascendc_v17_chunk128_component_bench_2026-08-02.json",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "warmup": WARMUP, "repeat": REPEAT, "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
