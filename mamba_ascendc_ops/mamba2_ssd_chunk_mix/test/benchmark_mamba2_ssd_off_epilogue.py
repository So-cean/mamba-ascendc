#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


library_path = os.environ["MAMBA_CHUNK_MIX_OP_API_LIB"]
ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
import ascend_kernel  # noqa: E402,F401


CASES = [
    (1, 1, 1, 1, 128),
    (1, 10, 2, 2, 128),
    (1, 8, 2, 8, 128),
    (2, 8, 2, 8, 128),
    (2, 8, 2, 16, 128),
    (2, 16, 4, 16, 128),
    (4, 16, 4, 16, 128),
    (4, 16, 4, 32, 128),
]


def native(c, states, da, y_diag, x, d, z):
    batch, heads, chunks, tile, state_dim = states.shape
    groups = c.shape[2]
    heads_per_group = heads // groups
    c_grouped = c.permute(0, 2, 1, 3, 4).unsqueeze(2)
    states_t = states.reshape(
        batch, groups, heads_per_group, chunks, tile, state_dim
    ).transpose(-1, -2).half()
    y_off = torch.matmul(c_grouped, states_t).float()
    y_off = y_off * torch.exp(
        da.reshape(batch, groups, heads_per_group, chunks, tile)
    ).unsqueeze(-1)
    y_local = y_diag.reshape(
        batch, groups, heads_per_group, chunks, tile, tile
    )
    out = (y_local + y_off).permute(0, 3, 4, 1, 2, 5)
    out = out.reshape(batch, chunks * tile, heads, tile)
    return (out + x * d.view(1, 1, heads, tile)) * F.silu(z)


def measure(fn, warmup=30, repeat=200):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.npu.synchronize()
    values = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return statistics.median(values)


def main():
    results = []
    with torch.no_grad():
        for batch, heads, groups, chunks, state_dim in CASES:
            tile = 64
            torch.manual_seed(20260802 + batch + heads + chunks)
            c = torch.randn(
                batch, chunks, groups, tile, state_dim, device="npu"
            ).half()
            states = torch.randn(
                batch, heads, chunks, tile, state_dim, device="npu"
            )
            da = -torch.rand(batch, heads, chunks, tile, device="npu")
            y_diag = torch.randn(
                batch, heads, chunks, tile, tile, device="npu"
            )
            x = torch.randn(
                batch, chunks * tile, heads, tile, device="npu"
            )
            d = torch.randn(heads, tile, device="npu")
            z = torch.randn_like(x)
            args = (c, states, da, y_diag, x, d, z)
            custom_ms = measure(
                lambda: torch.ops.mamba_ascend.mamba2_ssd_off_epilogue(*args)
            )
            native_ms = measure(lambda: native(*args))
            result = {
                "shape_b_h_g_k_n": [batch, heads, groups, chunks, state_dim],
                "tasks": batch * heads * chunks,
                "custom_ms": custom_ms,
                "native_ms": native_ms,
                "speedup": native_ms / custom_ms,
            }
            print(json.dumps(result), flush=True)
            results.append(result)
    output = Path(
        "mamba_ascendc_ops/mamba2_ssd_chunk_mix/test/"
        "mamba2_ssd_off_epilogue_scaling_report.json"
    )
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
