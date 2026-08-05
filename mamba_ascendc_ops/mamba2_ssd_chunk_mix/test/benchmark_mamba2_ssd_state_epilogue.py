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
    (1, 8, 2, 4, 64, 128),
    (2, 8, 2, 8, 64, 128),
    (4, 16, 4, 32, 64, 128),
    (4, 16, 4, 64, 64, 128),
    (4, 16, 4, 128, 64, 128),
    (8, 32, 8, 64, 64, 128),
    (8, 32, 8, 128, 64, 128),
    # Production T128 paths used by the batch32 Vision-Mamba2 stages.
    (4, 16, 4, 16, 128, 128),
    (8, 32, 8, 32, 128, 128),
    (8, 32, 8, 64, 128, 128),
]


def native(chunk_states, da, c, y_diag, x, d, z):
    chunk_states_pn = (chunk_states.transpose(-1, -2).contiguous()
                       if chunk_states.shape[-1] == 64 else chunk_states)
    states, final = torch.ops.mamba_ascend.mamba2_ssd_state_passing(
        chunk_states_pn, da, None
    )
    batch, heads, chunks, head_dim, state_dim = states.shape
    chunk_size = da.shape[-1]
    groups = c.shape[2]
    heads_per_group = heads // groups
    c_grouped = c.permute(0, 2, 1, 3, 4).unsqueeze(2)
    states_t = states.reshape(
        batch, groups, heads_per_group, chunks, head_dim, state_dim
    ).transpose(-1, -2).half()
    y_off = torch.matmul(c_grouped, states_t).float()
    y_off = y_off * torch.exp(
        da.reshape(batch, groups, heads_per_group, chunks, chunk_size)
    ).unsqueeze(-1)
    y_local = y_diag.reshape(
        batch, groups, heads_per_group, chunks, chunk_size, head_dim
    )
    out = (y_local + y_off).permute(0, 3, 4, 1, 2, 5)
    out = out.reshape(batch, chunks * chunk_size, heads, head_dim)
    out = (out + x * d.view(1, 1, heads, head_dim)) * F.silu(z)
    return out, final


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
    return statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )


def main():
    results = []
    with torch.no_grad():
        for batch, heads, groups, chunks, chunk_size, state_dim in CASES:
            head_dim = 64
            torch.manual_seed(
                20260802 + batch + heads + chunks + chunk_size
            )
            state_shape = ((batch, heads, chunks, state_dim, head_dim)
                           if state_dim == 64 or chunk_size == 128 else
                           (batch, heads, chunks, head_dim, state_dim))
            chunk_states = 0.05 * torch.randn(*state_shape, device="npu")
            da = torch.cumsum(
                -(0.001 + 0.01 * torch.rand(
                    batch, heads, chunks, chunk_size, device="npu"
                )),
                dim=-1,
            )
            c = (0.1 * torch.randn(
                batch, chunks, groups, chunk_size, state_dim, device="npu"
            )).half()
            y_diag = 0.1 * torch.randn(
                batch, heads, chunks, chunk_size, head_dim, device="npu"
            )
            x = 0.2 * torch.randn(
                batch, chunks * chunk_size, heads, head_dim, device="npu"
            )
            d = 0.2 * torch.randn(heads, head_dim, device="npu")
            z = 0.2 * torch.randn_like(x)
            args = (chunk_states, da, c, y_diag, x, d, z)
            fused_ms = measure(
                lambda: torch.ops.mamba_ascend.mamba2_ssd_state_epilogue(
                    *args, None
                )
            )
            native_ms = measure(lambda: native(*args))
            result = {
                "shape_b_h_g_k_c_n": [
                    batch, heads, groups, chunks, chunk_size, state_dim
                ],
                "stream_tasks": batch * heads,
                "chunk_tasks": batch * heads * chunks,
                "fused_ms": fused_ms,
                "native_ms": native_ms,
                "speedup": native_ms / fused_ms,
            }
            print(json.dumps(result), flush=True)
            results.append(result)
    output = Path(os.environ.get(
        "MAMBA_STATE_EPILOGUE_BENCH_OUTPUT",
        "mamba_ascendc_ops/mamba2_ssd_chunk_mix/test/"
        "mamba2_ssd_state_epilogue_scaling_report.json",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
