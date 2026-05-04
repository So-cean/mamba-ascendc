# Copyright (c) 2024, mamba-triton-ascend authors.
# Benchmark script for selective_scan and selective_state_update.

import argparse
import time
import torch

from mamba_triton_ascend.ops.reference import (
    selective_scan_ref,
    selective_state_update_ref,
)


def benchmark_selective_scan(batch, dim, dstate, seqlen, device="cpu", dtype=torch.float32, n_iters=10):
    """Benchmark selective_scan forward pass."""
    torch.manual_seed(42)
    u = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
    delta = torch.rand(batch, dim, seqlen, device=device, dtype=dtype) * 0.1 + 0.01
    A = torch.randn(dim, dstate, device=device, dtype=dtype) * 0.1
    B = torch.randn(batch, dstate, seqlen, device=device, dtype=dtype)
    C = torch.randn(batch, dstate, seqlen, device=device, dtype=dtype)
    D = torch.randn(dim, device=device, dtype=dtype)
    z = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
    delta_bias = torch.randn(dim, device=device, dtype=dtype) * 0.01

    # Warmup
    for _ in range(2):
        _ = selective_scan_ref(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True)

    # Timing
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        out = selective_scan_ref(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    elapsed = (t1 - t0) / n_iters * 1000  # ms
    tokens_per_sec = batch * seqlen / (elapsed / 1000)
    return elapsed, tokens_per_sec


def benchmark_selective_state_update(batch, dim, dstate, device="cpu", dtype=torch.float32, n_iters=100):
    """Benchmark selective_state_update (single step)."""
    torch.manual_seed(42)
    state = torch.randn(batch, dim, dstate, device=device, dtype=dtype)
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    dt = torch.rand(batch, dim, device=device, dtype=dtype) * 0.1 + 0.01
    A = torch.randn(dim, dstate, device=device, dtype=dtype) * 0.1
    B = torch.randn(batch, dstate, device=device, dtype=dtype)
    C = torch.randn(batch, dstate, device=device, dtype=dtype)

    # Warmup
    for _ in range(2):
        _ = selective_state_update_ref(state, x, dt, A, B, C)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        out = selective_state_update_ref(state, x, dt, A, B, C)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    elapsed = (t1 - t0) / n_iters * 1000  # ms per step
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba selective scan")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "npu"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--op", type=str, default="scan", choices=["scan", "state_update", "all"])
    args = parser.parse_args()

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    print(f"Device: {args.device}, dtype: {args.dtype}")
    print("=" * 60)

    if args.op in ("scan", "all"):
        print("\n--- Selective Scan Forward Benchmark ---")
        configs = [
            (1, 512, 16, 512),
            (2, 512, 16, 2048),
            (2, 1024, 64, 2048),
            (4, 1024, 64, 4096),
        ]
        for batch, dim, dstate, seqlen in configs:
            try:
                elapsed, tps = benchmark_selective_scan(
                    batch, dim, dstate, seqlen, device=args.device, dtype=dtype, n_iters=5
                )
                print(f"  B={batch:2d} D={dim:4d} N={dstate:3d} L={seqlen:5d} | "
                      f"{elapsed:7.3f} ms | {tps/1000:8.1f} K tokens/s")
            except Exception as e:
                print(f"  B={batch:2d} D={dim:4d} N={dstate:3d} L={seqlen:5d} | FAILED: {e}")

    if args.op in ("state_update", "all"):
        print("\n--- Selective State Update Benchmark ---")
        configs = [
            (1, 512, 16),
            (2, 512, 16),
            (4, 1024, 64),
        ]
        for batch, dim, dstate in configs:
            try:
                elapsed = benchmark_selective_state_update(
                    batch, dim, dstate, device=args.device, dtype=dtype, n_iters=100
                )
                print(f"  B={batch:2d} D={dim:4d} N={dstate:3d} | {elapsed:7.3f} ms/step")
            except Exception as e:
                print(f"  B={batch:2d} D={dim:4d} N={dstate:3d} | FAILED: {e}")


if __name__ == "__main__":
    main()
