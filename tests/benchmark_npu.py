# Copyright (c) 2024, mamba-triton-ascend authors.
# NPU performance benchmark for selective_scan and selective_state_update.
# Separates compile time from execution time; no reference comparison.

import argparse
import time
import torch

from mamba_triton_ascend.ops.triton_kernels.selective_scan_fwd import selective_scan_fwd
from mamba_triton_ascend.ops.triton_kernels.selective_scan_bwd import selective_scan_bwd
from mamba_triton_ascend.ops.selective_scan import selective_scan_fn
from mamba_triton_ascend.ops.selective_state_update import selective_state_update


def _synchronize():
    if torch.npu.is_available():
        torch.npu.synchronize()


def _make_scan_inputs(batch, dim, dstate, seqlen, device="npu", requires_grad=False):
    torch.manual_seed(42)
    u = torch.randn(batch, dim, seqlen, device=device, requires_grad=requires_grad)
    delta = torch.rand(batch, dim, seqlen, device=device) * 0.1 + 0.01
    delta.requires_grad_(requires_grad)
    A = torch.randn(dim, dstate, device=device) * 0.1
    A.requires_grad_(requires_grad)
    B = torch.randn(batch, dstate, seqlen, device=device)
    B.requires_grad_(requires_grad)
    C = torch.randn(batch, dstate, seqlen, device=device)
    C.requires_grad_(requires_grad)
    return u, delta, A, B, C


def benchmark_scan_forward(batch, dim, dstate, seqlen, BLOCK_L=32):
    """Benchmark forward selective_scan with layered timing."""
    device = "npu"
    u, delta, A, B, C = _make_scan_inputs(batch, dim, dstate, seqlen, device)

    # --- 1. Compile + first execution ---
    t0 = time.time()
    out, carry = selective_scan_fwd(u, delta, A, B, C, BLOCK_L=BLOCK_L)
    _synchronize()
    compile_exec_ms = (time.time() - t0) * 1000

    # --- 2. Cached execution (warm) ---
    t0 = time.time()
    out, carry = selective_scan_fwd(u, delta, A, B, C, BLOCK_L=BLOCK_L)
    _synchronize()
    cached_exec_ms = (time.time() - t0) * 1000

    # --- 3. Multiple iterations average ---
    n_iters = 10
    t0 = time.time()
    for _ in range(n_iters):
        out, carry = selective_scan_fwd(u, delta, A, B, C, BLOCK_L=BLOCK_L)
    _synchronize()
    avg_exec_ms = (time.time() - t0) * 1000 / n_iters

    return {
        "batch": batch, "dim": dim, "dstate": dstate, "seqlen": seqlen,
        "BLOCK_L": BLOCK_L,
        "compile_exec_ms": compile_exec_ms,
        "cached_exec_ms": cached_exec_ms,
        "avg_exec_ms": avg_exec_ms,
    }


def benchmark_scan_backward(batch, dim, dstate, seqlen, BLOCK_L=32):
    """Benchmark backward selective_scan with layered timing."""
    device = "npu"
    u, delta, A, B, C = _make_scan_inputs(batch, dim, dstate, seqlen, device, requires_grad=True)
    grad_y = torch.randn_like(u)
    out, carry = selective_scan_fwd(u, delta, A, B, C, BLOCK_L=BLOCK_L)

    # --- 1. Compile + first execution ---
    t0 = time.time()
    du, ddelta, dA, dB, dC, _ = selective_scan_bwd(
        u, delta, A, B, C, grad_y, carry, BLOCK_L=BLOCK_L
    )
    _synchronize()
    compile_exec_ms = (time.time() - t0) * 1000

    # --- 2. Cached execution ---
    t0 = time.time()
    du, ddelta, dA, dB, dC, _ = selective_scan_bwd(
        u, delta, A, B, C, grad_y, carry, BLOCK_L=BLOCK_L
    )
    _synchronize()
    cached_exec_ms = (time.time() - t0) * 1000

    # --- 3. Multiple iterations ---
    n_iters = 5
    t0 = time.time()
    for _ in range(n_iters):
        du, ddelta, dA, dB, dC, _ = selective_scan_bwd(
            u, delta, A, B, C, grad_y, carry, BLOCK_L=BLOCK_L
        )
    _synchronize()
    avg_exec_ms = (time.time() - t0) * 1000 / n_iters

    return {
        "batch": batch, "dim": dim, "dstate": dstate, "seqlen": seqlen,
        "BLOCK_L": BLOCK_L,
        "compile_exec_ms": compile_exec_ms,
        "cached_exec_ms": cached_exec_ms,
        "avg_exec_ms": avg_exec_ms,
    }


def benchmark_scan_end2end(batch, dim, dstate, seqlen, BLOCK_L=32):
    """Benchmark full autograd forward+backward."""
    device = "npu"
    u, delta, A, B, C = _make_scan_inputs(batch, dim, dstate, seqlen, device, requires_grad=True)
    D = torch.randn(dim, device=device, requires_grad=True)
    z = torch.randn(batch, dim, seqlen, device=device, requires_grad=True)
    delta_bias = torch.randn(dim, device=device, requires_grad=True) * 0.01

    # Warmup
    out = selective_scan_fn(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True)
    out.sum().backward()
    _synchronize()

    n_iters = 5
    t0 = time.time()
    for _ in range(n_iters):
        out = selective_scan_fn(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias, delta_softplus=True)
        loss = out.sum()
        loss.backward()
    _synchronize()
    avg_ms = (time.time() - t0) * 1000 / n_iters

    return {
        "batch": batch, "dim": dim, "dstate": dstate, "seqlen": seqlen,
        "avg_fwd_bwd_ms": avg_ms,
    }


def benchmark_state_update(batch, dim, dstate, n_iters=100):
    """Benchmark selective_state_update (single step)."""
    device = "npu"
    torch.manual_seed(42)
    state = torch.randn(batch, dim, dstate, device=device)
    x = torch.randn(batch, dim, device=device)
    dt = torch.rand(batch, dim, device=device) * 0.1 + 0.01
    A = torch.randn(dim, dstate, device=device) * 0.1
    B = torch.randn(batch, dstate, device=device)
    C = torch.randn(batch, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn(batch, dim, device=device)
    dt_bias = torch.randn(dim, device=device) * 0.01

    # Warmup
    for _ in range(2):
        _ = selective_state_update(
            state, x, dt, A, B, C,
            D=D, z=z, dt_bias=dt_bias, dt_softplus=True
        )
    _synchronize()

    t0 = time.time()
    for _ in range(n_iters):
        out = selective_state_update(
            state, x, dt, A, B, C,
            D=D, z=z, dt_bias=dt_bias, dt_softplus=True
        )
    _synchronize()
    avg_ms = (time.time() - t0) * 1000 / n_iters

    return {"batch": batch, "dim": dim, "dstate": dstate, "avg_ms": avg_ms}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mamba selective scan on NPU")
    parser.add_argument("--op", type=str, default="all", choices=["fwd", "bwd", "e2e", "state_update", "all"])
    parser.add_argument("--BLOCK_L", type=int, default=32)
    args = parser.parse_args()

    if not torch.npu.is_available():
        print("ERROR: NPU not available")
        return

    print("NPU Benchmark for Mamba selective scan")
    print(f"BLOCK_L = {args.BLOCK_L}")
    print("=" * 70)

    # Configs: small -> large
    configs = [
        (1, 4, 4, 8),
        (1, 16, 8, 32),
        (2, 32, 8, 64),
        (2, 64, 16, 128),
        (2, 128, 16, 256),
        (2, 256, 16, 512),
        (2, 512, 16, 1024),
        (4, 1024, 64, 2048),
    ]

    if args.op in ("fwd", "all"):
        print("\n--- Forward Selective Scan ---")
        print(f"{'B':>3} {'D':>4} {'N':>3} {'L':>5} | {'Compile+Exec':>12} {'Cached':>10} {'Avg(10)':>10}")
        for batch, dim, dstate, seqlen in configs:
            try:
                r = benchmark_scan_forward(batch, dim, dstate, seqlen, args.BLOCK_L)
                print(f"{r['batch']:>3} {r['dim']:>4} {r['dstate']:>3} {r['seqlen']:>5} | "
                      f"{r['compile_exec_ms']:>10.1f}ms {r['cached_exec_ms']:>8.1f}ms {r['avg_exec_ms']:>8.1f}ms")
            except Exception as e:
                print(f"{batch:>3} {dim:>4} {dstate:>3} {seqlen:>5} | FAILED: {e}")

    if args.op in ("bwd", "all"):
        print("\n--- Backward Selective Scan ---")
        print(f"{'B':>3} {'D':>4} {'N':>3} {'L':>5} | {'Compile+Exec':>12} {'Cached':>10} {'Avg(5)':>10}")
        for batch, dim, dstate, seqlen in configs:
            try:
                r = benchmark_scan_backward(batch, dim, dstate, seqlen, args.BLOCK_L)
                print(f"{r['batch']:>3} {r['dim']:>4} {r['dstate']:>3} {r['seqlen']:>5} | "
                      f"{r['compile_exec_ms']:>10.1f}ms {r['cached_exec_ms']:>8.1f}ms {r['avg_exec_ms']:>8.1f}ms")
            except Exception as e:
                print(f"{batch:>3} {dim:>4} {dstate:>3} {seqlen:>5} | FAILED: {e}")

    if args.op in ("e2e", "all"):
        print("\n--- End-to-End Forward+Backward (Autograd) ---")
        print(f"{'B':>3} {'D':>4} {'N':>3} {'L':>5} | {'Avg(5)':>10}")
        for batch, dim, dstate, seqlen in configs[:3]:  # e2e is slower, test fewer
            try:
                r = benchmark_scan_end2end(batch, dim, dstate, seqlen, args.BLOCK_L)
                print(f"{r['batch']:>3} {r['dim']:>4} {r['dstate']:>3} {r['seqlen']:>5} | "
                      f"{r['avg_fwd_bwd_ms']:>8.1f}ms")
            except Exception as e:
                print(f"{batch:>3} {dim:>4} {dstate:>3} {seqlen:>5} | FAILED: {e}")

    if args.op in ("state_update", "all"):
        print("\n--- Selective State Update (single step) ---")
        print(f"{'B':>3} {'D':>4} {'N':>3} | {'Avg/step':>10}")
        for batch, dim, dstate in [(1, 512, 16), (2, 512, 16), (4, 1024, 64)]:
            try:
                r = benchmark_state_update(batch, dim, dstate)
                print(f"{r['batch']:>3} {r['dim']:>4} {r['dstate']:>3} | {r['avg_ms']:>8.3f}ms")
            except Exception as e:
                print(f"{batch:>3} {dim:>4} {dstate:>3} | FAILED: {e}")


if __name__ == "__main__":
    main()
