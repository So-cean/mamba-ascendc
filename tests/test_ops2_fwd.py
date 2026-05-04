# Copyright (c) 2024, mamba-triton-ascend authors.
# Test forward selective scan (ops2) against reference with timing.

import time
import torch

from mamba_triton_ascend.ops.reference import selective_scan_ref
from mamba_triton_ascend.ops2.selective_scan import selective_scan_fn
from mamba_triton_ascend.ops2.triton_kernels.selective_scan_fwd import selective_scan_fwd


def _synchronize():
    if torch.npu.is_available():
        torch.npu.synchronize()


def test_fwd_small():
    """Small config correctness test."""
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 1, 4, 4, 8
    device = 'npu'

    u = torch.randn(batch, dim, seqlen, device=device)
    delta = torch.rand(batch, dim, seqlen, device=device) * 0.1 + 0.01
    A = torch.randn(dim, dstate, device=device) * 0.1
    B = torch.randn(batch, dstate, seqlen, device=device)
    C = torch.randn(batch, dstate, seqlen, device=device)

    out_ref = selective_scan_ref(
        u.cpu(), delta.cpu(), A.cpu(), B.cpu(), C.cpu(),
        delta_softplus=True
    ).to(device)

    out_tri = selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
    diff = (out_ref - out_tri).abs().max().item()
    print(f"[small] forward max diff: {diff:.2e}")
    assert diff < 1e-3, f"Forward diff too large: {diff}"
    print("[small] PASSED")


def test_fwd_medium():
    """Medium config correctness test."""
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 32, 8, 64
    device = 'npu'

    u = torch.randn(batch, dim, seqlen, device=device)
    delta = torch.rand(batch, dim, seqlen, device=device) * 0.1 + 0.01
    A = torch.randn(dim, dstate, device=device) * 0.1
    B = torch.randn(batch, dstate, seqlen, device=device)
    C = torch.randn(batch, dstate, seqlen, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn(batch, dim, seqlen, device=device)
    delta_bias = torch.randn(dim, device=device) * 0.01

    out_ref = selective_scan_ref(
        u.cpu(), delta.cpu(), A.cpu(), B.cpu(), C.cpu(),
        D=D.cpu(), z=z.cpu(), delta_bias=delta_bias.cpu(), delta_softplus=True
    ).to(device)

    out_tri = selective_scan_fn(
        u, delta, A, B, C,
        D=D, z=z, delta_bias=delta_bias, delta_softplus=True
    )
    
    abs_diff = (out_ref - out_tri).abs()
    rel_diff = (abs_diff / (out_ref.abs() + 1e-6)).max().item()
    max_abs = abs_diff.max().item()
    print(f"[medium] forward max abs diff: {max_abs:.2e}, max rel diff: {rel_diff:.2e}")
    assert rel_diff < 1e-2, f"Forward rel diff too large: {rel_diff}"
    print("[medium] PASSED")


def test_fwd_large():
    """Large config correctness test."""
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 2, 128, 16, 256
    device = 'npu'

    u = torch.randn(batch, dim, seqlen, device=device)
    delta = torch.rand(batch, dim, seqlen, device=device) * 0.1 + 0.01
    A = torch.randn(dim, dstate, device=device) * 0.1
    B = torch.randn(batch, dstate, seqlen, device=device)
    C = torch.randn(batch, dstate, seqlen, device=device)

    out_ref = selective_scan_ref(
        u.cpu(), delta.cpu(), A.cpu(), B.cpu(), C.cpu(),
        delta_softplus=True
    ).to(device)

    out_tri = selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
    
    abs_diff = (out_ref - out_tri).abs()
    rel_diff = (abs_diff / (out_ref.abs() + 1e-6)).max().item()
    max_abs = abs_diff.max().item()
    print(f"[large] forward max abs diff: {max_abs:.2e}, max rel diff: {rel_diff:.2e}")
    assert rel_diff < 1e-2, f"Forward rel diff too large: {rel_diff}"
    print("[large] PASSED")


def test_fwd_xlarge():
    """Extra large config smoke test (no reference comparison due to fp overflow)."""
    torch.manual_seed(42)
    batch, dim, dstate, seqlen = 4, 512, 32, 512
    device = 'npu'

    # Use smaller A and delta to prevent numerical overflow in long sequences
    u = torch.randn(batch, dim, seqlen, device=device)
    delta = torch.rand(batch, dim, seqlen, device=device) * 0.05 + 0.01
    A = torch.randn(dim, dstate, device=device) * 0.05
    B = torch.randn(batch, dstate, seqlen, device=device) * 0.5
    C = torch.randn(batch, dstate, seqlen, device=device) * 0.5
    D = torch.randn(dim, device=device) * 0.5

    out_tri = selective_scan_fn(u, delta, A, B, C, D=D, delta_softplus=True)
    
    # For very large configs, only check Triton output validity (no NaN/Inf)
    has_nan = torch.isnan(out_tri).any().item()
    has_inf = torch.isinf(out_tri).any().item()
    max_val = out_tri.abs().max().item()
    
    print(f"[xlarge] output max abs value: {max_val:.2e}, has NaN: {has_nan}, has Inf: {has_inf}")
    assert not has_nan, "Triton output has NaN"
    assert not has_inf, "Triton output has Inf"
    # Note: very large values expected for long sequences due to exp accumulation; 
    # this is a smoke test to ensure kernel doesn't crash
    print("[xlarge] PASSED (smoke test)")


def benchmark_fwd(configs):
    """Benchmark ops2 forward kernel with timing breakdown."""
    device = 'npu'
    print("\n" + "="*80)
    print("ops2 Forward Benchmark (with autograd wrapper)")
    print(f"{'Batch':>6} {'Dim':>6} {'Dstate':>6} {'SeqLen':>8} | {'Compile+1st':>12} {'Cached':>10} {'Avg(10)':>10}")
    print("-"*80)

    for batch, dim, dstate, seqlen in configs:
        torch.manual_seed(42)
        u = torch.randn(batch, dim, seqlen, device=device)
        delta = torch.rand(batch, dim, seqlen, device=device) * 0.1 + 0.01
        A = torch.randn(dim, dstate, device=device) * 0.1
        B = torch.randn(batch, dstate, seqlen, device=device)
        C = torch.randn(batch, dstate, seqlen, device=device)

        # --- 1. Compile + first execution ---
        _synchronize()
        t0 = time.time()
        _ = selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
        _synchronize()
        compile_exec_ms = (time.time() - t0) * 1000

        # --- 2. Cached execution ---
        _synchronize()
        t0 = time.time()
        _ = selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
        _synchronize()
        cached_exec_ms = (time.time() - t0) * 1000

        # --- 3. Multiple iterations average ---
        n_iters = 10
        _synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _ = selective_scan_fn(u, delta, A, B, C, delta_softplus=True)
        _synchronize()
        avg_exec_ms = (time.time() - t0) * 1000 / n_iters

        print(f"{batch:>6} {dim:>6} {dstate:>6} {seqlen:>8} | "
              f"{compile_exec_ms:>10.1f}ms {cached_exec_ms:>8.1f}ms {avg_exec_ms:>8.1f}ms")


def benchmark_raw_kernel(configs):
    """Benchmark raw kernel (without autograd wrapper overhead)."""
    device = 'npu'
    print("\n" + "="*80)
    print("ops2 Raw Kernel Benchmark (no autograd wrapper)")
    print(f"{'Batch':>6} {'Dim':>6} {'Dstate':>6} {'SeqLen':>8} | {'Compile+1st':>12} {'Cached':>10} {'Avg(10)':>10}")
    print("-"*80)

    for batch, dim, dstate, seqlen in configs:
        torch.manual_seed(42)
        # Kernel expects (batch, T, D)
        u = torch.randn(batch, seqlen, dim, device=device)
        delta = torch.rand(batch, seqlen, dim, device=device) * 0.1 + 0.01
        A = torch.randn(dim, dstate, device=device) * 0.1
        B = torch.randn(batch, seqlen, dstate, device=device)
        C = torch.randn(batch, seqlen, dstate, device=device)

        # --- 1. Compile + first execution ---
        _synchronize()
        t0 = time.time()
        _ = selective_scan_fwd(u, delta, A, B, C, delta_softplus=True)
        _synchronize()
        compile_exec_ms = (time.time() - t0) * 1000

        # --- 2. Cached execution ---
        _synchronize()
        t0 = time.time()
        _ = selective_scan_fwd(u, delta, A, B, C, delta_softplus=True)
        _synchronize()
        cached_exec_ms = (time.time() - t0) * 1000

        # --- 3. Multiple iterations average ---
        n_iters = 10
        _synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _ = selective_scan_fwd(u, delta, A, B, C, delta_softplus=True)
        _synchronize()
        avg_exec_ms = (time.time() - t0) * 1000 / n_iters

        print(f"{batch:>6} {dim:>6} {dstate:>6} {seqlen:>8} | "
              f"{compile_exec_ms:>10.1f}ms {cached_exec_ms:>8.1f}ms {avg_exec_ms:>8.1f}ms")


if __name__ == "__main__":
    # Correctness tests
    test_fwd_small()
    test_fwd_medium()
    test_fwd_large()
    test_fwd_xlarge()
    print("\nAll ops2 forward correctness tests passed!")

    # Benchmark configs: small -> very large
    benchmark_configs = [
        (1, 4, 4, 8),
        (1, 16, 8, 32),
        (2, 32, 8, 64),
        (2, 64, 16, 128),
        (2, 128, 16, 256),
        (2, 256, 16, 512),
        (2, 512, 16, 1024),
        (4, 1024, 32, 2048),
    ]

    benchmark_fwd(benchmark_configs)
    benchmark_raw_kernel(benchmark_configs)
    print("\nAll ops2 benchmarks completed!")
