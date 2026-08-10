#!/usr/bin/env python3
"""Exercise the fused gate and DtBwd-D kernels under mssanitizer."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def _load_local_extension() -> None:
    package_spec = importlib.util.find_spec("ascend_kernel")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError("ascend_kernel is not installed in this environment")
    package_root = Path(next(iter(package_spec.submodule_search_locations)))
    opp_root = Path(
        os.environ.get("MAMBA2_TEST_OPP_ROOT", package_root / "opp")
    )
    vendor_candidates = (
        opp_root / "vendors" / "customize",
        opp_root / "opp" / "vendors" / "customize",
    )
    vendor = next((path for path in vendor_candidates if path.is_dir()), None)
    if vendor is None:
        raise RuntimeError(f"Cannot find custom OPP under {opp_root}")
    op_api = vendor / "op_api" / "lib" / "libcust_opapi.so"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    os.environ["MAMBA_CHUNK_MIX_OP_API_LIB"] = str(op_api)
    ctypes.CDLL(str(op_api), mode=ctypes.RTLD_GLOBAL)
    extension = Path(
        os.environ.get(
            "MAMBA2_TEST_EXTENSION_LIB",
            package_root / "lib" / "libascend_kernel.so",
        )
    )
    torch.ops.load_library(str(extension))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--large-gate", action="store_true")
    parser.add_argument("--sanitizer-only", action="store_true")
    parser.add_argument("--heads", type=int)
    args = parser.parse_args()
    _load_local_extension()

    torch.manual_seed(20260807)
    batch, seqlen, heads, headdim = (
        (4, 8192, 1, 64) if args.large_gate else (2, 128, 4, 64)
    )
    if args.heads is not None:
        if args.heads <= 0:
            parser.error("--heads must be positive")
        heads = args.heads
    chunks = seqlen // 64
    shape = (batch, seqlen, heads, headdim)
    dout = torch.randn(shape, device="npu")
    z = torch.randn(shape, device="npu")
    y_pre = torch.randn(shape, device="npu")
    x = torch.randn(shape, device="npu")
    gy, dz, d_d = torch.ops.mamba_ascend.mamba2_ssd_bwd_gate(
        dout, z, y_pre, x
    )

    if args.gate_only:
        torch.npu.synchronize()
        if args.sanitizer_only:
            # Kernel-filtered initcheck does not track later framework ops or
            # an unfiltered companion reduction as producers.  Do not read
            # filtered outputs on the host; correctness is covered by the
            # separate large-shape pytest regression.
            print(json.dumps({"shape": [batch, seqlen, heads, headdim]}))
            return 0
        tensors = (gy, dz, d_d)
        finite = all(torch.isfinite(value).all().item() for value in tensors)
        result = {
            "shape": [batch, seqlen, heads, headdim],
            "finite": finite,
            "checksums": [value.float().sum().item() for value in tensors],
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return 0 if finite else 1

    d_xdt = torch.randn(
        batch, heads, chunks, 64, headdim, device="npu"
    )
    g_cs = torch.randn(batch, heads, chunks, 64, device="npu")
    dt = 0.01 + 0.1 * torch.rand(batch, seqlen, heads, device="npu")
    a = -0.1 - 0.4 * torch.rand(heads, device="npu")
    d = torch.randn(heads, headdim, device="npu")
    dt_bias = 0.1 * torch.randn(heads, device="npu")
    dx, ddt, d_a, d_dt_bias = (
        torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_d(
            x,
            d_xdt,
            g_cs,
            dt,
            a,
            gy,
            d,
            dt_bias,
            True,
            1.0e-3,
            1.0,
        )
    )
    torch.npu.synchronize()
    tensors = (gy, dz, d_d, dx, ddt, d_a, d_dt_bias)
    finite = all(torch.isfinite(value).all().item() for value in tensors)
    result = {
        "finite": finite,
        "checksums": [value.float().sum().item() for value in tensors],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
