#!/usr/bin/env python3
"""Dual-platform precision gate for the public grouped forward pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from mamba_torch.ssd_reference import ssd_chunk_scan_ref


os.environ.setdefault("MAMBA_ASCENDC_CHUNK_MIX", "1")
if os.environ.get("MAMBA2_TEST_EXTENSION_LIB"):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "benchmarks"))
    from mamba2_backward_bench import load_ascend_backend

    ascend_kernel = load_ascend_backend()
else:
    import ascend_kernel


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    error = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-12)
    nrmse = float((torch.linalg.vector_norm(error) / denominator).item())
    cosine = float(F.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item())
    return {
        "max_abs": float(error.abs().max().item()),
        "nrmse": nrmse,
        "cosine": cosine,
        "allclose": bool(torch.allclose(
            actual.float(), expected.float(), rtol=3.0e-2, atol=1.0e-2
        )),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def _run_case(batch: int, seqlen: int, heads: int, groups: int,
              seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    x = randn(batch, seqlen, heads, 64)
    dt = 0.01 + 0.1 * torch.rand(
        batch, seqlen, heads, generator=generator
    )
    a = -(0.1 + 0.4 * torch.rand(heads, generator=generator))
    b = 0.2 * randn(batch, seqlen, groups, 64)
    c = 0.2 * randn(batch, seqlen, groups, 64)
    d = randn(heads, 64)
    z = randn(batch, seqlen, heads, 64)
    dt_bias = 0.1 * randn(heads)
    initial = 0.1 * randn(batch, heads, 64, 64)
    values = tuple(
        value.npu() for value in (x, dt, a, b, c, d, z, dt_bias, initial)
    )
    x_npu, dt_npu, a_npu, b_npu, c_npu, d_npu, z_npu, bias_npu, init_npu = values

    with torch.no_grad():
        expected_out, expected_final = ssd_chunk_scan_ref(
            x_npu, dt_npu, a_npu, b_npu, c_npu, 64,
            D=d_npu, z=z_npu, dt_bias=bias_npu,
            dt_softplus=True, initial_states=init_npu,
            return_final_state=True,
        )
        outputs = [
            ascend_kernel.mamba2_ssd_fwd(
                x_npu, dt_npu, a_npu, b_npu, c_npu, 64,
                D=d_npu, z=z_npu, dt_bias=bias_npu,
                dt_softplus=True, initial_states=init_npu,
                return_final_state=True,
            )
            for _ in range(3)
        ]
        torch.npu.synchronize()

    out_metrics = _metrics(outputs[0][0], expected_out)
    final_metrics = _metrics(outputs[0][1], expected_final)
    deterministic = all(
        torch.equal(outputs[0][0], value[0])
        and torch.equal(outputs[0][1], value[1])
        for value in outputs[1:]
    )
    passed = bool(
        out_metrics["finite"] and out_metrics["allclose"]
        and out_metrics["nrmse"] <= 5.0e-3
        and out_metrics["cosine"] >= 0.999
        and final_metrics["finite"] and final_metrics["allclose"]
        and final_metrics["nrmse"] <= 5.0e-3
        and final_metrics["cosine"] >= 0.999
        and deterministic
    )
    result = {
        "shape_b_l_h_p_n_c_g": [
            batch, seqlen, heads, 64, 64, 64, groups
        ],
        "out": out_metrics,
        "final_state": final_metrics,
        "deterministic": deterministic,
        "passed": passed,
    }
    assert passed, json.dumps(result)
    return result


def test_mamba2_ssd_fwd_grouped_pipeline_dual() -> None:
    cases = (
        (1, 1024, 16, 4),
        (2, 512, 16, 4),
        (1, 2048, 8, 2),
        # Exactly 20 group tasks on 910B3: this prevents the occupancy guard
        # from silently falling back to the 2-head path and exercises the
        # 64x64 @ 64x256 four-head projection candidate.
        (1, 128, 80, 20),
    )
    results = [
        _run_case(*case, seed=20260808 + index)
        for index, case in enumerate(cases)
    ]
    print(json.dumps({
        "device": torch.npu.get_device_name(0),
        "results": results,
    }), flush=True)


if __name__ == "__main__":
    test_mamba2_ssd_fwd_grouped_pipeline_dual()
