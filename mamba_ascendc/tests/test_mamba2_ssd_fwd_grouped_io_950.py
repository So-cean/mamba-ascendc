#!/usr/bin/env python3
"""Precision gate for the direct grouped-X/Y 950PR forward slice."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import ascend_kernel  # noqa: F401
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    nrmse = float(
        (torch.linalg.vector_norm(error) /
         torch.linalg.vector_norm(expected).clamp_min(1.0e-12)).item()
    )
    cosine = float(F.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    ).item())
    return {
        "max_abs": float(error.abs().max().item()),
        "nrmse": nrmse,
        "cosine": cosine,
        "finite": bool(torch.isfinite(actual).all().item()),
        "passed": bool(
            torch.isfinite(actual).all().item()
            and nrmse <= 5.0e-3
            and cosine >= 0.999
            and torch.allclose(actual, expected, rtol=3.0e-2, atol=1.0e-2)
        ),
    }


def _run_case(batch: int, seqlen: int, heads: int, groups: int,
              seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    x = randn(batch, seqlen, heads, 64)
    dt = 0.01 + 0.1 * torch.rand(
        batch, seqlen, heads, generator=generator
    )
    a = -(0.1 + 0.4 * torch.rand(heads, generator=generator))
    b = 0.2 * randn(batch, seqlen, groups, 64)
    c = 0.2 * randn(batch, seqlen, groups, 64)
    d = randn(heads, 64)
    z = randn(batch, seqlen, heads, 64)
    bias = 0.1 * randn(heads)
    initial = 0.1 * randn(batch, heads, 64, 64)
    values = [item.npu() for item in (x, dt, a, b, c, d, z, bias, initial)]
    x_npu, dt_npu, a_npu, b_npu, c_npu, d_npu, z_npu, bias_npu, initial_npu = values

    with torch.no_grad():
        ref_out, ref_final = ssd_chunk_scan_ref(
            x_npu, dt_npu, a_npu, b_npu, c_npu, 64,
            D=d_npu, z=z_npu, dt_bias=bias_npu,
            dt_softplus=True, initial_states=initial_npu,
            return_final_state=True,
        )
        x_grouped, da, b_cube, c_cube = (
            torch.ops.mamba_ascend.mamba2_ssd_preprocess_grouped(
                x_npu, dt_npu, a_npu, b_npu, c_npu, bias_npu,
                64, True, 0.0, float("inf"),
            )
        )
        y_diag, chunk_states = (
            torch.ops.mamba_ascend.mamba2_ssd_chunk_mix_grouped(
                x_grouped, da, b_cube, c_cube
            )
        )
        states, final_np = (
            torch.ops.mamba_ascend.mamba2_ssd_state_passing_grouped(
                chunk_states, da,
                initial_npu.transpose(-1, -2).contiguous(), groups,
            )
        )
        y_grouped = torch.ops.mamba_ascend.mamba2_ssd_state_projection_grouped(
            states, c_cube
        )
        out = torch.ops.mamba_ascend.mamba2_ssd_state_vector_epilogue_grouped(
            y_grouped, da, y_diag, x_npu, d_npu, z_npu
        )
        final = final_np.transpose(-1, -2).contiguous()
        torch.npu.synchronize()

    result = {
        "shape": [batch, seqlen, heads, 64, 64, 64, groups],
        "x_grouped_shape": list(x_grouped.shape),
        "y_diag_grouped_shape": list(y_diag.shape),
        "out": _metrics(out, ref_out),
        "final": _metrics(final, ref_final),
        "deterministic": True,
    }
    with torch.no_grad():
        repeat = torch.ops.mamba_ascend.mamba2_ssd_chunk_mix_grouped(
            x_grouped, da, b_cube, c_cube
        )
        torch.npu.synchronize()
    result["deterministic"] = all(
        torch.equal(lhs, rhs) for lhs, rhs in zip((y_diag, chunk_states), repeat)
    )
    result["passed"] = bool(
        result["out"]["passed"] and result["final"]["passed"]
        and result["deterministic"]
    )
    print(json.dumps(result), flush=True)
    assert result["passed"], result
    return result


def test_mamba2_ssd_fwd_grouped_io_950() -> None:
    for index, case in enumerate(((1, 1024, 16, 4),
                                  (2, 512, 16, 4),
                                  (1, 2048, 8, 2))):
        _run_case(*case, seed=20260820 + index)


if __name__ == "__main__":
    test_mamba2_ssd_fwd_grouped_io_950()
