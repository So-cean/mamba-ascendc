#!/usr/bin/env python3
"""End-to-end precision gate for the 950PR grouped forward pipeline."""

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
    denominator = torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
    nrmse = float((torch.linalg.vector_norm(error) / denominator).item())
    cosine = float(F.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    ).item())
    return {
        "max_abs": float(error.abs().max().item()),
        "mean_abs": float(error.abs().mean().item()),
        "nrmse": nrmse,
        "cosine": cosine,
        "allclose": bool(torch.allclose(
            actual, expected, rtol=3.0e-2, atol=1.0e-2
        )),
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
    chunk_size = 64
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

    cpu_values = (x, dt, a, b, c, d, z, dt_bias, initial)
    npu_values = tuple(value.npu() for value in cpu_values)
    x_npu, dt_npu, a_npu, b_npu, c_npu, d_npu, z_npu, bias_npu, initial_npu = (
        npu_values
    )
    with torch.no_grad():
        ref_out, ref_final = ssd_chunk_scan_ref(
            x_npu, dt_npu, a_npu, b_npu, c_npu, chunk_size,
            D=d_npu, z=z_npu, dt_bias=bias_npu,
            dt_softplus=True, initial_states=initial_npu,
            return_final_state=True,
        )
        x_cube, da, b_cube, c_cube = (
            torch.ops.mamba_ascend.mamba2_ssd_preprocess(
                x_npu, dt_npu, a_npu, b_npu, c_npu, bias_npu,
                chunk_size, True, 0.0, float("inf"),
            )
        )
        y_diag, chunk_states_grouped = (
            torch.ops.mamba_ascend.mamba2_ssd_chunk_mix_grouped(
                x_cube, da, b_cube, c_cube
            )
        )
        states_grouped, final_np = (
            torch.ops.mamba_ascend.mamba2_ssd_state_passing_grouped(
                chunk_states_grouped, da,
                initial_npu.transpose(-1, -2).contiguous(), groups,
            )
        )
        y_grouped = (
            torch.ops.mamba_ascend.mamba2_ssd_state_projection_grouped(
                states_grouped, c_cube
            )
        )
        out = (
            torch.ops.mamba_ascend
            .mamba2_ssd_state_vector_epilogue_grouped(
                y_grouped, da, y_diag, x_npu, d_npu, z_npu
            )
        )
        final = final_np.transpose(-1, -2).contiguous()
        public_out, public_final = ascend_kernel.mamba2_ssd_fwd(
            x_npu, dt_npu, a_npu, b_npu, c_npu, chunk_size,
            D=d_npu, z=z_npu, dt_bias=bias_npu,
            dt_softplus=True, initial_states=initial_npu,
            return_final_state=True,
        )
        torch.npu.synchronize()

    out_metrics = _metrics(out, ref_out)
    final_metrics = _metrics(final, ref_final)
    public_out_metrics = _metrics(public_out, ref_out)
    public_final_metrics = _metrics(public_final, ref_final)
    result = {
        "shape_b_l_h_p_n_c_g": [
            batch, seqlen, heads, 64, 64, chunk_size, groups
        ],
        "out": out_metrics,
        "final_state": final_metrics,
        "public_out": public_out_metrics,
        "public_final_state": public_final_metrics,
        "public_matches_grouped": bool(
            torch.equal(public_out, out) and torch.equal(public_final, final)
        ),
        "passed": bool(
            out_metrics["passed"]
            and final_metrics["passed"]
            and public_out_metrics["passed"]
            and public_final_metrics["passed"]
            and torch.equal(public_out, out)
            and torch.equal(public_final, final)
        ),
    }
    assert result["passed"], json.dumps(result)
    return result


def test_mamba2_ssd_fwd_grouped_pipeline_950() -> None:
    cases = (
        # ChunkMixGrouped is a heavy-path operator and intentionally requires
        # B*G*K to fill all Cube cores.  Each case has 64 group-chunk tasks.
        (1, 1024, 16, 4),
        (2, 512, 16, 4),
        (1, 2048, 8, 2),
    )
    results = [
        _run_case(*case, seed=20260808 + index)
        for index, case in enumerate(cases)
    ]
    print(json.dumps({"results": results}), flush=True)


if __name__ == "__main__":
    test_mamba2_ssd_fwd_grouped_pipeline_950()
