#!/usr/bin/env python3
"""Precision gate for the 950PR grouped training Vector epilogue."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import ascend_kernel  # noqa: F401


def _reference(y_grouped, da, y_diag, x, d, z):
    batch, chunks, groups, chunk_size, heads_per_group, headdim = (
        y_grouped.shape
    )
    heads = groups * heads_per_group
    y_head = y_grouped.permute(0, 2, 4, 1, 3, 5).reshape(
        batch, heads, chunks, chunk_size, headdim
    )
    x_head = x.reshape(
        batch, chunks, chunk_size, heads, headdim
    ).permute(0, 3, 1, 2, 4)
    pre_head = (
        y_head.float() * torch.exp(da.unsqueeze(-1))
        + y_diag
        + x_head * d.view(1, heads, 1, 1, headdim)
    )
    pre_gate = pre_head.permute(0, 2, 3, 1, 4).reshape_as(x)
    return pre_gate * F.silu(z), pre_gate


def _run_case(batch: int, chunks: int, groups: int, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    heads_per_group = 4
    heads = groups * heads_per_group
    y_grouped = (0.05 * torch.randn(
        batch, chunks, groups, 64, heads_per_group, 64,
        generator=generator,
    )).half()
    da = -0.05 * torch.rand(
        batch, heads, chunks, 64, generator=generator
    )
    y_diag = 0.05 * torch.randn(
        batch, heads, chunks, 64, 64, generator=generator
    )
    x = 0.05 * torch.randn(
        batch, chunks * 64, heads, 64, generator=generator
    )
    d = 0.05 * torch.randn(heads, 64, generator=generator)
    z = 0.05 * torch.randn(x.shape, generator=generator)
    expected_out, expected_pre = _reference(
        y_grouped, da, y_diag, x, d, z
    )
    values = [value.npu() for value in (y_grouped, da, y_diag, x, d, z)]
    outputs = []
    for _ in range(3):
        out, pre_gate = (
            torch.ops.mamba_ascend
            .mamba2_ssd_state_vector_epilogue_grouped_train(*values)
        )
        torch.npu.synchronize()
        outputs.append((out.cpu(), pre_gate.cpu()))
    out_diff = (outputs[0][0] - expected_out).abs()
    pre_diff = (outputs[0][1].float() - expected_pre.float()).abs()
    result = {
        "case": [batch, chunks, groups, heads_per_group],
        "out_max_abs": float(out_diff.max()),
        "pre_gate_max_abs": float(pre_diff.max()),
        "pre_gate_dtype": str(outputs[0][1].dtype),
        "out_allclose": bool(torch.allclose(
            outputs[0][0], expected_out, rtol=2.0e-5, atol=2.0e-6
        )),
        "pre_gate_allclose": bool(torch.allclose(
            outputs[0][1].float(), expected_pre.float(),
            rtol=5.0e-3, atol=5.0e-4
        )),
        "deterministic": all(
            torch.equal(outputs[0][0], value[0])
            and torch.equal(outputs[0][1], value[1])
            for value in outputs[1:]
        ),
    }
    assert result["out_allclose"], result
    assert result["pre_gate_allclose"], result
    assert result["pre_gate_dtype"] == "torch.float16", result
    assert result["deterministic"], result
    return result


def test_mamba2_ssd_state_vector_epilogue_grouped_train_950() -> None:
    cases = ((1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 3, 2))
    results = [
        _run_case(*case, seed=20260808 + index)
        for index, case in enumerate(cases)
    ]
    print(json.dumps({"results": results}), flush=True)


if __name__ == "__main__":
    test_mamba2_ssd_state_vector_epilogue_grouped_train_950()
