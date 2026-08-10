#!/usr/bin/env python3
"""Independent precision gate for the grouped Vector epilogue on 950PR."""

from __future__ import annotations

import json

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

import ascend_kernel  # noqa: F401


def _reference(
    y_grouped: torch.Tensor,
    da: torch.Tensor,
    y_diag: torch.Tensor,
    x: torch.Tensor,
    d: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Compute the public [B,L,H,P] result without another custom op."""
    batch, chunks, groups, chunk_size, heads_per_group, headdim = (
        y_grouped.shape
    )
    heads = groups * heads_per_group
    y_head = (
        y_grouped.permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(batch, heads, chunks, chunk_size, headdim)
    )
    x_head = (
        x.view(batch, chunks, chunk_size, heads, headdim)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    z_head = (
        z.view(batch, chunks, chunk_size, heads, headdim)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    pre_gate = (
        y_head * torch.exp(da.unsqueeze(-1))
        + y_diag
        + x_head * d.view(1, heads, 1, 1, headdim)
    )
    return (
        (pre_gate * F.silu(z_head))
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(batch, chunks * chunk_size, heads, headdim)
    )


def _run_case(batch: int, chunks: int, groups: int, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    heads_per_group = 4
    heads = groups * heads_per_group
    y_grouped = (0.05 * torch.randn(
        (batch, chunks, groups, 64, heads_per_group, 64),
        generator=generator,
    )).half()
    da = -0.05 * torch.rand(
        (batch, heads, chunks, 64), generator=generator
    )
    y_diag = 0.05 * torch.randn(
        (batch, heads, chunks, 64, 64), generator=generator
    )
    x = 0.05 * torch.randn(
        (batch, chunks * 64, heads, 64), generator=generator
    )
    d = 0.05 * torch.randn((heads, 64), generator=generator)
    z = 0.05 * torch.randn(x.shape, generator=generator)
    expected = _reference(y_grouped, da, y_diag, x, d, z)
    tensors = [value.npu() for value in (y_grouped, da, y_diag, x, d, z)]
    yg, da_npu, yd, x_npu, d_npu, z_npu = tensors
    outputs = []
    for _ in range(3):
        output = (
            torch.ops.mamba_ascend
            .mamba2_ssd_state_vector_epilogue_grouped(
                yg, da_npu, yd, x_npu, d_npu, z_npu
            )
        )
        torch.npu.synchronize()
        outputs.append(output.cpu())
    diff = (outputs[0] - expected).abs()
    reference_norm = torch.linalg.vector_norm(expected.float())
    result = {
        "case": [batch, chunks, groups, heads_per_group],
        "shape": list(outputs[0].shape),
        "max_abs": float(diff.max().item()),
        "nrmse": float(
            torch.linalg.vector_norm(diff.float()).item()
            / max(reference_norm.item(), 1.0e-12)
        ),
        "deterministic": all(
            torch.equal(outputs[0], value) for value in outputs[1:]
        ),
        "allclose": bool(torch.allclose(
            outputs[0], expected, rtol=2.0e-5, atol=2.0e-6
        )),
    }
    print(json.dumps(result), flush=True)
    assert result["shape"] == [batch, chunks * 64, heads, 64]
    assert result["deterministic"]
    assert result["allclose"]
    return result


def test_mamba2_ssd_state_vector_epilogue_grouped() -> None:
    cases = [
        (1, 1, 1),
        (1, 2, 2),
        (1, 4, 2),
        (1, 4, 4),
        (2, 1, 2),
        (2, 3, 4),
    ]
    results = [
        _run_case(batch, chunks, groups, 20260807 + index)
        for index, (batch, chunks, groups) in enumerate(cases)
    ]
    print(json.dumps({"results": results}), flush=True)


if __name__ == "__main__":
    test_mamba2_ssd_state_vector_epilogue_grouped()
