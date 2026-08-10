#!/usr/bin/env python3
"""Precision gate for direct grouped-X preprocess on Ascend 950PR."""

from __future__ import annotations

import json

import torch
import torch_npu  # noqa: F401

import ascend_kernel  # noqa: F401


def _to_grouped_x(x_head: torch.Tensor, groups: int) -> torch.Tensor:
    batch, heads, chunks, tokens, width = x_head.shape
    return (
        x_head.view(batch, groups, 4, chunks, tokens, width)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
    )


def _run_case(batch: int, chunks: int, groups: int, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    length = chunks * 64
    heads = groups * 4
    x = 0.1 * torch.randn(batch, length, heads, 64, generator=generator)
    dt = 0.1 * torch.randn(batch, length, heads, generator=generator)
    a = -0.1 * torch.rand(heads, generator=generator)
    b = 0.1 * torch.randn(batch, length, groups, 64, generator=generator)
    c = 0.1 * torch.randn(batch, length, groups, 64, generator=generator)
    bias = 0.1 * torch.randn(heads, generator=generator)
    values = [value.npu() for value in (x, dt, a, b, c, bias)]

    reference = torch.ops.mamba_ascend.mamba2_ssd_preprocess(
        *values[:5], values[5], 64, True, 0.0, 3.0
    )
    expected = (_to_grouped_x(reference[0], groups), *reference[1:])
    outputs = []
    for _ in range(3):
        actual = torch.ops.mamba_ascend.mamba2_ssd_preprocess_grouped(
            *values[:5], values[5], 64, True, 0.0, 3.0
        )
        torch.npu.synchronize()
        outputs.append(tuple(value.cpu() for value in actual))
    expected_cpu = tuple(value.cpu() for value in expected)

    metrics = []
    for actual, target in zip(outputs[0], expected_cpu):
        diff = (actual.float() - target.float()).abs()
        metrics.append({
            "shape": list(actual.shape),
            "max_abs": float(diff.max()),
            "equal": bool(torch.equal(actual, target)),
        })
    result = {
        "case": [batch, chunks, groups],
        "outputs": metrics,
        "deterministic": all(
            all(torch.equal(lhs, rhs) for lhs, rhs in zip(outputs[0], item))
            for item in outputs[1:]
        ),
    }
    print(json.dumps(result), flush=True)
    assert metrics[0]["shape"] == [batch, chunks, groups, 64, 4, 64]
    assert all(item["equal"] for item in metrics), result
    assert result["deterministic"], result
    return result


def test_mamba2_ssd_preprocess_grouped_950() -> None:
    for index, case in enumerate(((1, 1, 1), (1, 2, 2), (2, 3, 4))):
        _run_case(*case, seed=20260808 + index)


if __name__ == "__main__":
    test_mamba2_ssd_preprocess_grouped_950()
