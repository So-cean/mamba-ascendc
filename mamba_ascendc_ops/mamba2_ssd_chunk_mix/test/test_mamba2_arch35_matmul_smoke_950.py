#!/usr/bin/env python3
"""Internal-only arch3510 Cube Matmul precision gate."""

from __future__ import annotations

import json

import torch
import torch_npu  # noqa: F401

import ascend_kernel  # noqa: F401


def _distinct_group() -> torch.Tensor:
    """Values make token/head/column layout mistakes independently visible."""
    token = torch.arange(64, dtype=torch.float32)[:, None, None]
    head = torch.arange(4, dtype=torch.float32)[None, :, None]
    column = torch.arange(64, dtype=torch.float32)[None, None, :]
    return (token * 0.25 + head * 2.0 + column * 0.03125).half()


def _inputs(pattern: str) -> tuple[torch.Tensor, torch.Tensor]:
    if pattern == "zero":
        return (
            torch.zeros((64, 64), dtype=torch.float16),
            torch.zeros((64, 4, 64), dtype=torch.float16),
        )
    if pattern == "identity":
        return torch.eye(64, dtype=torch.float16), _distinct_group()
    if pattern == "one_hot":
        weight = torch.zeros((64, 64), dtype=torch.float16)
        weight[7, 13] = 1.0
        return weight, _distinct_group()
    if pattern == "random":
        generator = torch.Generator(device="cpu").manual_seed(20260807)
        return (
            (0.125 * torch.randn((64, 64), generator=generator)).half(),
            (0.125 * torch.randn((64, 4, 64), generator=generator)).half(),
        )
    raise ValueError(pattern)


def _run(pattern: str) -> dict:
    print(json.dumps({"pattern": pattern, "stage": "prepare_cpu"}), flush=True)
    weight, x_group = _inputs(pattern)
    # The CPU float32 product is the independent precision reference. Inputs
    # are quantized to fp16 before both reference and device computation.
    expected = torch.einsum(
        "mk,krp->mrp", weight.float(), x_group.float()
    )
    print(json.dumps({"pattern": pattern, "stage": "copy_to_npu"}), flush=True)
    weight_npu = weight.npu()
    x_group_npu = x_group.npu()
    torch.npu.synchronize()
    outputs = []
    for repeat in range(3):
        print(json.dumps({
            "pattern": pattern, "stage": "launch", "repeat": repeat
        }), flush=True)
        output = (
            torch.ops.mamba_ascend
            .mamba2_arch35_matmul_smoke_internal_test_only(
                weight_npu, x_group_npu
            )
        )
        torch.npu.synchronize()
        print(json.dumps({
            "pattern": pattern, "stage": "launch_done", "repeat": repeat
        }), flush=True)
        outputs.append(output.cpu())

    actual = outputs[0].float()
    finite = bool(torch.isfinite(actual).all().item())
    layout_ok = actual.shape == (64, 4, 64) and actual.stride() == (256, 64, 1)
    diff = (actual - expected).abs()
    denom = expected.abs().clamp_min(1.0e-6)
    result = {
        "pattern": pattern,
        "finite": finite,
        "layout_ok": layout_ok,
        "deterministic": all(torch.equal(outputs[0], x) for x in outputs[1:]),
        "max_abs": float(diff.max().item()),
        "max_rel": float((diff / denom).max().item()),
        "allclose": bool(torch.allclose(actual, expected, rtol=5.0e-3, atol=5.0e-3)),
    }
    print(json.dumps(result), flush=True)
    return result


def test_mamba2_arch35_matmul_smoke() -> None:
    for pattern in ("zero", "identity", "one_hot", "random"):
        result = _run(pattern)
        assert result["finite"]
        assert result["layout_ok"]
        assert result["deterministic"]
        assert result["allclose"]


def test_mamba2_arch35_batch_matmul_smoke() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260809)
    weight = (0.125 * torch.randn(
        (4, 64, 64), generator=generator
    )).half()
    x_group = (0.125 * torch.randn(
        (4, 64, 64), generator=generator
    )).half()
    expected = torch.bmm(weight.float(), x_group.float())
    weight_npu = weight.npu()
    x_group_npu = x_group.npu()
    outputs = []
    for _ in range(3):
        output = (
            torch.ops.mamba_ascend
            .mamba2_arch35_matmul_smoke_internal_test_only(
                weight_npu, x_group_npu
            )
        )
        torch.npu.synchronize()
        outputs.append(output.cpu())
    actual = outputs[0].float()
    result = {
        "pattern": "batch4_random",
        "shape": list(actual.shape),
        "finite": bool(torch.isfinite(actual).all().item()),
        "deterministic": all(
            torch.equal(outputs[0], value) for value in outputs[1:]
        ),
        "max_abs": float((actual - expected).abs().max().item()),
        "allclose": bool(torch.allclose(
            actual, expected, rtol=5.0e-3, atol=5.0e-3
        )),
    }
    print(json.dumps(result), flush=True)
    assert result["finite"]
    assert result["deterministic"]
    assert result["allclose"]


if __name__ == "__main__":
    test_mamba2_arch35_matmul_smoke()
    test_mamba2_arch35_batch_matmul_smoke()
