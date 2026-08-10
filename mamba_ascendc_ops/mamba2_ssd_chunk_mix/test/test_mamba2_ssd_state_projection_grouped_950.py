#!/usr/bin/env python3
"""Precision and repeated-workspace gate for grouped 950PR projection."""

from __future__ import annotations

import json

import torch
import torch_npu  # noqa: F401

import ascend_kernel  # noqa: F401


def _inputs(pattern: str) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, 8, 4, 64, 4, 64)
    if pattern == "identity":
        token = torch.arange(64, dtype=torch.float32)[None, None, None, :, None, None]
        head = torch.arange(4, dtype=torch.float32)[None, None, None, None, :, None]
        column = torch.arange(64, dtype=torch.float32)[None, None, None, None, None, :]
        chunk = torch.arange(8, dtype=torch.float32)[None, :, None, None, None, None]
        group = torch.arange(4, dtype=torch.float32)[None, None, :, None, None, None]
        states = (0.03125 * token + 0.25 * head + 0.0078125 * column +
                  0.5 * chunk + group).expand(shape).half().contiguous()
        c_cube = torch.eye(64, dtype=torch.float16).view(1, 1, 1, 64, 64)
        c_cube = c_cube.expand(1, 8, 4, 64, 64).contiguous()
        return states, c_cube
    if pattern == "random":
        generator = torch.Generator(device="cpu").manual_seed(20260807)
        states = (0.125 * torch.randn(shape, generator=generator)).half()
        c_cube = (0.125 * torch.randn(
            (1, 8, 4, 64, 64), generator=generator
        )).half()
        return states, c_cube
    raise ValueError(pattern)


def test_mamba2_ssd_state_projection_grouped() -> None:
    for pattern in ("identity", "random"):
        states, c_cube = _inputs(pattern)
        expected = torch.einsum(
            "bkgtn,bkgnrp->bkgtrp", c_cube.float(), states.float()
        )
        states_npu = states.npu()
        c_npu = c_cube.npu()
        outputs = []
        for _ in range(3):
            output = torch.ops.mamba_ascend.mamba2_ssd_state_projection_grouped(
                states_npu, c_npu
            )
            torch.npu.synchronize()
            outputs.append(output.cpu())

        actual = outputs[0].float()
        diff = (actual - expected).abs()
        result = {
            "pattern": pattern,
            "shape": list(actual.shape),
            "stride": list(actual.stride()),
            "finite": bool(torch.isfinite(actual).all().item()),
            "deterministic": all(
                torch.equal(outputs[0], value) for value in outputs[1:]
            ),
            "max_abs": float(diff.max().item()),
            "nrmse": float(
                torch.sqrt(torch.mean(diff.square())).item()
                / expected.square().mean().sqrt().clamp_min(1.0e-12).item()
            ),
            "allclose": bool(torch.allclose(
                actual, expected, rtol=5.0e-3, atol=5.0e-3
            )),
        }
        print(json.dumps(result), flush=True)
        assert result["shape"] == [1, 8, 4, 64, 4, 64]
        assert result["stride"] == [524288, 65536, 16384, 256, 64, 1]
        assert result["finite"]
        assert result["deterministic"]
        assert result["allclose"]


if __name__ == "__main__":
    test_mamba2_ssd_state_projection_grouped()
