#!/usr/bin/env python3
"""Precision gate for direct group-owned ChunkMix state output."""

from __future__ import annotations

import json

import torch
import torch_npu  # noqa: F401

import ascend_kernel  # noqa: F401


def test_mamba2_ssd_chunk_mix_grouped() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    batch, heads, chunks, groups = 1, 16, 8, 4
    x = (0.05 * torch.randn(
        (batch, heads, chunks, 64, 64), generator=generator
    )).half()
    increments = -0.01 * torch.rand(
        (batch, heads, chunks, 64), generator=generator
    )
    da = torch.cumsum(increments, dim=-1)
    b = (0.05 * torch.randn(
        (batch, chunks, groups, 64, 64), generator=generator
    )).half()
    c = (0.05 * torch.randn(
        (batch, chunks, groups, 64, 64), generator=generator
    )).half()
    x_npu, da_npu, b_npu, c_npu = [value.npu() for value in (x, da, b, c)]
    expected_y = torch.empty((batch, heads, chunks, 64, 64))
    expected_group = torch.empty((batch, chunks, groups, 64, 4, 64))
    for batch_index in range(batch):
        for chunk_index in range(chunks):
            for group_index in range(groups):
                cb = torch.matmul(
                    c[batch_index, chunk_index, group_index].float(),
                    b[batch_index, chunk_index, group_index].float(),
                )
                for head_offset in range(4):
                    head_index = group_index * 4 + head_offset
                    local_da = da[batch_index, head_index, chunk_index]
                    decay_matrix = torch.exp(
                        local_da[:, None] - local_da[None, :]
                    ) * torch.tril(torch.ones((64, 64)))
                    weights = (cb * decay_matrix).half()
                    expected_y[
                        batch_index, head_index, chunk_index
                    ] = torch.matmul(
                        weights.float(),
                        x[batch_index, head_index, chunk_index].float(),
                    )
                    decay = torch.exp(
                        local_da[-1] - local_da
                    ).half()
                    weighted_x = (
                        x[batch_index, head_index, chunk_index]
                        * decay[:, None]
                    ).half()
                    expected_group[
                        batch_index, chunk_index, group_index, :,
                        head_offset, :
                    ] = torch.matmul(
                        b[batch_index, chunk_index, group_index].float(),
                        weighted_x.float(),
                    )
    outputs = []
    for _ in range(3):
        y_diag, states_grouped = (
            torch.ops.mamba_ascend.mamba2_ssd_chunk_mix_grouped(
                x_npu, da_npu, b_npu, c_npu
            )
        )
        torch.npu.synchronize()
        outputs.append((y_diag.cpu(), states_grouped.cpu()))
    expected_y_cpu = expected_y
    expected_group_cpu = expected_group
    actual_y = outputs[0][0]
    y_diff = (actual_y - expected_y_cpu).abs()
    state_diff = (outputs[0][1] - expected_group_cpu).abs()
    result = {
        "y_shape": list(outputs[0][0].shape),
        "state_shape": list(outputs[0][1].shape),
        "state_stride": list(outputs[0][1].stride()),
        "y_max_abs": float(y_diff.max().item()),
        "state_max_abs": float(state_diff.max().item()),
        "deterministic": all(
            torch.equal(outputs[0][0], value[0]) and
            torch.equal(outputs[0][1], value[1])
            for value in outputs[1:]
        ),
        "state_nrmse": float(
            torch.linalg.vector_norm(state_diff)
            / torch.linalg.vector_norm(expected_group_cpu)
        ),
        "y_nrmse": float(
            torch.linalg.vector_norm(y_diff)
            / torch.linalg.vector_norm(expected_y_cpu)
        ),
        "allclose": bool(
            torch.allclose(
                actual_y, expected_y_cpu,
                atol=3e-4, rtol=3e-3,
            ) and
            torch.allclose(
                outputs[0][1], expected_group_cpu,
                atol=2e-5, rtol=2e-3,
            )
        ),
    }
    print(json.dumps(result), flush=True)
    assert result["y_shape"] == [1, 16, 8, 64, 64]
    assert result["state_shape"] == [1, 8, 4, 64, 4, 64]
    assert result["state_stride"] == [524288, 65536, 16384, 256, 64, 1]
    assert result["deterministic"]
    assert result["allclose"]


if __name__ == "__main__":
    test_mamba2_ssd_chunk_mix_grouped()
