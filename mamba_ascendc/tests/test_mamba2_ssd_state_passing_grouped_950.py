#!/usr/bin/env python3
"""Precision gate for group-contiguous StatePassing on Ascend 950PR."""

from __future__ import annotations

import json

import torch
import torch_npu  # noqa: F401

import ascend_kernel  # noqa: F401


def _reference(
    chunk_states: torch.Tensor,
    d_a_cumsum: torch.Tensor,
    initial: torch.Tensor | None,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, chunks, state_dim, head_dim = chunk_states.shape
    state = (
        initial.float().clone() if initial is not None
        else torch.zeros(
            (batch, heads, state_dim, head_dim), dtype=torch.float32
        )
    )
    starts = []
    for chunk in range(chunks):
        starts.append(state.clone())
        decay = torch.exp(d_a_cumsum[:, :, chunk, -1]).view(
            batch, heads, 1, 1
        )
        state = state * decay + chunk_states[:, :, chunk]
    head_major = torch.stack(starts, dim=2)
    heads_per_group = heads // groups
    grouped = (
        head_major.view(
            batch, groups, heads_per_group, chunks, state_dim, head_dim
        )
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .half()
    )
    return grouped, state


def test_mamba2_ssd_state_passing_grouped() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    chunk_states = 0.025 * torch.randn(
        (1, 8, 4, 64, 64), generator=generator
    )
    increments = -0.02 * torch.rand(
        (1, 8, 4, 64), generator=generator
    )
    d_a_cumsum = torch.cumsum(increments, dim=-1)
    initial = 0.025 * torch.randn(
        (1, 8, 64, 64), generator=generator
    )

    chunk_states_grouped = (
        chunk_states.view(1, 2, 4, 4, 64, 64)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
    )
    for input_grouped in (False, True):
        for with_initial in (False, True):
            initial_value = initial if with_initial else None
            expected_grouped, expected_final = _reference(
                chunk_states, d_a_cumsum, initial_value, groups=2
            )
            chunk_input = (
                chunk_states_grouped if input_grouped else chunk_states
            )
            chunk_npu = chunk_input.npu()
            da_npu = d_a_cumsum.npu()
            initial_npu = initial.npu() if with_initial else None
            outputs = []
            for _ in range(3):
                states_grouped, final_np = (
                    torch.ops.mamba_ascend.mamba2_ssd_state_passing_grouped(
                        chunk_npu, da_npu, initial_npu, 2
                    )
                )
                torch.npu.synchronize()
                outputs.append((states_grouped.cpu(), final_np.cpu()))

            actual_grouped = outputs[0][0]
            actual_final = outputs[0][1]
            grouped_diff = (
                actual_grouped.float() - expected_grouped.float()
            ).abs()
            final_diff = (actual_final - expected_final).abs()
            result = {
                "input_grouped": input_grouped,
                "with_initial": with_initial,
                "grouped_shape": list(actual_grouped.shape),
                "grouped_stride": list(actual_grouped.stride()),
                "grouped_max_abs": float(grouped_diff.max().item()),
                "final_max_abs": float(final_diff.max().item()),
                "deterministic": all(
                    torch.equal(outputs[0][0], value[0]) and
                    torch.equal(outputs[0][1], value[1])
                    for value in outputs[1:]
                ),
                "grouped_allclose": bool(torch.allclose(
                    actual_grouped.float(), expected_grouped.float(),
                    rtol=5.0e-3, atol=5.0e-3
                )),
                "final_allclose": bool(torch.allclose(
                    actual_final, expected_final,
                    rtol=5.0e-3, atol=5.0e-3
                )),
            }
            print(json.dumps(result), flush=True)
            assert result["grouped_shape"] == [1, 4, 2, 64, 4, 64]
            assert result["grouped_stride"] == [
                131072, 32768, 16384, 256, 64, 1
            ]
            assert result["deterministic"]
            assert result["grouped_allclose"]
            assert result["final_allclose"]


if __name__ == "__main__":
    test_mamba2_ssd_state_passing_grouped()
