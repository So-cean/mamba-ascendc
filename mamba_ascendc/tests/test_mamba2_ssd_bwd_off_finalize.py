"""Dataflow and precision gate for the FP16 off-finalize boundary."""

from __future__ import annotations

import torch

import ascend_kernel  # noqa: F401 -- register torch.ops.mamba_ascend


def test_mamba2_ssd_bwd_off_finalize_aliases_dstate():
    batch, groups, heads_per_group, chunks = 1, 2, 2, 2
    heads = groups * heads_per_group
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    d_states_half = (
        0.1
        * torch.randn(
            batch,
            groups,
            heads_per_group,
            chunks,
            64,
            64,
            generator=generator,
        )
    ).half()
    d_c_head_half = (
        0.1
        * torch.randn(
            batch,
            groups,
            heads_per_group,
            chunks,
            64,
            64,
            generator=generator,
        )
    ).half()
    c_cube = (
        0.1
        * torch.randn(
            batch, chunks, groups, 64, 64, generator=generator
        )
    ).half()

    expected_states = d_states_half.view(batch, heads, chunks, 64, 64)
    expected_c = (
        d_c_head_half.float().sum(dim=2).permute(0, 2, 3, 1, 4)
    )
    c_group = c_cube.permute(0, 2, 1, 3, 4).unsqueeze(2).float()
    expected_gda = (d_c_head_half.float() * c_group).sum(dim=-1).reshape(
        batch, heads, chunks, 64
    )

    d_states_npu = d_states_half.npu()
    actual_states, actual_c, actual_gda = (
        torch.ops.mamba_ascend.mamba2_ssd_bwd_off_finalize(
            d_states_npu,
            d_c_head_half.npu(),
            c_cube.npu(),
        )
    )
    torch.npu.synchronize()

    assert actual_states.dtype == torch.float16
    assert actual_states.data_ptr() == d_states_npu.data_ptr()
    assert torch.equal(actual_states.cpu(), expected_states)
    torch.testing.assert_close(
        actual_c.cpu(), expected_c, rtol=1.0e-6, atol=1.0e-6
    )
    torch.testing.assert_close(
        actual_gda.cpu(), expected_gda, rtol=1.0e-5, atol=1.0e-6
    )
