"""Precision and cross-core reuse gate for grouped Off finalize."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
import pytest

import ascend_kernel  # noqa: F401  -- configures OPP and loads the extension


@pytest.mark.parametrize(
    ("batch", "chunks", "groups", "scale"),
    (
        (2, 8, 8, 0.1),
        # 1024 tasks force repeated UB/MTE3 reuse on every core.  The larger
        # values cover the dynamic range seen after the H256 Cube stages,
        # where a small g_dA_cs error is amplified by the reverse scan.
        (2, 16, 32, 3.0),
    ),
)
def test_grouped_off_finalize_cross_core_is_correct_and_deterministic(
    batch, chunks, groups, scale
):
    # B*K*G exceeds both tested AIV core counts, forcing each core to reuse
    # dCFloat/gDaHead after an MTE3 store from an earlier task.
    heads_per_group = 4
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    q = (
        scale
        * torch.randn(
            batch,
            chunks,
            groups,
            64,
            heads_per_group,
            64,
            generator=generator,
        )
    ).half()
    y = (scale * torch.randn(q.shape, generator=generator)).half()
    d_states = (scale * torch.randn(q.shape, generator=generator)).half()
    d_c_half = (
        scale
        * torch.randn(
            batch, chunks, groups, 64, 64, generator=generator
        )
    ).half()

    expected_c = d_c_half.float().permute(0, 1, 3, 2, 4).contiguous()
    expected_gda = (q.float() * y.float()).sum(dim=-1)
    expected_gda = expected_gda.permute(0, 2, 4, 1, 3).reshape(
        batch, groups * heads_per_group, chunks, 64
    )
    args = tuple(value.npu() for value in (d_states, d_c_half, q, y))
    first = torch.ops.mamba_ascend.mamba2_ssd_bwd_off_finalize_grouped(*args)
    second = torch.ops.mamba_ascend.mamba2_ssd_bwd_off_finalize_grouped(*args)
    torch.npu.synchronize()

    assert first[0].data_ptr() == args[0].data_ptr()
    assert torch.equal(first[0].cpu(), d_states)
    torch.testing.assert_close(
        first[1].cpu(), expected_c, rtol=1.0e-6, atol=1.0e-6
    )
    torch.testing.assert_close(
        first[2].cpu(),
        expected_gda,
        rtol=1.0e-5,
        atol=1.0e-6 if scale < 1.0 else 5.0e-5,
    )
    for actual, repeated in zip(first, second):
        assert torch.equal(actual.cpu(), repeated.cpu())
