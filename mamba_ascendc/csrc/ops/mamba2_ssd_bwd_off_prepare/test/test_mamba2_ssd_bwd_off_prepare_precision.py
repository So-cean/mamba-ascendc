"""Precision and tail-tiling gate for Mamba2 SSD OffPrepare."""

from __future__ import annotations

import pytest
import torch

import ascend_kernel  # noqa: F401 -- register torch.ops.mamba_ascend


CASES = (
    (1, 1, 1, torch.float16),
    (1, 2, 2, torch.float32),
    (1, 4, 3, torch.float16),
    (2, 3, 4, torch.float32),
    (1, 8, 5, torch.float16),
    (2, 8, 7, torch.float32),
    (1, 16, 64, torch.float16),
    (1, 8, 65, torch.float16),
)


@pytest.mark.parametrize("batch,nheads,nchunks,state_dtype", CASES)
def test_off_prepare_precision_and_determinism(
    batch: int,
    nheads: int,
    nchunks: int,
    state_dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(
        20260808 + batch * 1000 + nheads * 10 + nchunks
    )
    shape = (batch, nheads, nchunks, 64, 64)
    gy_cpu = (0.2 * torch.randn(shape, generator=generator)).half()
    state_cpu = (0.2 * torch.randn(shape, generator=generator)).to(state_dtype)
    d_a_cpu = -0.05 * torch.rand(
        batch, nheads, nchunks, 64, generator=generator
    )

    expected_q = (
        gy_cpu.float() * torch.exp(d_a_cpu).unsqueeze(-1)
    ).half()
    expected_state = state_cpu.half()

    gy = gy_cpu.npu()
    state = state_cpu.npu()
    d_a = d_a_cpu.npu()
    q0, state0 = torch.ops.mamba_ascend.mamba2_ssd_bwd_off_prepare(
        gy, state, d_a
    )
    q1, state1 = torch.ops.mamba_ascend.mamba2_ssd_bwd_off_prepare(
        gy, state, d_a
    )
    torch.npu.synchronize()

    q0_cpu = q0.cpu()
    state0_cpu = state0.cpu()
    assert torch.equal(q0_cpu, q1.cpu())
    assert torch.equal(state0_cpu, state1.cpu())
    torch.testing.assert_close(q0_cpu, expected_q, rtol=1.0e-3, atol=1.0e-3)
    torch.testing.assert_close(
        state0_cpu, expected_state, rtol=0.0, atol=0.0
    )
    if state_dtype == torch.float16:
        assert state0.data_ptr() == state.data_ptr()
