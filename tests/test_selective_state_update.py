# Copyright (c) 2024, mamba-triton-ascend authors.
# Smoke tests for selective_state_update (inference kernel).
# No reference needed; verifies shape, determinism, state mutation.

import pytest
import torch

from mamba_triton_ascend.ops.selective_state_update import selective_state_update


def _make_state_inputs(batch, dim, dstate, device="npu"):
    torch.manual_seed(42)
    state = torch.randn(batch, dim, dstate, device=device)
    x = torch.randn(batch, dim, device=device)
    dt = torch.rand(batch, dim, device=device) * 0.1 + 0.01
    A = torch.randn(dim, dstate, device=device) * 0.1
    B = torch.randn(batch, dstate, device=device)
    C = torch.randn(batch, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn(batch, dim, device=device)
    dt_bias = torch.randn(dim, device=device) * 0.01
    return state, x, dt, A, B, C, D, z, dt_bias


class TestStateUpdateSmoke:
    @pytest.mark.usefixtures("clear_triton_cache")
    def test_shape(self, npu_available):
        if not npu_available:
            pytest.skip("NPU not available")
        batch, dim, dstate = 2, 64, 16
        state, x, dt, A, B, C, D, z, dt_bias = _make_state_inputs(batch, dim, dstate)
        out = selective_state_update(
            state, x, dt, A, B, C,
            D=D, z=z, dt_bias=dt_bias, dt_softplus=True
        )
        assert out.shape == (batch, dim)

    @pytest.mark.usefixtures("clear_triton_cache")
    def test_no_nan(self, npu_available):
        if not npu_available:
            pytest.skip("NPU not available")
        batch, dim, dstate = 2, 64, 16
        state, x, dt, A, B, C, D, z, dt_bias = _make_state_inputs(batch, dim, dstate)
        out = selective_state_update(
            state, x, dt, A, B, C,
            D=D, z=z, dt_bias=dt_bias, dt_softplus=True
        )
        assert torch.isfinite(out).all()

    @pytest.mark.usefixtures("clear_triton_cache")
    def test_state_mutated(self, npu_available):
        if not npu_available:
            pytest.skip("NPU not available")
        batch, dim, dstate = 2, 64, 16
        state, x, dt, A, B, C, D, z, dt_bias = _make_state_inputs(batch, dim, dstate)
        state_before = state.clone()
        _ = selective_state_update(
            state, x, dt, A, B, C,
            D=D, z=z, dt_bias=dt_bias, dt_softplus=True
        )
        assert not torch.allclose(state, state_before), \
            "State should be mutated in-place"

    @pytest.mark.usefixtures("clear_triton_cache")
    def test_determinism(self, npu_available):
        if not npu_available:
            pytest.skip("NPU not available")
        batch, dim, dstate = 2, 64, 16
        # Run 1
        torch.manual_seed(42)
        s1, x1, dt1, A1, B1, C1, D1, z1, db1 = _make_state_inputs(batch, dim, dstate)
        out1 = selective_state_update(s1, x1, dt1, A1, B1, C1, D=D1, z=z1, dt_bias=db1, dt_softplus=True)
        # Run 2
        torch.manual_seed(42)
        s2, x2, dt2, A2, B2, C2, D2, z2, db2 = _make_state_inputs(batch, dim, dstate)
        out2 = selective_state_update(s2, x2, dt2, A2, B2, C2, D=D2, z=z2, dt_bias=db2, dt_softplus=True)
        assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)
