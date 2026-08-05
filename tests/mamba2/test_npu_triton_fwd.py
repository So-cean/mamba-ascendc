"""Mamba-2 SSD forward precision tests for Triton-Ascend."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from mamba_torch.ssd_reference import ssd_chunk_scan_ref

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def _npu_available():
    return torch_npu is not None and torch.npu.is_available()


pytestmark = [
    pytest.mark.mamba2_npu,
    pytest.mark.skipif(not _npu_available(), reason="Ascend NPU not available"),
]


@dataclass(frozen=True)
class FeatureConfig:
    use_d: bool = False
    use_z: bool = False
    use_dt_bias: bool = False
    dt_softplus: bool = False
    use_initial_states: bool = False
    dt_limit: tuple[float, float] = (0.0, float("inf"))


SHAPES = [
    pytest.param(1, 32, 2, 8, 8, 16, 1, id="small"),
    pytest.param(1, 65, 4, 16, 16, 32, 2, id="tail-chunk"),
    pytest.param(2, 128, 8, 32, 32, 64, 2, id="multi-group"),
    pytest.param(1, 256, 8, 32, 64, 128, 2, id="chunk128"),
    pytest.param(1, 128, 2, 64, 64, 64, 1, id="cube-fast-path"),
    pytest.param(1, 128, 2, 64, 64, 128, 1, id="cube-fast-128"),
    pytest.param(1, 128, 2, 64, 128, 128, 1, id="cube-fast-dstate128"),
]

FEATURES = [
    pytest.param(FeatureConfig(), id="basic"),
    pytest.param(FeatureConfig(use_d=True), id="D"),
    pytest.param(FeatureConfig(use_z=True), id="z"),
    pytest.param(FeatureConfig(use_initial_states=True), id="initial-state"),
    pytest.param(
        FeatureConfig(use_dt_bias=True, dt_softplus=True, dt_limit=(0.005, 0.2)),
        id="softplus-bias",
    ),
    pytest.param(
        FeatureConfig(
            use_d=True,
            use_z=True,
            use_dt_bias=True,
            dt_softplus=True,
            use_initial_states=True,
            dt_limit=(0.005, 0.2),
        ),
        id="all",
    ),
]


def _make_inputs(batch, seqlen, nheads, headdim, dstate, ngroups, features):
    generator = torch.Generator(device="cpu").manual_seed(42)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    x = randn(batch, seqlen, nheads, headdim)
    dt = 0.01 + 0.1 * torch.rand(
        batch, seqlen, nheads, generator=generator, dtype=torch.float32
    )
    A = -0.1 - 0.4 * torch.rand(nheads, generator=generator)
    B = randn(batch, seqlen, ngroups, dstate) / 5
    C = randn(batch, seqlen, ngroups, dstate) / 5
    D = randn(nheads, headdim) if features.use_d else None
    z = randn(batch, seqlen, nheads, headdim) if features.use_z else None
    dt_bias = randn(nheads) * 0.1 if features.use_dt_bias else None
    initial_states = (
        randn(batch, nheads, headdim, dstate) / 10
        if features.use_initial_states
        else None
    )
    return x, dt, A, B, C, D, z, dt_bias, initial_states


def _to_npu(value):
    return value.npu() if isinstance(value, torch.Tensor) else value


@pytest.mark.parametrize(
    "batch,seqlen,nheads,headdim,dstate,chunk_size,ngroups", SHAPES
)
@pytest.mark.parametrize("features", FEATURES)
def test_triton_fwd_matches_cpu_reference(
    batch,
    seqlen,
    nheads,
    headdim,
    dstate,
    chunk_size,
    ngroups,
    features,
):
    from mamba_triton_ascend.mamba2.ops.ssd_combined import (
        mamba_chunk_scan_combined,
    )

    cpu_inputs = _make_inputs(
        batch, seqlen, nheads, headdim, dstate, ngroups, features
    )
    x, dt, A, B, C, D, z, dt_bias, initial_states = cpu_inputs
    out_ref, final_ref = ssd_chunk_scan_ref(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=features.dt_softplus,
        dt_limit=features.dt_limit,
        initial_states=initial_states,
        return_final_state=True,
    )

    x_n, dt_n, A_n, B_n, C_n, D_n, z_n, dt_bias_n, initial_states_n = tuple(
        _to_npu(value) for value in cpu_inputs
    )
    with torch.no_grad():
        out_npu, final_npu = mamba_chunk_scan_combined(
            x_n,
            dt_n,
            A_n,
            B_n,
            C_n,
            chunk_size,
            D=D_n,
            z=z_n,
            dt_bias=dt_bias_n,
            dt_softplus=features.dt_softplus,
            dt_limit=features.dt_limit,
            initial_states=initial_states_n,
            return_final_states=True,
            backend="triton",
        )
    torch.npu.synchronize()
    out_npu = out_npu.cpu().float()
    final_npu = final_npu.cpu().float()
    out_ref = out_ref.float()
    final_ref = final_ref.float()

    out_nrmse = (
        torch.linalg.vector_norm(out_npu - out_ref)
        / torch.linalg.vector_norm(out_ref).clamp_min(1e-12)
    ).item()
    final_nrmse = (
        torch.linalg.vector_norm(final_npu - final_ref)
        / torch.linalg.vector_norm(final_ref).clamp_min(1e-12)
    ).item()
    print(f"out_nrmse={out_nrmse:.3e} final_nrmse={final_nrmse:.3e}")

    assert torch.isfinite(out_npu).all()
    assert torch.isfinite(final_npu).all()
    torch.testing.assert_close(out_npu, out_ref, rtol=1e-2, atol=3e-3)
    torch.testing.assert_close(final_npu, final_ref, rtol=1e-2, atol=3e-3)
