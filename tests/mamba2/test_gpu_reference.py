"""CPU PyTorch Mamba-2 SSD forward vs official GPU mamba_ssm.

The reference is deliberately executed on CPU. Only the official
``mamba_chunk_scan_combined`` call runs on CUDA. This file validates forward
output and final recurrent state; backward is outside the current scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from mamba_torch.ssd_reference import ssd_chunk_scan_ref

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None


pytestmark = [
    pytest.mark.mamba2_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(mamba_chunk_scan_combined is None, reason="mamba_ssm not installed"),
]


@dataclass(frozen=True)
class FeatureConfig:
    d_kind: str = "none"
    use_z: bool = False
    use_dt_bias: bool = False
    dt_softplus: bool = False
    use_initial_state: bool = False
    dt_limit: tuple[float, float] = (0.0, float("inf"))


SHAPES = [
    pytest.param(1, 32, 2, 8, 8, 16, 1, id="small"),
    pytest.param(2, 64, 4, 16, 16, 32, 1, id="batch"),
    pytest.param(1, 65, 4, 16, 16, 32, 2, id="tail-chunk"),
    pytest.param(2, 128, 8, 32, 32, 64, 2, id="multi-group"),
    pytest.param(1, 256, 8, 32, 64, 64, 4, id="medium"),
]

FEATURES = [
    pytest.param(FeatureConfig(), id="basic"),
    pytest.param(FeatureConfig(d_kind="head"), id="D-head"),
    pytest.param(FeatureConfig(d_kind="channel", use_z=True), id="D-channel-z"),
    pytest.param(
        FeatureConfig(use_dt_bias=True, dt_softplus=True), id="bias-softplus"
    ),
    pytest.param(
        FeatureConfig(
            d_kind="channel",
            use_z=True,
            use_dt_bias=True,
            dt_softplus=True,
            use_initial_state=True,
            dt_limit=(1e-3, 0.2),
        ),
        id="all-initial-limit",
    ),
]

DTYPES = [
    # Match mamba_ssm/tests/ops/triton/test_ssd.py.
    pytest.param(torch.float32, 1e-2, 3e-3, id="fp32"),
    pytest.param(torch.float16, 1e-2, 3e-3, id="fp16"),
    # One BF16 ULP is 4.88e-3 around values seen in the medium cases.
    pytest.param(torch.bfloat16, 1e-2, 5e-3, id="bf16"),
]


def _make_cpu_inputs(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    dstate: int,
    ngroups: int,
    dtype: torch.dtype,
    features: FeatureConfig,
    seed: int = 42,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(dtype)

    x = randn(batch, seqlen, nheads, headdim)
    dt = (0.01 + 0.1 * torch.rand(
        batch, seqlen, nheads, generator=generator, dtype=torch.float32
    )).to(dtype)
    # mamba_chunk_scan_combined expects the negative decay value, not A_log.
    A = (-0.1 - 0.4 * torch.rand(nheads, generator=generator)).float()
    # Match the official mamba_ssm SSD tests and keep the recurrent state in a
    # representative, numerically stable range.
    B = randn(batch, seqlen, ngroups, dstate) / 5
    C = randn(batch, seqlen, ngroups, dstate) / 5

    if features.d_kind == "head":
        D = randn(nheads)
    elif features.d_kind == "channel":
        D = randn(nheads, headdim)
    else:
        D = None

    z = randn(batch, seqlen, nheads, headdim) if features.use_z else None
    dt_bias = randn(nheads) * 0.1 if features.use_dt_bias else None
    initial_states = (
        randn(batch, nheads, headdim, dstate) / 5
        if features.use_initial_state
        else None
    )
    return x, dt, A, B, C, D, z, dt_bias, initial_states


def _to_cuda(value):
    return value.cuda() if isinstance(value, torch.Tensor) else value


def _metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    diff = (actual - expected).abs()
    nrmse = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    ).clamp_min(1e-12)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "nrmse": nrmse.item(),
    }


@pytest.mark.parametrize(
    "batch,seqlen,nheads,headdim,dstate,chunk_size,ngroups", SHAPES
)
@pytest.mark.parametrize("features", FEATURES)
@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_cpu_reference_matches_gpu_mamba_fwd(
    batch,
    seqlen,
    nheads,
    headdim,
    dstate,
    chunk_size,
    ngroups,
    features,
    dtype,
    rtol,
    atol,
):
    cpu_inputs = _make_cpu_inputs(
        batch,
        seqlen,
        nheads,
        headdim,
        dstate,
        ngroups,
        dtype,
        features,
    )
    x, dt, A, B, C, D, z, dt_bias, initial_states = cpu_inputs

    out_cpu, final_cpu = ssd_chunk_scan_ref(
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
    assert out_cpu.device.type == "cpu"
    assert final_cpu.device.type == "cpu"

    cuda_inputs = tuple(_to_cuda(value) for value in cpu_inputs)
    x_g, dt_g, A_g, B_g, C_g, D_g, z_g, dt_bias_g, initial_g = cuda_inputs
    with torch.no_grad():
        out_gpu, final_gpu = mamba_chunk_scan_combined(
            x_g,
            dt_g,
            A_g,
            B_g,
            C_g,
            chunk_size,
            D=D_g,
            z=z_g,
            dt_bias=dt_bias_g,
            initial_states=initial_g,
            dt_softplus=features.dt_softplus,
            dt_limit=features.dt_limit,
            return_final_states=True,
        )
    torch.cuda.synchronize()

    out_gpu = out_gpu.cpu()
    final_gpu = final_gpu.cpu()
    out_metrics = _metrics(out_gpu, out_cpu)
    final_metrics = _metrics(final_gpu, final_cpu)
    print(f"output={out_metrics} final_state={final_metrics}")

    assert torch.isfinite(out_gpu).all()
    assert torch.isfinite(final_gpu).all()
    torch.testing.assert_close(out_gpu.float(), out_cpu.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(
        final_gpu.float(), final_cpu.float(), rtol=rtol, atol=atol
    )
