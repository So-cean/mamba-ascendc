"""Smoke and precision test for an isolated mamba-ascendc wheel install."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

import torch

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


def _metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    diff = actual - expected
    return {
        "max_abs": diff.abs().max().item(),
        "nrmse": (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(expected).clamp_min(1e-12)
        ).item(),
    }


def test_isolated_wheel_install():
    expected_root = os.environ.get("MAMBA_WHEEL_TEST_TARGET")
    info = ascend_kernel.runtime_info()
    package_root = Path(info["package_root"]).resolve()
    if expected_root:
        assert package_root.is_relative_to(Path(expected_root).resolve())

    assert importlib.metadata.version("mamba-ascendc") == "0.1.0"
    assert Path(info["op_api_library"]).is_file()
    assert Path(info["extension_library"]).is_file()
    assert info["custom_opp_path"] in os.environ["ASCEND_CUSTOM_OPP_PATH"]
    assert os.environ["MAMBA_ASCENDC_CHUNK_MIX"] == "1"

    generator = torch.Generator(device="cpu").manual_seed(20260803)
    batch, seqlen, heads, headdim, dstate, groups = 1, 128, 2, 64, 64, 1

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    x = randn(batch, seqlen, heads, headdim)
    dt = 0.01 + 0.05 * torch.rand(
        batch, seqlen, heads, generator=generator, dtype=torch.float32
    )
    A = -0.1 - 0.2 * torch.rand(heads, generator=generator)
    B = randn(batch, seqlen, groups, dstate) / 5
    C = randn(batch, seqlen, groups, dstate) / 5
    D = randn(heads, headdim)
    z = randn(batch, seqlen, heads, headdim)
    dt_bias = randn(heads) * 0.05
    values = [value.npu() for value in (x, dt, A, B, C, D, z, dt_bias)]
    x_npu, dt_npu, A_npu, B_npu, C_npu, D_npu, z_npu, bias_npu = values

    out, final_state = ascend_kernel.mamba2_ssd_fwd(
        x_npu,
        dt_npu,
        A_npu,
        B_npu,
        C_npu,
        64,
        D=D_npu,
        z=z_npu,
        dt_bias=bias_npu,
        dt_softplus=True,
        return_final_state=True,
    )
    ref_out, ref_state = ssd_chunk_scan_ref(
        x_npu,
        dt_npu,
        A_npu,
        B_npu,
        C_npu,
        64,
        D=D_npu,
        z=z_npu,
        dt_bias=bias_npu,
        dt_softplus=True,
        return_final_state=True,
    )
    torch.npu.synchronize()

    out_metrics = _metrics(out, ref_out)
    state_metrics = _metrics(final_state, ref_state)
    assert out_metrics["nrmse"] <= 5e-3
    assert state_metrics["nrmse"] <= 7e-3

    # Exercise public autograd from the isolated install as well as forward.
    cpu_core = (x, dt, A, B, C)
    ref_values = tuple(value.detach().requires_grad_(True) for value in cpu_core)
    ref_train_out = ssd_chunk_scan_ref(*ref_values, 64)
    grad_generator = torch.Generator(device="cpu").manual_seed(20260804)
    dout = torch.randn(ref_train_out.shape, generator=grad_generator)
    ref_grads = torch.autograd.grad(ref_train_out, ref_values, dout)

    npu_values = tuple(
        value.npu().contiguous().detach().requires_grad_(True)
        for value in cpu_core
    )
    train_out = ascend_kernel.mamba2_ssd_fwd(*npu_values, 64)
    npu_grads = torch.autograd.grad(train_out, npu_values, dout.npu())
    torch.npu.synchronize()
    grad_metrics = {
        name: _metrics(actual, expected)
        for name, actual, expected in zip(
            ("dx", "ddt", "dA", "dB", "dC"), npu_grads, ref_grads
        )
    }
    for name, metrics in grad_metrics.items():
        assert metrics["nrmse"] <= 1e-2, f"{name}: {metrics}"

    print(
        json.dumps(
            {
                "version": importlib.metadata.version("mamba-ascendc"),
                "runtime": info,
                "out": out_metrics,
                "final_state": state_metrics,
                "gradients": grad_metrics,
            },
            indent=2,
        )
    )
