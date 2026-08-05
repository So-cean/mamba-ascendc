"""Public API and Triton-Ascend compilation smoke tests for Mamba-2 SSD."""

from __future__ import annotations

import pytest
import torch

from mamba_torch.ssd_reference import ssd_chunk_scan_ref
from mamba_triton_ascend.mamba2.ops import mamba_chunk_scan_combined

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def _inputs():
    generator = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(1, 32, 2, 8, generator=generator)
    dt = 0.01 + 0.1 * torch.rand(1, 32, 2, generator=generator)
    A = -0.1 - 0.4 * torch.rand(2, generator=generator)
    B = torch.randn(1, 32, 1, 8, generator=generator) / 5
    C = torch.randn(1, 32, 1, 8, generator=generator) / 5
    return x, dt, A, B, C


def test_public_cpu_dispatch_matches_reference():
    inputs = _inputs()
    expected = ssd_chunk_scan_ref(*inputs, chunk_size=16)
    actual = mamba_chunk_scan_combined(*inputs, chunk_size=16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_forced_triton_rejects_cpu_inputs():
    with pytest.raises(ValueError, match="requires NPU inputs"):
        mamba_chunk_scan_combined(*_inputs(), chunk_size=16, backend="triton")


@pytest.mark.mamba2_npu
@pytest.mark.skipif(
    torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU not available",
)
def test_public_npu_triton_compile_and_run():
    inputs = _inputs()
    npu_inputs = tuple(value.npu() for value in inputs)
    with torch.no_grad():
        out, final_state = mamba_chunk_scan_combined(
            *npu_inputs,
            chunk_size=16,
            return_final_states=True,
            backend="triton",
        )
    torch.npu.synchronize()
    assert out.shape == (1, 32, 2, 8)
    assert final_state.shape == (1, 2, 8, 8)
    assert out.device.type == "npu"
    assert torch.isfinite(out).all()
    assert torch.isfinite(final_state).all()
