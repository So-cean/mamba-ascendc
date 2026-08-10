"""Four-case correctness gate for Mamba2SsdChunkScanBwdOff."""

from __future__ import annotations

import ctypes
import os

import pytest
import torch
import torch_npu  # noqa: F401


def _load_custom_operator():
    library_path = os.environ.get("MAMBA_CHUNK_MIX_OP_API_LIB")
    if library_path:
        library = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
        assert getattr(library, "aclnnMamba2SsdChunkScanBwdOff")
    import ascend_kernel  # noqa: F401,E402


_load_custom_operator()


def _make_inputs(shape, seed):
    batch, heads, chunks, groups = shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gy = 0.1 * torch.randn(
        batch, heads, chunks, 64, 64, generator=generator,
        dtype=torch.float32,
    )
    states = 0.1 * torch.randn(
        batch, heads, chunks, 64, 64, generator=generator,
        dtype=torch.float32,
    )
    # Stable negative prefix sums exercise row-dependent decay without making
    # the FP16 Q path underflow.
    d_a = -0.01 - 0.2 * torch.rand(
        batch, heads, chunks, 64, generator=generator,
        dtype=torch.float32,
    )
    c_cube = (
        0.1
        * torch.randn(
            batch, chunks, groups, 64, 64, generator=generator,
            dtype=torch.float32,
        )
    ).half()
    return gy, states, d_a, c_cube


def _reference(gy, states, d_a, c_cube):
    heads = gy.shape[1]
    groups = c_cube.shape[2]
    c_head = c_cube.permute(0, 2, 1, 3, 4).repeat_interleave(
        heads // groups, dim=1
    )
    # Model the two FP16 Cube inputs explicitly, while keeping FP32
    # accumulation in the CPU reference.
    q = (gy * torch.exp(d_a)[..., None]).half().float()
    state_half = states.half().float()
    c_float = c_head.float()
    d_states = torch.matmul(q.transpose(-1, -2), c_float)
    d_c_head = torch.matmul(q, state_half)
    g_d_a = (d_c_head * c_float).sum(dim=-1)
    return d_states, d_c_head, g_d_a


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),
        (1, 2, 2, 1),
        (1, 4, 2, 2),
        (2, 4, 1, 4),
    ],
)
def test_mamba2_ssd_chunk_scan_bwd_off(shape):
    inputs = _make_inputs(shape, seed=20260808 + sum(shape))
    expected = _reference(*inputs)
    actual = torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_off(
        *(tensor.npu().contiguous() for tensor in inputs)
    )
    torch.npu.synchronize()

    for name, got, ref in zip(
        ("d_states_start", "d_c_head", "g_dA_cs_off"),
        actual,
        expected,
    ):
        assert got.dtype == torch.float32
        assert torch.isfinite(got).all(), f"{name} contains non-finite values"
        torch.testing.assert_close(
            got.cpu(), ref, rtol=3.0e-3, atol=3.0e-3,
            msg=lambda message: f"{name}: {message}",
        )
