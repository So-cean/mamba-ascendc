"""Small complete-output gate for the Arch35 direct-Matmul Off path."""

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


def _reference(gy, states, d_a, c_cube):
    q = (gy * torch.exp(d_a)[..., None]).half().float()
    state_half = states.half().float()
    c_float = c_cube.permute(0, 2, 1, 3, 4).float()
    d_states = torch.matmul(q.transpose(-1, -2), c_float)
    d_c_head = torch.matmul(q, state_half)
    g_d_a = (d_c_head * c_float).sum(dim=-1)
    return d_states, d_c_head, g_d_a


def _inputs(mode: str):
    shape = (1, 1, 1, 64, 64)
    if mode == "zero":
        gy = torch.zeros(shape, dtype=torch.float32)
        states = torch.zeros(shape, dtype=torch.float32)
        d_a = torch.zeros((1, 1, 1, 64), dtype=torch.float32)
        c_cube = torch.zeros((1, 1, 1, 64, 64), dtype=torch.float16)
        return gy, states, d_a, c_cube

    if mode == "identity":
        identity = torch.eye(64, dtype=torch.float32).reshape(shape)
        return (
            identity.clone(),
            identity.clone(),
            torch.zeros((1, 1, 1, 64), dtype=torch.float32),
            identity.half(),
        )

    if mode == "one_hot":
        gy = torch.zeros(shape, dtype=torch.float32)
        states = torch.zeros(shape, dtype=torch.float32)
        c_cube = torch.zeros((1, 1, 1, 64, 64), dtype=torch.float16)
        gy[0, 0, 0, 3, 5] = 1.0
        states[0, 0, 0, 5, 7] = 1.0
        c_cube[0, 0, 0, 3, 11] = 1.0
        d_a = torch.zeros((1, 1, 1, 64), dtype=torch.float32)
        return gy, states, d_a, c_cube

    generator = torch.Generator(device="cpu").manual_seed(20260809)
    gy = 0.1 * torch.randn(shape, generator=generator)
    states = 0.1 * torch.randn(shape, generator=generator)
    d_a = -0.01 - 0.2 * torch.rand(
        (1, 1, 1, 64), generator=generator
    )
    c_cube = (
        0.1
        * torch.randn((1, 1, 1, 64, 64), generator=generator)
    ).half()
    return gy, states, d_a, c_cube


@pytest.mark.parametrize("mode", ("zero", "identity", "one_hot", "random"))
def test_arch35_direct_complete_outputs(mode):
    inputs = _inputs(mode)
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
        host = got.cpu()
        assert got.dtype == torch.float32
        assert torch.isfinite(host).all(), f"{name} contains non-finite values"
        torch.testing.assert_close(
            host,
            ref,
            rtol=3.0e-3,
            atol=3.0e-3,
            msg=lambda message: f"{mode}/{name}: {message}",
        )


def test_arch35_direct_is_deterministic():
    inputs = tuple(tensor.npu().contiguous() for tensor in _inputs("random"))
    first = torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_off(*inputs)
    second = torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_off(*inputs)
    torch.npu.synchronize()

    for name, first_output, second_output in zip(
        ("d_states_start", "d_c_head", "g_dA_cs_off"), first, second
    ):
        assert torch.equal(first_output.cpu(), second_output.cpu()), (
            f"{name} differs across identical launches"
        )
