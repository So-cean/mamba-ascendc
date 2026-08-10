"""Precision, layout, and determinism gates for OffGroupReduce."""

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
        assert getattr(library, "aclnnMamba2SsdBwdOffGroupReduce")
    import ascend_kernel  # noqa: F401,E402


_load_custom_operator()


def _make_inputs(shape, pattern, seed):
    batch, groups, heads_per_group, chunks = shape
    full_shape = (batch, groups, heads_per_group, chunks, 64, 64)
    c_shape = (batch, chunks, groups, 64, 64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if pattern == "zero":
        q = torch.zeros(full_shape, dtype=torch.float16)
        state = torch.zeros_like(q)
        c = torch.zeros(c_shape, dtype=torch.float16)
    elif pattern == "identity":
        eye = torch.eye(64, dtype=torch.float16)
        q = eye.expand(full_shape).clone()
        state = (0.5 * eye).expand(full_shape).clone()
        c = (0.25 * eye).expand(c_shape).clone()
    elif pattern == "one_hot":
        q = torch.zeros(full_shape, dtype=torch.float16)
        state = torch.zeros_like(q)
        c = torch.zeros(c_shape, dtype=torch.float16)
        q[..., 7, 11] = 1.0
        state[..., 11, 13] = -0.5
        c[..., 7, 13] = 0.25
    else:
        q = (
            0.1 * torch.randn(full_shape, generator=generator)
        ).to(torch.float16)
        state = (
            0.1 * torch.randn(full_shape, generator=generator)
        ).to(torch.float16)
        c = (
            0.1 * torch.randn(c_shape, generator=generator)
        ).to(torch.float16)
    return q, state, c


def _reference(q, state, c):
    d_c_head = torch.matmul(q.float(), state.float())
    d_c_group = d_c_head.sum(dim=2).permute(0, 2, 3, 1, 4).contiguous()
    c_group = c.permute(0, 2, 1, 3, 4).unsqueeze(2).float()
    g_d_a = (d_c_head * c_group).sum(dim=-1).reshape(
        q.shape[0], q.shape[1] * q.shape[2], q.shape[3], 64
    )
    return d_c_group, g_d_a


@pytest.mark.parametrize(
    "shape,pattern",
    [
        ((1, 1, 1, 1), "zero"),
        ((1, 1, 2, 2), "identity"),
        ((1, 2, 4, 1), "one_hot"),
        ((1, 2, 2, 3), "random"),
        ((2, 4, 1, 2), "random"),
        ((1, 16, 4, 16), "random"),
    ],
)
def test_mamba2_ssd_bwd_off_group_reduce(shape, pattern):
    inputs = _make_inputs(shape, pattern, 20260808 + sum(shape))
    expected = _reference(*inputs)
    npu_inputs = tuple(tensor.npu().contiguous() for tensor in inputs)
    first = torch.ops.mamba_ascend.mamba2_ssd_bwd_off_group_reduce(
        *npu_inputs
    )
    second = torch.ops.mamba_ascend.mamba2_ssd_bwd_off_group_reduce(
        *npu_inputs
    )
    torch.npu.synchronize()

    batch, groups, heads_per_group, chunks = shape
    expected_shapes = (
        (batch, chunks, 64, groups, 64),
        (batch, groups * heads_per_group, chunks, 64),
    )
    for name, got, repeated, ref, expected_shape in zip(
        ("d_c_group", "g_d_a"),
        first,
        second,
        expected,
        expected_shapes,
    ):
        assert got.dtype == torch.float32
        assert tuple(got.shape) == expected_shape
        assert got.is_contiguous()
        assert torch.isfinite(got).all(), f"{name} contains non-finite values"
        torch.testing.assert_close(
            got.cpu(), ref, rtol=4.0e-3, atol=4.0e-3,
            msg=lambda message: f"{name}: {message}",
        )
        torch.testing.assert_close(got, repeated, rtol=0.0, atol=0.0)
