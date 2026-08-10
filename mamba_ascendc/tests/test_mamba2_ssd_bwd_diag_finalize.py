"""Precision and cross-core-task regression for diagonal backward finalize."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401


# This is an internal source-tree test.  Load the freshly built extension
# directly so an older installed wheel cannot hide a registration regression.
_opp = Path(os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":", 1)[0])
ctypes.CDLL(str(_opp / "op_api/lib/libcust_opapi.so"), mode=ctypes.RTLD_GLOBAL)
_root = Path(__file__).resolve().parents[2]
if not hasattr(torch.ops.mamba_ascend, "mamba2_ssd_bwd_diag_finalize"):
    torch.ops.load_library(
        str(
            _root
            / "mamba_ascendc/python/ascend_kernel/ascend_kernel/lib"
            / "libascend_kernel.so"
        )
    )


def _reference(
    inputs: list[torch.Tensor], d_a: torch.Tensor,
    groups: int, dc_off: torch.Tensor,
):
    dx_diag, d_r, db_diag, db_state, dc_diag, d_w, w, r = (
        value.float() for value in inputs
    )
    decay = torch.exp(d_a[..., -1:] - d_a)
    dx = dx_diag + d_r * decay[..., :, None]

    product = d_w * w
    g_diag = product.sum(dim=-1) - product.sum(dim=-2)
    state = (d_r * r).sum(dim=-1)
    g_cs = g_diag - state
    g_cs[..., -1] += state.sum(dim=-1)

    batch, heads, chunks, tile, _ = db_diag.shape
    heads_per_group = heads // groups
    db_group = (db_diag + db_state).reshape(
        batch, groups, heads_per_group, chunks, tile, tile
    ).sum(dim=2).permute(0, 2, 3, 1, 4).contiguous()
    dc_group = dc_diag.reshape(
        batch, groups, heads_per_group, chunks, tile, tile
    ).sum(dim=2).permute(0, 2, 3, 1, 4).contiguous() + dc_off
    return dx, db_group, dc_group, g_cs


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    scale = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float((error / scale).cpu())


@pytest.mark.parametrize(
    "batch,heads,chunks,groups,grouped_dr",
    [
        # H/G=1 keeps the original single-head copy path.
        (1, 1, 1, 1, False),
        (1, 4, 2, 2, False),
        # Head-block layout coverage: short tail, full block, and 4+1 tail.
        (1, 3, 2, 1, False),
        (1, 8, 2, 2, False),
        (1, 5, 2, 1, False),
        # 64 group tasks exceed the 910B3's 48 AIV cores.  This guards the
        # MTE3->Vector ordering required when a core reuses its accumulators.
        (4, 8, 8, 2, False),
        # 128 group tasks exercise the producer-native [B,K,G,T,R,P] dR
        # layout and make every core process multiple tasks on both devices.
        (2, 32, 8, 8, True),
    ],
)
def test_diag_finalize_matches_reference_and_is_deterministic(
    batch: int, heads: int, chunks: int, groups: int, grouped_dr: bool
):
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    shape = (batch, heads, chunks, 64, 64)
    host_inputs = [
        (torch.randn(shape, generator=generator) * 0.15).half()
        for _ in range(8)
    ]
    host_d_a = torch.cumsum(
        torch.randn(batch, heads, chunks, 64, generator=generator) * 0.01,
        dim=-1,
    )
    host_dc_off = torch.randn(
        batch, chunks, 64, groups, 64, generator=generator
    ) * 0.15
    expected = _reference(host_inputs, host_d_a, groups, host_dc_off)
    device_inputs = list(host_inputs)
    if grouped_dr:
        heads_per_group = heads // groups
        device_inputs[1] = (
            host_inputs[1]
            .reshape(
                batch, groups, heads_per_group, chunks, 64, 64
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
        )
    inputs = [value.npu() for value in device_inputs]
    d_a = host_d_a.npu()
    dc_off = host_dc_off.npu()

    first = torch.ops.mamba_ascend.mamba2_ssd_bwd_diag_finalize(
        *inputs, d_a, groups, dc_off
    )
    second = torch.ops.mamba_ascend.mamba2_ssd_bwd_diag_finalize(
        *inputs, d_a, groups, dc_off
    )
    torch.npu.synchronize()

    for actual, reference in zip(first, expected):
        assert _nrmse(actual.cpu(), reference) < 2e-3
    for actual, repeated in zip(first, second):
        assert torch.equal(actual.cpu(), repeated.cpu())
