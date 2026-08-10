"""Correctness gate for the native fused x/dt/A backward epilogue."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ascend_kernel  # noqa: F401  -- loads the torch custom-op library


def _reference(x, d_xdt, g_cs, dt, a, dt_bias, dt_softplus, dt_limit):
    batch, heads, chunks, chunk_size, headdim = d_xdt.shape
    u = dt + (0.0 if dt_bias is None else dt_bias)
    q_pre = F.softplus(u) if dt_softplus else u
    q = q_pre.clamp(min=dt_limit[0], max=dt_limit[1])
    q_bhkt = q.permute(0, 2, 1).reshape(batch, heads, chunks, chunk_size)
    x_bhktp = x.permute(0, 2, 1, 3).reshape_as(d_xdt)

    gw = torch.flip(
        torch.cumsum(torch.flip(g_cs, dims=(-1,)), dim=-1), dims=(-1,)
    )
    dq_direct = (d_xdt * x_bhktp).sum(dim=-1)
    dq = dq_direct + gw * a[None, :, None, None]
    clamp_grad = (
        (q_pre >= dt_limit[0]) & (q_pre <= dt_limit[1])
    ).to(dt.dtype)
    softplus_grad = torch.sigmoid(u) if dt_softplus else torch.ones_like(u)
    chain = (clamp_grad * softplus_grad).permute(0, 2, 1).reshape_as(dq)
    ddt_bhkt = dq * chain

    dx = (d_xdt * q_bhkt[..., None]).reshape(
        batch, heads, chunks * chunk_size, headdim
    ).permute(0, 2, 1, 3)
    ddt = ddt_bhkt.reshape(batch, heads, chunks * chunk_size).permute(0, 2, 1)
    d_a = (gw * q_bhkt).sum(dim=(0, 2, 3))
    d_bias = ddt.sum(dim=(0, 1))
    return dx, ddt, d_a, d_bias


@pytest.mark.parametrize(
    "shape,use_bias,dt_softplus,dt_limit",
    [
        ((1, 64, 1, 64), False, False, (-10.0, 10.0)),
        ((1, 256, 4, 64), True, True, (0.0, float("inf"))),
        ((2, 128, 8, 64), True, False, (-0.15, 0.2)),
        ((2, 256, 2, 64), False, True, (0.2, 1.5)),
    ],
)
def test_mamba2_ssd_dt_bwd(shape, use_bias, dt_softplus, dt_limit):
    batch, seqlen, heads, headdim = shape
    chunk_size = 128 if seqlen % 128 == 0 else 64
    chunks = seqlen // chunk_size
    generator = torch.Generator(device="cpu").manual_seed(20260807)
    x = 0.2 * torch.randn(
        batch, seqlen, heads, headdim, generator=generator
    )
    d_xdt = 0.1 * torch.randn(
        batch, heads, chunks, chunk_size, headdim, generator=generator
    )
    g_cs = 0.1 * torch.randn(
        batch, heads, chunks, chunk_size, generator=generator
    )
    dt = 0.3 * torch.randn(batch, seqlen, heads, generator=generator)
    a = -0.1 - 0.4 * torch.rand(heads, generator=generator)
    dt_bias = (
        0.2 * torch.randn(heads, generator=generator) if use_bias else None
    )
    expected = _reference(
        x, d_xdt, g_cs, dt, a, dt_bias, dt_softplus, dt_limit
    )
    actual = torch.ops.mamba_ascend.mamba2_ssd_dt_bwd(
        x.npu(),
        d_xdt.npu(),
        g_cs.npu(),
        dt.npu(),
        a.npu(),
        None if dt_bias is None else dt_bias.npu(),
        dt_softplus,
        dt_limit[0],
        dt_limit[1],
    )
    torch.npu.synchronize()

    tolerances = {
        "dx": (2.0e-4, 2.0e-4),
        "ddt": (3.0e-4, 3.0e-4),
        "dA": (4.0e-4, 4.0e-4),
        "dt_bias": (4.0e-4, 4.0e-4),
    }
    for name, got, want in zip(tolerances, actual, expected):
        rtol, atol = tolerances[name]
        torch.testing.assert_close(
            got.cpu(), want, rtol=rtol, atol=atol, msg=name
        )


@pytest.mark.parametrize("batch,chunks,groups", [(1, 2, 2), (2, 4, 4)])
def test_grouped_dt_fuses_diag_off_and_chunk_tail(batch, chunks, groups):
    """Grouped consumer must match the validated head-major Dt path."""
    heads_per_group = 4
    heads = groups * heads_per_group
    chunk_size = 64
    headdim = 64
    seqlen = chunks * chunk_size
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    x = 0.15 * torch.randn(
        batch, seqlen, heads, headdim, generator=generator
    )
    d_xdt = 0.1 * torch.randn(
        batch, heads, chunks, chunk_size, headdim, generator=generator
    )
    g_diag = 0.05 * torch.randn(
        batch, heads, chunks, chunk_size, generator=generator
    )
    g_off = 0.05 * torch.randn(
        batch,
        chunks,
        groups,
        chunk_size,
        heads_per_group,
        generator=generator,
    )
    g_last = 0.05 * torch.randn(
        batch, heads, chunks, generator=generator
    )
    g_off_head = g_off.permute(0, 2, 4, 1, 3).reshape_as(g_diag)
    g_total = g_diag + g_off_head
    g_total[..., -1] += g_last
    dt = 0.2 * torch.randn(batch, seqlen, heads, generator=generator)
    a = -0.1 - 0.3 * torch.rand(heads, generator=generator)
    dt_bias = 0.1 * torch.randn(heads, generator=generator)
    gy = (0.1 * torch.randn(d_xdt.shape, generator=generator)).half()
    d_matrix = 0.2 * torch.randn(heads, headdim, generator=generator)

    common = tuple(
        value.npu()
        for value in (x, d_xdt, dt, a, gy, d_matrix, dt_bias)
    )
    x_npu, d_xdt_npu, dt_npu, a_npu, gy_npu, d_npu, bias_npu = common
    expected = torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_d_dd(
        x_npu,
        d_xdt_npu,
        g_total.contiguous().npu(),
        dt_npu,
        a_npu,
        gy_npu,
        d_npu,
        bias_npu,
        True,
        0.0,
        2.0,
    )
    actual = torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_grouped_d_dd(
        x_npu,
        d_xdt_npu,
        g_diag.npu(),
        g_off.npu(),
        g_last.npu(),
        dt_npu,
        a_npu,
        gy_npu,
        d_npu,
        bias_npu,
        True,
        0.0,
        2.0,
    )
    torch.npu.synchronize()
    for index, (got, want) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            got.cpu(), want.cpu(), rtol=2.0e-5, atol=2.0e-5,
            msg=f"output[{index}]",
        )


@pytest.mark.parametrize("batch,chunks,heads", [(1, 2, 4), (2, 4, 8)])
def test_fused_gcs_dt_matches_materialized_head_major(batch, chunks, heads):
    """Both experimental g_cs consumers must match materialized input."""
    chunk_size = 64
    headdim = 64
    seqlen = chunks * chunk_size
    generator = torch.Generator(device="cpu").manual_seed(20260809)
    x = 0.15 * torch.randn(
        batch, seqlen, heads, headdim, generator=generator
    )
    # The current hybrid Diag producer stores this internal workspace as FP16.
    d_xdt = (
        0.1
        * torch.randn(
            batch, heads, chunks, chunk_size, headdim, generator=generator
        )
    ).half()
    g_diag = 0.05 * torch.randn(
        batch, heads, chunks, chunk_size, generator=generator
    )
    g_off = 0.05 * torch.randn(
        batch, heads, chunks, chunk_size, generator=generator
    )
    g_last = 0.05 * torch.randn(
        batch, heads, chunks, generator=generator
    )
    g_total = g_diag + g_off
    g_total[..., -1] += g_last
    dt = 0.2 * torch.randn(batch, seqlen, heads, generator=generator)
    a = -0.1 - 0.3 * torch.rand(heads, generator=generator)
    dt_bias = 0.1 * torch.randn(heads, generator=generator)
    gy = (
        0.1
        * torch.randn(
            batch, heads, chunks, chunk_size, headdim, generator=generator
        )
    ).half()
    d_matrix = 0.2 * torch.randn(heads, headdim, generator=generator)

    x_npu, d_xdt_npu, dt_npu, a_npu, gy_npu, d_npu, bias_npu = (
        value.npu()
        for value in (x, d_xdt, dt, a, gy, d_matrix, dt_bias)
    )
    expected = torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_d_dd(
        x_npu,
        d_xdt_npu,
        g_total.contiguous().npu(),
        dt_npu,
        a_npu,
        gy_npu,
        d_npu,
        bias_npu,
        True,
        0.0,
        2.0,
    )
    actual = torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_fused_gcs_d_dd(
        x_npu,
        d_xdt_npu,
        g_diag.npu(),
        g_off.npu(),
        g_last.npu(),
        dt_npu,
        a_npu,
        gy_npu,
        d_npu,
        bias_npu,
        True,
        0.0,
        2.0,
    )
    actual_tail = torch.ops.mamba_ascend.mamba2_ssd_dt_bwd_chunk_tail_d_dd(
        x_npu,
        d_xdt_npu,
        (g_diag + g_off).contiguous().npu(),
        g_last.npu(),
        dt_npu,
        a_npu,
        gy_npu,
        d_npu,
        bias_npu,
        True,
        0.0,
        2.0,
    )
    torch.npu.synchronize()
    for candidate_name, candidate in (
        ("full_gcs", actual),
        ("chunk_tail", actual_tail),
    ):
        for index, (got, want) in enumerate(zip(candidate, expected)):
            torch.testing.assert_close(
                got.cpu(),
                want.cpu(),
                rtol=2.0e-5,
                atol=2.0e-5,
                msg=f"{candidate_name}.output[{index}]",
            )
