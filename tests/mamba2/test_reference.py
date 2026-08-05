# Copyright (c) 2026, mamba-triton-ascend authors.
# Tests for Mamba-2 SSD PyTorch reference implementation.

import torch
import torch.nn.functional as F
from mamba_torch.ssd_reference import segsum, _ssd_core, ssd_chunk_scan_ref


# ---------------------------------------------------------------------------
# segsum
# ---------------------------------------------------------------------------

def test_segsum():
    """segsum helper correctness: diagonal==0, lower-tri > 0, upper-tri = -inf."""
    x = torch.tensor([1.0, 2.0, 3.0])
    s = segsum(x)
    L = torch.exp(s)
    assert L.shape == (3, 3)
    assert abs(L[0, 0].item() - 1.0) < 1e-6          # exp(0) = 1
    assert L[1, 0].item() > 1.0                       # exp(sum 2) ≈ 7.4
    assert L[2, 1].item() > 1.0                       # exp(sum 3) ≈ 20.1
    assert L[0, 1].item() == 0.0                       # upper-tri → 0
    assert L[0, 2].item() == 0.0                       # upper-tri → 0


def test_segsum_batched():
    """segsum with batched input (batch, 1, chunk_size)."""
    x = torch.randn(2, 3, 8)
    s = segsum(x)
    L = torch.exp(s)
    assert L.shape == (2, 3, 8, 8)
    # Diagonal = 1
    for i in range(8):
        assert torch.allclose(L[:, :, i, i], torch.ones(2, 3), atol=1e-6)
    # Upper-tri = 0
    assert (L[:, :, 1, 2] == 0.0).all()
    # Lower-tri > 0
    assert (L[:, :, -1, 0] > 0.0).all()


# ---------------------------------------------------------------------------
# _ssd_core
# ---------------------------------------------------------------------------

def test_ssd_core_vs_sequential():
    """_ssd_core (per-head, no dt/groups) vs step-by-step sequential scan."""
    batch, seqlen, nheads, headdim, dstate = 1, 32, 2, 4, 8
    chunk_size = 8
    torch.manual_seed(42)

    # _ssd_core expects: x already * dt, A = dA = dt * A
    # A is the negative decay value, NOT log-space
    A = torch.log(torch.tensor(0.5)).expand(nheads)  # = log(0.5) ≈ -0.693
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads)     # dt > 0, stable
    x_raw = torch.randn(batch, seqlen, nheads, headdim)
    x_scaled = x_raw * dt.unsqueeze(-1)                      # x * dt
    dA = dt * A.view(1, 1, -1)                               # dA = dt * A

    B_h = torch.randn(batch, seqlen, nheads, dstate)
    C_h = torch.randn(batch, seqlen, nheads, dstate)

    # SSD chunked
    y_ssd, final_ssd = _ssd_core(x_scaled, dA, B_h, C_h, chunk_size)

    # Sequential scan (x_scaled already has dt baked in, so no dt factor in dBx)
    h = torch.zeros(batch, nheads, headdim, dstate)
    ys = []
    for t in range(seqlen):
        dA_exp = torch.exp(dA[:, t, :]).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        xt = x_scaled[:, t]                                            # (B, H, P)
        Bt = B_h[:, t]                                                 # (B, H, N)
        Ct = C_h[:, t]                                                 # (B, H, N)
        dBx = torch.einsum("bhn, bhp -> bhpn", Bt, xt)
        h   = h * dA_exp + dBx
        yt  = torch.einsum("bhpn, bhn -> bhp", h, Ct)
        ys.append(yt)
    y_seq = torch.stack(ys, dim=1)

    abs_diff = (y_ssd - y_seq).abs()
    assert abs_diff.max() < 1e-3, f"max error={abs_diff.max():.2e}"


def test_ssd_core_batched():
    """_ssd_core with batch > 1."""
    batch, seqlen, nheads, headdim, dstate = 3, 16, 2, 4, 8
    chunk_size = 4
    torch.manual_seed(42)

    x  = torch.randn(batch, seqlen, nheads, headdim)
    dA = -0.1 * torch.rand(batch, seqlen, nheads)  # negative A for stability
    B_h = torch.randn(batch, seqlen, nheads, dstate)
    C_h = torch.randn(batch, seqlen, nheads, dstate)

    y, final = _ssd_core(x, dA, B_h, C_h, chunk_size)
    assert y.shape == (batch, seqlen, nheads, headdim)
    assert final.shape == (batch, 1, nheads, headdim, dstate)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


# ---------------------------------------------------------------------------
# ssd_chunk_scan_ref (full interface)
# ---------------------------------------------------------------------------

def test_ssd_chunk_scan_basic():
    """Full interface: basic shape + no NaN/Inf."""
    batch, seqlen, nheads, headdim, dstate = 2, 32, 4, 16, 8
    chunk_size = 16
    ngroups = 1
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads)
    A_log = torch.log(torch.tensor(0.5)).repeat(nheads)
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    out = ssd_chunk_scan_ref(x, dt, A_log, B, C, chunk_size, dt_softplus=True)
    assert out.shape == (batch, seqlen, nheads, headdim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_ssd_vs_sequential():
    """Full SSD vs step-by-step sequential scan (ngroups=1)."""
    batch, seqlen, nheads, headdim, dstate = 1, 32, 2, 4, 8
    chunk_size = 8
    ngroups = 1
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads)
    A_log = torch.log(torch.tensor(0.5)).repeat(nheads)
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    out_ssd = ssd_chunk_scan_ref(x, dt, A_log, B, C, chunk_size, dt_softplus=True)

    # Sequential reference (A is the decay value, NOT log-space, used directly)
    dt_f = F.softplus(dt.float())
    h = torch.zeros(batch, nheads, headdim, dstate, dtype=torch.float32)
    ys = []
    for t in range(seqlen):
        dt_t = dt_f[:, t, :]
        dA_t = torch.exp(dt_t * A_log).unsqueeze(-1).unsqueeze(-1)
        B_t = B[:, t, 0, :]
        x_t = x[:, t]
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt_t, B_t, x_t)
        h = h * dA_t + dBx
        C_t = C[:, t, 0, :]
        y_t = torch.einsum("bhpn, bn -> bhp", h, C_t)
        ys.append(y_t)
    out_seq = torch.stack(ys, dim=1)

    abs_diff = (out_ssd.float() - out_seq.float()).abs()
    assert abs_diff.max() < 1e-3, f"max error={abs_diff.max():.2e}"


def test_ssd_with_D_z_dt_bias():
    """Full features: D skip, z gating, dt_bias."""
    batch, seqlen, nheads, headdim, dstate = 2, 32, 4, 16, 8
    chunk_size = 16
    ngroups = 1
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads)
    A_log = torch.log(torch.tensor(0.5)).repeat(nheads)
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)
    D = torch.randn(nheads, headdim)
    z = torch.randn(batch, seqlen, nheads, headdim)
    dt_bias = 0.1 * torch.randn(nheads)

    out = ssd_chunk_scan_ref(
        x, dt, A_log, B, C, chunk_size,
        D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )
    assert out.shape == (batch, seqlen, nheads, headdim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_ssd_with_D_1d():
    """D as 1D vector (nheads,)."""
    batch, seqlen, nheads, headdim, dstate = 1, 16, 2, 4, 8
    chunk_size = 8
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads)
    A_log = torch.log(torch.tensor(0.5)).repeat(nheads)
    B = torch.randn(batch, seqlen, 1, dstate)
    C = torch.randn(batch, seqlen, 1, dstate)
    D_1d = torch.randn(nheads)

    out = ssd_chunk_scan_ref(x, dt, A_log, B, C, chunk_size, D=D_1d)
    assert out.shape == (batch, seqlen, nheads, headdim)
    assert not torch.isnan(out).any()


def test_ssd_chunk_boundary():
    """Non-multiple chunk_size (auto-padding test)."""
    batch, seqlen, nheads, headdim, dstate = 2, 30, 2, 4, 8
    chunk_size = 7
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads)
    A_log = torch.log(torch.tensor(0.5)).repeat(nheads)
    B = torch.randn(batch, seqlen, 1, dstate)
    C = torch.randn(batch, seqlen, 1, dstate)

    out = ssd_chunk_scan_ref(x, dt, A_log, B, C, chunk_size, dt_softplus=True)
    assert out.shape == (batch, seqlen, nheads, headdim)
    assert not torch.isnan(out).any()


def test_ssd_ngroups_2():
    """ngroups=2 (each group shares B/C among heads)."""
    batch, seqlen, nheads, headdim, dstate = 1, 16, 4, 4, 8
    chunk_size = 8
    ngroups = 2
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads)
    A_log = torch.log(torch.tensor(0.5)).repeat(nheads)
    B = torch.randn(batch, seqlen, ngroups, dstate)
    C = torch.randn(batch, seqlen, ngroups, dstate)

    out = ssd_chunk_scan_ref(x, dt, A_log, B, C, chunk_size, dt_softplus=True)
    assert out.shape == (batch, seqlen, nheads, headdim)

    # Sequential reference for ngroups=2 (A is decay value, NOT log-space)
    nheads_per_group = nheads // ngroups
    dt_f = F.softplus(dt.float())
    h = torch.zeros(batch, nheads, headdim, dstate, dtype=torch.float32)
    ys = []
    for t in range(seqlen):
        dt_t = dt_f[:, t, :]
        dA_t = torch.exp(dt_t * A_log).unsqueeze(-1).unsqueeze(-1)
        y_heads = []
        for g in range(ngroups):
            hs = g * nheads_per_group
            he = hs + nheads_per_group
            B_tg = B[:, t, g, :]
            x_tg = x[:, t, hs:he, :]
            dBx = torch.einsum("bh, bn, bhp -> bhpn", dt_t[:, hs:he], B_tg, x_tg)
            h[:, hs:he] = h[:, hs:he] * dA_t[:, hs:he] + dBx
            C_tg = C[:, t, g, :]
            y_tg = torch.einsum("bhpn, bn -> bhp", h[:, hs:he], C_tg)
            y_heads.append(y_tg)
        ys.append(torch.cat(y_heads, dim=1))
    out_seq = torch.stack(ys, dim=1)

    abs_diff = (out.float() - out_seq.float()).abs()
    assert abs_diff.max() < 1e-3, f"ngroups=2 max error={abs_diff.max():.2e}"


def test_ssd_return_final_state():
    """return_final_state flag returns last chunk state."""
    batch, seqlen, nheads, headdim, dstate = 2, 32, 4, 16, 8
    chunk_size = 16
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads)
    A_log = torch.log(torch.tensor(0.5)).repeat(nheads)
    B = torch.randn(batch, seqlen, 1, dstate)
    C = torch.randn(batch, seqlen, 1, dstate)

    out, final = ssd_chunk_scan_ref(
        x, dt, A_log, B, C, chunk_size,
        dt_softplus=True, return_final_state=True
    )
    assert out.shape == (batch, seqlen, nheads, headdim)
    assert final.shape == (batch, nheads, headdim, dstate)
    assert not torch.isnan(final).any()


def test_ssd_dtype_preservation():
    """Output dtype matches input dtype."""
    batch, seqlen, nheads, headdim, dstate = 1, 16, 2, 4, 8
    chunk_size = 8
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.bfloat16)
    dt = 0.1 + 0.1 * torch.rand(batch, seqlen, nheads, dtype=torch.bfloat16)
    A_log = torch.log(torch.tensor(0.5, dtype=torch.float32)).repeat(nheads)
    B = torch.randn(batch, seqlen, 1, dstate, dtype=torch.bfloat16)
    C = torch.randn(batch, seqlen, 1, dstate, dtype=torch.bfloat16)

    out = ssd_chunk_scan_ref(x, dt, A_log, B, C, chunk_size, dt_softplus=True)
    assert out.dtype == torch.bfloat16
    assert out.shape == (batch, seqlen, nheads, headdim)


def test_ssd_autograd():
    """Verify backward through reference works."""
    batch, seqlen, nheads, headdim, dstate = 1, 16, 2, 4, 8
    chunk_size = 8
    torch.manual_seed(42)

    x = torch.randn(batch, seqlen, nheads, headdim, requires_grad=True)
    dt = torch.rand(batch, seqlen, nheads) * 0.1 + 0.1
    dt.requires_grad_(True)
    A_log = torch.log(torch.tensor(0.5)).repeat(nheads).requires_grad_(True)
    B = torch.randn(batch, seqlen, 1, dstate, requires_grad=True)
    C = torch.randn(batch, seqlen, 1, dstate, requires_grad=True)

    out = ssd_chunk_scan_ref(x, dt, A_log, B, C, chunk_size, dt_softplus=True)
    loss = out.sum()
    loss.backward()

    for name, p in [("x", x), ("dt", dt), ("A_log", A_log), ("B", B), ("C", C)]:
        assert p.grad is not None, f"{name} grad is None"
        assert not torch.isnan(p.grad).any(), f"{name} grad has NaN"
        assert not torch.isinf(p.grad).any(), f"{name} grad has Inf"
