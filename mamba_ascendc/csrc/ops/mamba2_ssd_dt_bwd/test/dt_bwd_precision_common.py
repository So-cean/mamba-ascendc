"""Shared deterministic cases and metrics for Mamba2SsdDtBwd."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

THRESHOLD = 2**-13

SHAPES = [
    ("Single", "one stream one T64 chunk", (1, 64, 1, 64, 64)),
    ("Heads", "two heads two T64 chunks", (1, 128, 2, 64, 64)),
    ("T128", "three heads one T128 chunk", (1, 128, 3, 64, 128)),
    ("Batch", "batch2 four heads T64", (2, 256, 4, 64, 64)),
    ("Scale", "batch2 eight heads T128", (2, 512, 8, 64, 128)),
    ("Scale", "batch4 four heads T128", (4, 256, 4, 64, 128)),
    ("Chunks", "eight T64 chunks", (2, 512, 4, 64, 64)),
    ("Occupancy", "batch4 eight heads", (4, 128, 8, 64, 128)),
]

FEATURES = [
    ("plain_positive_dt", False, False, (0.0, torch.finfo(torch.float32).max)),
    ("bias_finite_clamp", True, False, (-0.2, 0.3)),
    ("bias_softplus", True, True, (0.0, torch.finfo(torch.float32).max)),
    ("bias_softplus_finite_clamp", True, True, (0.02, 0.5)),
]

DIRECTED = [
    "raw_at_min",
    "raw_at_max",
    "raw_one_ulp_outside",
    "softplus_neg30",
    "softplus_pos30",
    "a_zero",
    "gcs_first_token",
    "gcs_last_token_k4",
]


@dataclass(frozen=True)
class Case:
    case_id: int
    category: str
    description: str
    shape: tuple[int, int, int, int, int]
    feature: str
    use_bias: bool
    softplus: bool
    limits: tuple[float, float]
    directed: str | None = None


def all_cases() -> list[Case]:
    cases: list[Case] = []
    case_id = 0
    for category, description, shape in SHAPES:
        for feature, use_bias, softplus, limits in FEATURES:
            case_id += 1
            cases.append(
                Case(
                    case_id,
                    category,
                    description,
                    shape,
                    feature,
                    use_bias,
                    softplus,
                    limits,
                )
            )
    for directed in DIRECTED:
        case_id += 1
        shape = (1, 256, 2, 64, 64) if directed == "gcs_last_token_k4" else (1, 64, 2, 64, 64)
        softplus = directed.startswith("softplus")
        cases.append(
            Case(
                case_id,
                "Boundary",
                directed,
                shape,
                directed,
                True,
                softplus,
                (-0.2, 0.3) if not softplus else (0.0, torch.finfo(torch.float32).max),
                directed,
            )
        )
    return cases


def make_inputs(case: Case):
    batch, seqlen, heads, headdim, chunk_size = case.shape
    chunks = seqlen // chunk_size
    generator = torch.Generator(device="cpu").manual_seed(20260807 + case.case_id)
    x = 0.15 + 0.05 * torch.rand(
        batch, seqlen, heads, headdim, generator=generator
    )
    d_xdt = 0.08 + 0.04 * torch.rand(
        batch, heads, chunks, chunk_size, headdim, generator=generator
    )
    g_cs = 0.001 + 0.002 * torch.rand(
        batch, heads, chunks, chunk_size, generator=generator
    )
    a = -0.05 - 0.05 * torch.rand(heads, generator=generator)
    bias = torch.linspace(-0.04, 0.04, heads) if case.use_bias else None

    if case.feature == "plain_positive_dt":
        dt = 0.05 + 0.3 * torch.rand(batch, seqlen, heads, generator=generator)
    elif case.feature == "bias_finite_clamp":
        base = torch.linspace(-0.5, 0.55, seqlen).view(1, seqlen, 1)
        dt = base.expand(batch, seqlen, heads).contiguous()
    elif case.feature == "bias_softplus":
        base = torch.linspace(-4.0, 4.0, seqlen).view(1, seqlen, 1)
        dt = base.expand(batch, seqlen, heads).contiguous()
    elif case.feature == "bias_softplus_finite_clamp":
        base = torch.linspace(-7.0, 1.0, seqlen).view(1, seqlen, 1)
        dt = base.expand(batch, seqlen, heads).contiguous()
    else:
        dt = torch.full((batch, seqlen, heads), 0.1)

    if case.directed == "raw_at_min":
        dt.fill_(case.limits[0])
        bias.zero_()
    elif case.directed == "raw_at_max":
        dt.fill_(case.limits[1])
        bias.zero_()
    elif case.directed == "raw_one_ulp_outside":
        low = torch.nextafter(
            torch.tensor(case.limits[0]), torch.tensor(float("-inf"))
        )
        high = torch.nextafter(
            torch.tensor(case.limits[1]), torch.tensor(float("inf"))
        )
        dt[:, 0::2].fill_(low.item())
        dt[:, 1::2].fill_(high.item())
        bias.zero_()
    elif case.directed == "softplus_neg30":
        dt.fill_(-30.0)
        bias.zero_()
    elif case.directed == "softplus_pos30":
        dt.fill_(30.0)
        bias.zero_()
    elif case.directed == "a_zero":
        a.zero_()
    elif case.directed == "gcs_first_token":
        g_cs.zero_()
        g_cs[..., 0] = 0.1
    elif case.directed == "gcs_last_token_k4":
        g_cs.zero_()
        g_cs[..., -1] = 0.1

    return x, d_xdt, g_cs, dt, a, bias


def reference(x, d_xdt, g_cs, dt, a, bias, softplus, limits):
    batch, heads, chunks, chunk_size, headdim = d_xdt.shape
    u = dt + (0.0 if bias is None else bias)
    q_pre = F.softplus(u) if softplus else u
    q = q_pre.clamp(min=limits[0], max=limits[1])
    q_h = q.permute(0, 2, 1).reshape(batch, heads, chunks, chunk_size)
    x_h = x.permute(0, 2, 1, 3).reshape_as(d_xdt)
    gw = torch.flip(
        torch.cumsum(torch.flip(g_cs, dims=(-1,)), dim=-1), dims=(-1,)
    )
    dq = (d_xdt * x_h).sum(dim=-1) + gw * a[None, :, None, None]
    clamp_grad = ((q_pre >= limits[0]) & (q_pre <= limits[1])).to(dt.dtype)
    if softplus:
        clamp_grad = clamp_grad * torch.sigmoid(u)
    ddt_h = dq * clamp_grad.permute(0, 2, 1).reshape_as(dq)
    dx = (d_xdt * q_h[..., None]).reshape(
        batch, heads, chunks * chunk_size, headdim
    ).permute(0, 2, 1, 3)
    ddt = ddt_h.reshape(batch, heads, chunks * chunk_size).permute(0, 2, 1)
    d_a = (gw * q_h).sum(dim=(0, 2, 3))
    d_bias = ddt.sum(dim=(0, 1))
    return dx, ddt, d_a, d_bias


def npu_call(case: Case, inputs):
    x, d_xdt, g_cs, dt, a, bias = inputs
    # Keep every asynchronous H2D input alive until the custom kernel has
    # completed.  Inline `.npu()` temporaries can otherwise be released before
    # mssanitizer observes the device read.
    x_npu = x.npu()
    d_xdt_npu = d_xdt.npu()
    g_cs_npu = g_cs.npu()
    dt_npu = dt.npu()
    a_npu = a.npu()
    bias_npu = None if bias is None else bias.npu()
    result = torch.ops.mamba_ascend.mamba2_ssd_dt_bwd(
        x_npu,
        d_xdt_npu,
        g_cs_npu,
        dt_npu,
        a_npu,
        bias_npu,
        case.softplus,
        case.limits[0],
        case.limits[1],
    )
    torch.npu.synchronize()
    return tuple(value.cpu() for value in result)


def metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    abs_err = (actual - expected).abs()
    rel_err = abs_err / (expected.abs() + 1.0e-7)
    if actual.numel() == 0:
        cosine = 1.0
    elif actual.count_nonzero() == 0 and expected.count_nonzero() == 0:
        cosine = 1.0
    else:
        cosine = F.cosine_similarity(
            actual.flatten().unsqueeze(0), expected.flatten().unsqueeze(0)
        ).item()
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "MARE": rel_err.max().item(),
        "MERE": rel_err.mean().item(),
        "cosine_sim": cosine,
    }


def evaluate(case: Case):
    inputs = make_inputs(case)
    expected = reference(*inputs, case.softplus, case.limits)
    actual = npu_call(case, inputs)
    names = ("dx", "ddt", "dA", "dt_bias")
    output_metrics = {
        name: metrics(got, want)
        for name, got, want in zip(names, actual, expected)
    }
    passed = all(
        values["MERE"] < THRESHOLD and values["MARE"] < 10 * THRESHOLD
        for values in output_metrics.values()
    )
    return output_metrics, passed
