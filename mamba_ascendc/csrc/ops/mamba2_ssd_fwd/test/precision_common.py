"""Shared 45-case precision evaluation for Mamba2 SSD forward."""

import math

import torch

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


DTYPE = torch.float32
THRESHOLD = 2**-13

BASIC = "basic"
D_ONLY = "d_only"
Z_ONLY = "z_only"
INITIAL = "initial"
SOFTPLUS = "softplus"
ALL = "all"


def _six(shape, category):
    return [
        (category, shape, BASIC),
        (category, shape, D_ONLY),
        (category, shape, Z_ONLY),
        (category, shape, INITIAL),
        (category, shape, SOFTPLUS),
        (category, shape, ALL),
    ]


REGULAR_CASES = (
    _six((1, 32, 2, 8, 8, 16, 1), "small")
    + _six((1, 65, 4, 16, 16, 32, 2), "tail")
    + _six((2, 128, 8, 32, 32, 64, 2), "multi_group")
    + [
        ("chunk128", (1, 256, 8, 32, 64, 128, 2), BASIC),
        ("chunk128", (1, 256, 8, 32, 64, 128, 2), ALL),
        ("aligned", (1, 128, 2, 64, 64, 64, 1), BASIC),
        ("aligned", (1, 128, 2, 64, 64, 64, 1), ALL),
        ("aligned128", (1, 128, 2, 64, 64, 128, 1), BASIC),
        ("aligned128", (1, 128, 2, 64, 64, 128, 1), ALL),
        ("dstate128", (1, 128, 2, 64, 128, 128, 1), BASIC),
        ("dstate128", (1, 128, 2, 64, 128, 128, 1), ALL),
        ("single_token", (1, 1, 1, 8, 8, 16, 1), BASIC),
        ("odd_l", (1, 31, 2, 16, 16, 16, 1), ALL),
        ("head_group_ratio", (1, 96, 16, 16, 32, 32, 1), BASIC),
        ("many_groups", (1, 96, 16, 16, 32, 32, 8), ALL),
    ]
)

BOUNDARY_CASES = [
    "x_zero",
    "b_zero",
    "c_zero",
    "a_near_zero",
    "a_strong",
    "dt_softplus_negative",
    "dt_softplus_positive",
    "dt_limit_fixed",
    "z_negative",
    "z_zero",
    "z_positive",
    "initial_zero",
    "initial_one",
    "d_per_head",
    "d_per_channel",
]


def _feature_flags(feature):
    return {
        "D": feature in (D_ONLY, ALL),
        "z": feature in (Z_ONLY, ALL),
        "dt_bias": feature in (SOFTPLUS, ALL),
        "dt_softplus": feature in (SOFTPLUS, ALL),
        "initial": feature in (INITIAL, ALL),
    }


def make_case(case_id, category, shape, feature=BASIC, boundary=None):
    batch, seqlen, nheads, headdim, dstate, chunk_size, ngroups = shape
    generator = torch.Generator().manual_seed(20260801 + case_id)
    # The strict MARE suite uses a non-cancelling positive domain. MARE has a
    # fixed 1e-7 denominator and is intentionally evaluated away from random
    # signed cancellation; the separate basic suite retains signed stress data.
    x = 0.1 + 0.9 * torch.rand(batch, seqlen, nheads, headdim, generator=generator)
    dt = 0.01 + 0.1 * torch.rand(batch, seqlen, nheads, generator=generator)
    A = -(0.1 + 0.4 * torch.rand(nheads, generator=generator))
    B = 0.02 + 0.18 * torch.rand(batch, seqlen, ngroups, dstate, generator=generator)
    C = 0.02 + 0.18 * torch.rand(batch, seqlen, ngroups, dstate, generator=generator)

    flags = _feature_flags(feature)
    D = 0.1 + torch.rand(nheads, headdim, generator=generator) if flags["D"] else None
    z = 0.1 + torch.rand(batch, seqlen, nheads, headdim, generator=generator) if flags["z"] else None
    dt_bias = 0.01 * torch.rand(nheads, generator=generator) if flags["dt_bias"] else None
    initial = 0.1 + torch.rand(batch, nheads, headdim, dstate, generator=generator) if flags["initial"] else None
    dt_softplus = flags["dt_softplus"]
    dt_limit = (0.005, 0.2) if dt_softplus else (0.0, float("inf"))

    if boundary == "x_zero":
        x.zero_()
    elif boundary == "b_zero":
        B.zero_()
    elif boundary == "c_zero":
        C.zero_()
    elif boundary == "a_near_zero":
        A.fill_(-1e-4)
    elif boundary == "a_strong":
        A.fill_(-10.0)
    elif boundary in ("dt_softplus_negative", "dt_softplus_positive"):
        dt.fill_(-20.0 if boundary.endswith("negative") else 20.0)
        dt_softplus = True
        dt_limit = (0.0, 0.2)
    elif boundary == "dt_limit_fixed":
        dt_softplus = True
        dt_limit = (0.05, 0.05)
    elif boundary in ("z_negative", "z_zero", "z_positive"):
        z = torch.full_like(x, {"z_negative": -20.0, "z_zero": 0.0, "z_positive": 20.0}[boundary])
    elif boundary == "initial_zero":
        initial = torch.zeros(batch, nheads, headdim, dstate)
    elif boundary == "initial_one":
        initial = torch.ones(batch, nheads, headdim, dstate)
    elif boundary == "d_per_head":
        D = 0.1 + torch.rand(nheads, generator=generator)
    elif boundary == "d_per_channel":
        D = 0.1 + torch.rand(nheads, headdim, generator=generator)

    return {
        "case_id": case_id,
        "category": category,
        "description": boundary or feature,
        "shape": shape,
        "chunk_size": chunk_size,
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "z": z,
        "dt_bias": dt_bias,
        "initial_states": initial,
        "dt_softplus": dt_softplus,
        "dt_limit": dt_limit,
    }


def iter_case_specs():
    case_id = 0
    for category, shape, feature in REGULAR_CASES:
        case_id += 1
        yield case_id, category, shape, feature, None
    boundary_shape = (1, 32, 2, 8, 8, 16, 1)
    for boundary in BOUNDARY_CASES:
        case_id += 1
        yield case_id, "boundary", boundary_shape, BASIC, boundary


def _cpu_optional(value):
    return None if value is None else value.cpu()


def _npu_optional(value):
    return None if value is None else value.npu().contiguous()


def compute_metrics(actual, reference):
    actual = actual.cpu().float()
    reference = reference.cpu().float()
    abs_err = (actual - reference).abs()
    rel_err = abs_err / (reference.abs() + 1e-7)
    ref_norm = torch.linalg.vector_norm(reference)
    nrmse = torch.sqrt(torch.mean(abs_err.square())) / (ref_norm / math.sqrt(reference.numel()) + 1e-12)
    finite = torch.isfinite(actual)
    if torch.count_nonzero(actual).item() == 0 and torch.count_nonzero(reference).item() == 0:
        cosine = 1.0
    else:
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten().unsqueeze(0), reference.flatten().unsqueeze(0)
        ).item()
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "MERE": rel_err.mean().item(),
        "MARE": rel_err.max().item(),
        "NRMSE": nrmse.item(),
        "cosine_sim": cosine,
        "nan_inf": int((~finite).sum().item()),
    }


def evaluate_case(spec):
    case_id, category, shape, feature, boundary = spec
    case = make_case(case_id, category, shape, feature, boundary)
    args = (case["x"], case["dt"], case["A"], case["B"], case["C"])
    ref_out, ref_final = ssd_chunk_scan_ref(
        *args,
        case["chunk_size"],
        D=case["D"],
        z=case["z"],
        dt_bias=case["dt_bias"],
        dt_softplus=case["dt_softplus"],
        dt_limit=case["dt_limit"],
        initial_states=case["initial_states"],
        return_final_state=True,
    )
    npu_args = tuple(value.npu().contiguous() for value in args)
    out, final = ascend_kernel.mamba2_ssd_fwd(
        *npu_args,
        chunk_size=case["chunk_size"],
        D=_npu_optional(case["D"]),
        z=_npu_optional(case["z"]),
        dt_bias=_npu_optional(case["dt_bias"]),
        dt_softplus=case["dt_softplus"],
        dt_limit=case["dt_limit"],
        initial_states=_npu_optional(case["initial_states"]),
        return_final_state=True,
    )
    out_metrics = compute_metrics(out, ref_out)
    final_metrics = compute_metrics(final, ref_final)
    passed = all(
        metrics["MERE"] < THRESHOLD
        and metrics["MARE"] < 10 * THRESHOLD
        and metrics["nan_inf"] == 0
        for metrics in (out_metrics, final_metrics)
    )
    return {
        "case_id": case_id,
        "category": category,
        "description": case["description"],
        "shape": list(shape),
        "dtype": "float32",
        "out": out_metrics,
        "final_state": final_metrics,
        "threshold": THRESHOLD,
        "passed": passed,
    }
