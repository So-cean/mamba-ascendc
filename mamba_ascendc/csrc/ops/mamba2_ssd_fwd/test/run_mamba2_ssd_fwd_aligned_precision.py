"""Evaluate the aligned-16 Cube+Vector path with mixed-precision gates."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


SHAPES = [
    (1, 64, 1, 16, 16, 16, 1),
    (1, 64, 2, 32, 16, 32, 1),
    (1, 128, 2, 64, 64, 64, 1),
    (2, 128, 4, 32, 32, 64, 2),
    (1, 256, 4, 64, 64, 64, 1),
    (1, 256, 8, 32, 64, 128, 2),
    (2, 256, 8, 64, 64, 64, 4),
    (1, 512, 8, 64, 64, 64, 2),
    (1, 512, 16, 32, 128, 128, 4),
]
VARIANTS = ("basic", "initial", "all")
TARGET_MEDIUM = (4, 2048, 16, 64, 128, 128, 4)
TARGET_EXTREME = (8, 4096, 32, 64, 128, 128, 8)
OUT_NRMSE_LIMIT = 5e-3
STATE_NRMSE_LIMIT = 7e-3
COSINE_LIMIT = 0.999
RTOL = 3e-2
ATOL = 1e-2


def _make_case(case_id, shape, variant):
    batch, seqlen, nheads, headdim, dstate, chunk_size, ngroups = shape
    generator = torch.Generator().manual_seed(20260801 + case_id)

    def randn(*dims):
        return torch.randn(*dims, generator=generator, dtype=torch.float32)

    x = randn(batch, seqlen, nheads, headdim)
    dt = 0.01 + 0.1 * torch.rand(
        batch, seqlen, nheads, generator=generator, dtype=torch.float32
    )
    A = -(0.1 + 0.4 * torch.rand(nheads, generator=generator))
    B = randn(batch, seqlen, ngroups, dstate) * 0.2
    C = randn(batch, seqlen, ngroups, dstate) * 0.2
    initial = None
    D = None
    z = None
    dt_bias = None
    dt_softplus = False
    dt_limit = (0.0, float("inf"))
    if variant in ("initial", "all"):
        initial = randn(batch, nheads, headdim, dstate) * 0.1
    if variant == "all":
        D = randn(nheads, headdim)
        z = randn(batch, seqlen, nheads, headdim)
        dt_bias = randn(nheads) * 0.1
        dt_softplus = True
        dt_limit = (0.001, 2.0)
    return {
        "shape": shape,
        "variant": variant,
        "x": x.npu(),
        "dt": dt.npu(),
        "A": A.npu(),
        "B": B.npu(),
        "C": C.npu(),
        "D": None if D is None else D.npu(),
        "z": None if z is None else z.npu(),
        "dt_bias": None if dt_bias is None else dt_bias.npu(),
        "initial_states": None if initial is None else initial.npu(),
        "dt_softplus": dt_softplus,
        "dt_limit": dt_limit,
    }


def _metrics(actual, expected, nrmse_limit):
    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    finite = bool(torch.isfinite(actual).all().item())
    nrmse = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    ).item()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    abs_error = error.abs()
    allclose = bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL))
    return {
        "max_abs": abs_error.max().item(),
        "mean_abs": abs_error.mean().item(),
        "nrmse": nrmse,
        "cosine": cosine,
        "allclose": allclose,
        "finite": finite,
        "passed": finite and allclose and nrmse <= nrmse_limit and cosine >= COSINE_LIMIT,
    }


def _specs():
    specs = [(shape, variant) for shape in SHAPES for variant in VARIANTS]
    specs.extend(
        [
            (TARGET_MEDIUM, "basic"),
            ((2, 512, 8, 64, 64, 64, 1), "all"),
            ((1, 1024, 16, 64, 64, 128, 4), "initial"),
            (TARGET_EXTREME, "all"),
        ]
    )
    assert len(specs) == 31
    return specs


def _run(case_id, shape, variant):
    case = _make_case(case_id, shape, variant)
    args = tuple(case[key] for key in ("x", "dt", "A", "B", "C"))
    kwargs = {
        "D": case["D"],
        "z": case["z"],
        "dt_bias": case["dt_bias"],
        "dt_softplus": case["dt_softplus"],
        "dt_limit": case["dt_limit"],
        "initial_states": case["initial_states"],
    }
    with torch.no_grad():
        ref_out, ref_state = ssd_chunk_scan_ref(
            *args, shape[5], return_final_state=True, **kwargs
        )
        out, state = ascend_kernel.mamba2_ssd_fwd(
            *args, chunk_size=shape[5], return_final_state=True, **kwargs
        )
        torch.npu.synchronize()
    out_metrics = _metrics(out, ref_out, OUT_NRMSE_LIMIT)
    state_metrics = _metrics(state, ref_state, STATE_NRMSE_LIMIT)
    return {
        "case_id": case_id,
        "shape": list(shape),
        "variant": variant,
        "out": out_metrics,
        "final_state": state_metrics,
        "passed": out_metrics["passed"] and state_metrics["passed"],
    }


def main():
    results = []
    for case_id, (shape, variant) in enumerate(_specs(), 1):
        result = _run(case_id, shape, variant)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {case_id:02d} {variant} {shape} "
            f"out(nrmse={result['out']['nrmse']:.3e},cos={result['out']['cosine']:.8f}) "
            f"state(nrmse={result['final_state']['nrmse']:.3e},"
            f"cos={result['final_state']['cosine']:.8f})",
            flush=True,
        )

    root = Path(__file__).resolve().parents[5]
    report_root = root / "docs" / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = Path(os.environ.get(
        "MAMBA_ALIGNED_PRECISION_JSON",
        report_root / "mamba2_ascendc_aligned16_precision_2026-08-01.json",
    ))
    md_path = Path(os.environ.get(
        "MAMBA_ALIGNED_PRECISION_MD",
        report_root / "mamba2_ascendc_aligned16_precision_2026-08-01.md",
    ))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    passed = sum(item["passed"] for item in results)
    lines = [
        "# Mamba2 AscendC aligned-16 mixed-precision report",
        "",
        f"- Total: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {len(results) - passed}",
        f"- Gate: `rtol={RTOL}`, `atol={ATOL}`, out NRMSE <= `{OUT_NRMSE_LIMIT}`, "
        f"final-state NRMSE <= `{STATE_NRMSE_LIMIT}`, cosine >= `{COSINE_LIMIT}`, finite.",
        "",
        "| ID | Shape `[B,L,H,P,N,C,G]` | Variant | Out NRMSE | Out cosine | State NRMSE | State cosine | Result |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['case_id']} | `{item['shape']}` | {item['variant']} | "
            f"{item['out']['nrmse']:.3e} | {item['out']['cosine']:.8f} | "
            f"{item['final_state']['nrmse']:.3e} | "
            f"{item['final_state']['cosine']:.8f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"summary total={len(results)} passed={passed} failed={len(results) - passed}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
