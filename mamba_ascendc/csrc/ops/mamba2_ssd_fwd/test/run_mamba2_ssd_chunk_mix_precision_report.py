"""Public-API precision gate restricted to the AscendC Cube/MIX path."""

from __future__ import annotations

import ctypes
import importlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


library_path = os.environ.get("MAMBA_CHUNK_MIX_OP_API_LIB")
if not library_path:
    raise RuntimeError("MAMBA_CHUNK_MIX_OP_API_LIB must point to libcust_opapi.so")
ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)

import ascend_kernel  # noqa: E402
from mamba_torch.ssd_reference import ssd_chunk_scan_ref  # noqa: E402


SHAPES = (
    (1, 64, 1, 64, 64, 64, 1),
    (1, 128, 2, 64, 64, 64, 1),
    (1, 128, 4, 64, 128, 64, 2),
    (2, 256, 8, 64, 64, 64, 2),
    (1, 256, 8, 64, 128, 128, 4),
    (2, 512, 8, 64, 128, 128, 2),
    (1, 512, 16, 64, 64, 64, 4),
    (1, 1024, 16, 64, 128, 128, 4),
)
VARIANTS = ("basic", "d_z", "initial", "all")
# These two cases cross the 512-task epilogue threshold and therefore verify
# the Cube/MIX core together with the native ACLNN epilogue fallback.
LARGE_CASES = (
    ((4, 2048, 16, 64, 128, 128, 4), "basic"),
    ((4, 2048, 16, 64, 128, 128, 4), "all"),
    # 4096 micro-chunk head tasks is the measured crossover for the fused
    # state-passing/Cube/epilogue kernel.  This case exercises that public
    # dispatch with initial state, D/z, bias and softplus all enabled.
    ((4, 4096, 16, 64, 128, 128, 4), "all"),
)

MAX_ABS_LIMIT = 5.0e-2
NRMSE_LIMIT = 5.0e-3
COSINE_LIMIT = 0.999


def _make_case(case_id: int, shape, variant: str):
    batch, seqlen, nheads, headdim, dstate, _, ngroups = shape
    generator = torch.Generator().manual_seed(20260802 + case_id)

    def randn(*dims):
        return torch.randn(*dims, generator=generator, dtype=torch.float32)

    x = randn(batch, seqlen, nheads, headdim)
    dt = 0.01 + 0.1 * torch.rand(
        batch, seqlen, nheads, generator=generator, dtype=torch.float32
    )
    a = -(0.1 + 0.4 * torch.rand(nheads, generator=generator))
    b = 0.2 * randn(batch, seqlen, ngroups, dstate)
    c = 0.2 * randn(batch, seqlen, ngroups, dstate)
    d = z = dt_bias = initial = None
    dt_softplus = False
    dt_limit = (0.0, float("inf"))
    if variant in ("d_z", "all"):
        d = randn(nheads, headdim)
        z = randn(batch, seqlen, nheads, headdim)
    if variant in ("initial", "all"):
        initial = 0.1 * randn(batch, nheads, headdim, dstate)
    if variant == "all":
        dt_bias = 0.1 * randn(nheads)
        dt_softplus = True
        dt_limit = (0.001, 2.0)

    def npu(value):
        return None if value is None else value.npu().contiguous()

    return {
        "x": npu(x),
        "dt": npu(dt),
        "A": npu(a),
        "B": npu(b),
        "C": npu(c),
        "D": npu(d),
        "z": npu(z),
        "dt_bias": npu(dt_bias),
        "initial_states": npu(initial),
        "dt_softplus": dt_softplus,
        "dt_limit": dt_limit,
    }


def _metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    finite = bool(torch.isfinite(actual).all().item())
    max_abs = error.abs().max().item()
    nrmse = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
    ).item()
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    return {
        "max_abs": max_abs,
        "mean_abs": error.abs().mean().item(),
        "nrmse": nrmse,
        "cosine": cosine,
        "finite": finite,
        "passed": (
            finite
            and max_abs <= MAX_ABS_LIMIT
            and nrmse <= NRMSE_LIMIT
            and cosine >= COSINE_LIMIT
        ),
    }


def _run(case_id: int, shape, variant: str):
    case = _make_case(case_id, shape, variant)
    chunk_size = shape[5]
    args = tuple(case[name] for name in ("x", "dt", "A", "B", "C"))
    kwargs = {
        "D": case["D"],
        "z": case["z"],
        "dt_bias": case["dt_bias"],
        "dt_softplus": case["dt_softplus"],
        "dt_limit": case["dt_limit"],
        "initial_states": case["initial_states"],
    }
    implementation = importlib.import_module("ascend_kernel.mamba2")
    if not implementation._can_use_chunk_mix_path(args[0], args[3], chunk_size):
        raise AssertionError(f"case {case_id} did not select the Cube/MIX path")
    with torch.no_grad():
        expected_out, expected_state = ssd_chunk_scan_ref(
            *args, chunk_size, return_final_state=True, **kwargs
        )
        actual_out, actual_state = ascend_kernel.mamba2_ssd_fwd(
            *args, chunk_size=chunk_size, return_final_state=True, **kwargs
        )
        torch.npu.synchronize()
    out_metrics = _metrics(actual_out, expected_out)
    state_metrics = _metrics(actual_state, expected_state)
    return {
        "case_id": case_id,
        "shape_b_l_h_p_n_c_g": list(shape),
        "variant": variant,
        "out": out_metrics,
        "final_state": state_metrics,
        "passed": out_metrics["passed"] and state_metrics["passed"],
    }


def main():
    if os.environ.get("MAMBA_ASCENDC_CHUNK_MIX", "0") != "1":
        raise RuntimeError("MAMBA_ASCENDC_CHUNK_MIX=1 is required")
    specs = [(shape, variant) for shape in SHAPES for variant in VARIANTS]
    specs.extend(LARGE_CASES)
    assert len(specs) == 35
    results = []
    for case_id, (shape, variant) in enumerate(specs, 1):
        result = _run(case_id, shape, variant)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {case_id:02d} {variant} {shape} "
            f"out(max={result['out']['max_abs']:.3e},"
            f"nrmse={result['out']['nrmse']:.3e},cos={result['out']['cosine']:.8f}) "
            f"state(max={result['final_state']['max_abs']:.3e},"
            f"nrmse={result['final_state']['nrmse']:.3e},"
            f"cos={result['final_state']['cosine']:.8f})",
            flush=True,
        )

    project_root = Path(__file__).resolve().parents[5]
    report_root = project_root / "docs" / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = Path(
        os.environ.get("MAMBA_CUBE_MIX_PRECISION_JSON")
        or os.environ.get("MAMBA_V2_PRECISION_JSON")
        or report_root / "mamba2_ascendc_cube_mix_precision.json"
    )
    md_path = Path(
        os.environ.get("MAMBA_CUBE_MIX_PRECISION_MD")
        or os.environ.get("MAMBA_V2_PRECISION_MD")
        or report_root / "mamba2_ascendc_cube_mix_precision.md"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    passed = sum(item["passed"] for item in results)
    lines = [
        "# Mamba2 AscendC Cube/MIX precision report",
        "",
        f"- Total: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {len(results) - passed}",
        f"- Gate: max abs <= `{MAX_ABS_LIMIT}`, NRMSE <= `{NRMSE_LIMIT}`, "
        f"cosine >= `{COSINE_LIMIT}`, finite.",
        "- Every case asserts that public dispatch selected the Cube/MIX path.",
        "",
        "| ID | Shape `[B,L,H,P,N,C,G]` | Variant | Out max | Out NRMSE | "
        "Out cosine | State max | State NRMSE | State cosine | Result |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['case_id']} | `{item['shape_b_l_h_p_n_c_g']}` | "
            f"{item['variant']} | {item['out']['max_abs']:.3e} | "
            f"{item['out']['nrmse']:.3e} | {item['out']['cosine']:.8f} | "
            f"{item['final_state']['max_abs']:.3e} | "
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
