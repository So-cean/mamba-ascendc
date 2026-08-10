"""Run the 32-case Mamba2 SSD backward M0 precision suite."""

from __future__ import annotations

import json
from pathlib import Path

from precision_common import GRAD_NAMES, THRESHOLD, evaluate_case, iter_case_specs


def main():
    results = []
    for spec in iter_case_specs():
        result = evaluate_case(spec)
        results.append(result)
        worst = max(result["metrics"].items(), key=lambda item: item[1]["MARE"])
        print(
            f"[{'PASS' if result['passed'] else 'FAIL'}] "
            f"{result['case_id']:02d} {result['category']}/{result['grad_mode']} "
            f"worst={worst[0]} MERE={worst[1]['MERE']:.3e} "
            f"MARE={worst[1]['MARE']:.3e}"
        )

    report_dir = Path(__file__).resolve().parent
    json_path = report_dir / "mamba2_ssd_bwd_precision_report.json"
    md_path = report_dir / "mamba2_ssd_bwd_precision_report.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    passed = sum(item["passed"] for item in results)
    worst_entries = {
        name: max(results, key=lambda item: item["metrics"][name]["MARE"])
        for name in GRAD_NAMES
    }
    lines = [
        "# mamba2_ssd_bwd M0 precision report",
        "",
        "## Overview",
        "",
        "| Dtype | Total | Passed | Failed | Pass rate | MERE limit | MARE limit |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| float32 | {len(results)} | {passed} | {len(results) - passed} | "
        f"{passed / len(results):.2%} | {THRESHOLD:.8e} | {10 * THRESHOLD:.8e} |",
        "",
        "Decision standard: every returned gradient must satisfy "
        "`MERE < 2^-13`, `MARE < 10 * 2^-13`, and contain no NaN/Inf.",
        "",
        "The suite covers the currently documented native M0 domain only: "
        "FP32, `P=N=chunk=64`, `L % 64 == 0`, default `dt_limit`, and no optional tensors.",
        "",
        "## Results",
        "",
        "| ID | Shape (B,L,H,P,N,C,G) | Gradient path | Worst tensor | MERE | MARE | NRMSE | Cosine | Result |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        worst_name, worst = max(
            result["metrics"].items(), key=lambda item: item[1]["MARE"]
        )
        lines.append(
            f"| {result['case_id']:02d} | `{result['shape']}` | "
            f"{result['grad_mode']} | {worst_name} | {worst['MERE']:.3e} | "
            f"{worst['MARE']:.3e} | {worst['NRMSE']:.3e} | "
            f"{worst['cosine_sim']:.6f} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Worst case by returned gradient",
        "",
        "| Gradient | Case | Gradient path | MERE | MARE | NRMSE | Cosine |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, result in worst_entries.items():
        value = result["metrics"][name]
        lines.append(
            f"| {name} | {result['case_id']:02d} | {result['grad_mode']} | "
            f"{value['MERE']:.3e} | {value['MARE']:.3e} | "
            f"{value['NRMSE']:.3e} | {value['cosine_sim']:.6f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- The test invokes the public `ascend_kernel.mamba2_ssd_fwd` API and PyTorch autograd, not the internal backward operator directly.",
        "- `final_only` cases verify the `dout=None` autograd path.",
        "- Shared-group shapes verify the required cross-head reductions for `dB` and `dC`.",
        "- Positive non-cancelling inputs and upstream gradients make strict relative-error metrics stable; signed random stress remains covered by the pytest gate.",
        "- This is a correctness milestone. M1 replaces the direct reverse recurrence with chunk/Cube decomposition before performance claims are made.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"summary total={len(results)} passed={passed} failed={len(results) - passed}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
