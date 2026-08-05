"""Execute all precision cases and write JSON plus Markdown reports."""

import json
from pathlib import Path

from precision_common import THRESHOLD, evaluate_case, iter_case_specs


def main():
    results = []
    for spec in iter_case_specs():
        result = evaluate_case(spec)
        results.append(result)
        out = result["out"]
        final = result["final_state"]
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{status}] {result['case_id']:02d} "
            f"{result['category']}/{result['description']} "
            f"out(MERE={out['MERE']:.3e},MARE={out['MARE']:.3e}) "
            f"final(MERE={final['MERE']:.3e},MARE={final['MARE']:.3e})"
        )

    report_dir = Path(__file__).resolve().parent
    json_path = report_dir / "mamba2_ssd_fwd_precision_report.json"
    md_path = report_dir / "mamba2_ssd_fwd_precision_report.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    passed = sum(result["passed"] for result in results)
    failed = len(results) - passed
    regular = results[:30]
    boundary = results[30:]
    worst_out = max(results, key=lambda item: item["out"]["MARE"])
    worst_final = max(results, key=lambda item: item["final_state"]["MARE"])
    lines = [
        "# mamba2_ssd_fwd precision report",
        "",
        "## Overview",
        "",
        "| Dtype | Total | Passed | Failed | Pass rate | MERE limit | MARE limit |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| float32 | {len(results)} | {passed} | {failed} | {passed / len(results):.2%} | {THRESHOLD:.8e} | {10 * THRESHOLD:.8e} |",
        "",
        "Decision standard: `MERE < 2^-13` and `MARE < 10 * 2^-13` for both `out` and `final_state`; NaN/Inf count must be zero.",
        "",
        "## Category summary",
        "",
        "| Category | Total | Passed | Failed |",
        "|---|---:|---:|---:|",
        f"| shape/feature | {len(regular)} | {sum(x['passed'] for x in regular)} | {sum(not x['passed'] for x in regular)} |",
        f"| boundary | {len(boundary)} | {sum(x['passed'] for x in boundary)} | {sum(not x['passed'] for x in boundary)} |",
        "",
        "## Results",
        "",
        "| ID | Category | Description | Shape (B,L,H,P,N,C,G) | Out MERE | Out MARE | Final MERE | Final MARE | Result |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['case_id']:02d} | {result['category']} | {result['description']} | "
            f"`{result['shape']}` | {result['out']['MERE']:.3e} | "
            f"{result['out']['MARE']:.3e} | {result['final_state']['MERE']:.3e} | "
            f"{result['final_state']['MARE']:.3e} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Key findings",
        "",
        f"- Worst output MARE: case {worst_out['case_id']:02d} `{worst_out['category']}/{worst_out['description']}` = {worst_out['out']['MARE']:.3e}.",
        f"- Worst final-state MARE: case {worst_final['case_id']:02d} `{worst_final['category']}/{worst_final['description']}` = {worst_final['final_state']['MARE']:.3e}.",
        "- The same gate is applied to regular shapes, tails, grouping, all optional features and numerical boundary values; no case-specific threshold is used.",
        "- Strict MARE inputs use a positive non-cancelling domain because signed reductions can have arbitrarily large pointwise relative error near a zero crossing; the separate signed stress suite is judged with absolute/relative allclose and remains mandatory.",
        "- Key0 is the FP32 numerical oracle for later optimized tiling keys; performance changes must preserve this report's pass status.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"summary total={len(results)} passed={passed} failed={failed}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
