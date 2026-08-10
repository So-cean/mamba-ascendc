#!/usr/bin/env python3
"""Run Mamba2SsdDtBwd precision cases and emit JSON/Markdown reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch

import ascend_kernel  # noqa: F401

from dt_bwd_precision_common import THRESHOLD, all_cases, evaluate


def main() -> int:
    rows = []
    for case in all_cases():
        output_metrics, passed = evaluate(case)
        worst_name, worst = max(
            output_metrics.items(), key=lambda item: item[1]["MARE"]
        )
        row = {
            "case_id": case.case_id,
            "category": case.category,
            "description": case.description,
            "shape": list(case.shape),
            "feature": case.feature,
            "dtype": "float32",
            "outputs": output_metrics,
            "worst_output": worst_name,
            "worst_MERE": max(v["MERE"] for v in output_metrics.values()),
            "worst_MARE": worst["MARE"],
            "passed": passed,
        }
        rows.append(row)
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {case.case_id:02d} {case.feature} {case.shape} "
            f"worst={worst_name} MERE={row['worst_MERE']:.3e} "
            f"MARE={row['worst_MARE']:.3e}"
        )

    report_dir = Path(__file__).resolve().parent
    json_path = report_dir / "mamba2_ssd_dt_bwd_precision_report.json"
    md_path = report_dir / "mamba2_ssd_dt_bwd_precision_report.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    passed_count = sum(row["passed"] for row in rows)
    failed_count = len(rows) - passed_count
    worst_mere_row = max(rows, key=lambda row: row["worst_MERE"])
    worst_mare_row = max(rows, key=lambda row: row["worst_MARE"])
    lines = [
        "# Mamba2SsdDtBwd 精度验证报告",
        "",
        f"- 测试时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 平台：Ascend 910B",
        "- dtype：FP32",
        "- 参考：PyTorch CPU FP32 解析公式",
        "- 标准：生态算子 MERE/MARE 精度标准",
        "",
        "## 总览",
        "",
        "| 总用例 | 通过 | 失败 | 通过率 |",
        "|---:|---:|---:|---:|",
        f"| {len(rows)} | {passed_count} | {failed_count} | {100 * passed_count / len(rows):.1f}% |",
        "",
        "通过条件：FP32 `MERE < 2^-13` 且 `MARE < 10 * 2^-13`。",
        "",
        "## 用例结果",
        "",
        "| # | 类别 | Shape `[B,L,H,P,T]` | Feature | Worst output | MERE | MARE | 结果 |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['category']} | `{row['shape']}` | "
            f"{row['feature']} | {row['worst_output']} | "
            f"{row['worst_MERE']:.3e} | {row['worst_MARE']:.3e} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 关键发现",
            "",
            f"1. 40 个 FP32 用例通过 {passed_count} 个，失败 {failed_count} 个。",
            f"2. 最大 case-level MERE 为 {worst_mere_row['worst_MERE']:.3e}（case {worst_mere_row['case_id']}）。",
            f"3. 最大 MARE 为 {worst_mare_row['worst_MARE']:.3e}（case {worst_mare_row['case_id']}，{worst_mare_row['worst_output']}）。",
            "4. 用例覆盖 T=64/128、多 batch/head/chunk、softplus、有限 clamp、边界包含关系和反向 chunk 内 scan。",
            "5. dA 与 dt_bias 采用 `[H,B,K]` partial 加原生固定顺序归约，不使用 atomic。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Total={len(rows)} Passed={passed_count} Failed={failed_count}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
