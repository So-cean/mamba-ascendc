#!/usr/bin/env python3
"""Run fused diagonal/state precision cases and emit JSON/Markdown."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from chunk_scan_bwd_diag_state_precision_common import all_cases, evaluate


def main() -> int:
    rows = []
    for case in all_cases():
        output_metrics, passed = evaluate(case)
        worst_name, worst = max(
            output_metrics.items(), key=lambda item: item[1]["NRMSE"]
        )
        row = {
            "case_id": case.case_id,
            "category": case.category,
            "description": case.description,
            "shape": list(case.shape),
            "mode": case.mode,
            "outputs": output_metrics,
            "worst_output": worst_name,
            "worst_NRMSE": worst["NRMSE"],
            "lowest_cosine": min(
                value["cosine_sim"] for value in output_metrics.values()
            ),
            "passed": passed,
        }
        rows.append(row)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {case.case_id:02d} "
            f"{case.mode} {case.shape} worst={worst_name} "
            f"NRMSE={row['worst_NRMSE']:.3e} "
            f"cos={row['lowest_cosine']:.9f}"
        )

    report_dir = Path(__file__).resolve().parent
    stem = "mamba2_ssd_chunk_scan_bwd_diag_state_precision_report"
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    passed_count = sum(row["passed"] for row in rows)
    worst = max(rows, key=lambda row: row["worst_NRMSE"])
    lowest = min(rows, key=lambda row: row["lowest_cosine"])
    lines = [
        "# Mamba2SsdChunkScanBwdDiagState 精度验证报告",
        "",
        f"- 测试时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 平台：Ascend 910B3",
        "- 精度合同：FP16 Cube 输入，FP32 累加与输出",
        "- 参考：显式模拟各 Cube 输入的 FP16 舍入",
        "- 通过条件：每个输出 `NRMSE < 2e-3` 且 `cosine > 0.99999`",
        "",
        "## 总览",
        "",
        "| 总用例 | 通过 | 失败 | 通过率 | Worst NRMSE | Lowest cosine |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {len(rows)} | {passed_count} | {len(rows) - passed_count} | "
        f"{100 * passed_count / len(rows):.1f}% | "
        f"{worst['worst_NRMSE']:.3e} | {lowest['lowest_cosine']:.9f} |",
        "",
        "## 用例结果",
        "",
        "| # | 类别 | Shape `[B,H,K,G]` | 数值模式 | Worst output | NRMSE | cosine | 结果 |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['category']} | `{row['shape']}` | "
            f"{row['mode']} | {row['worst_output']} | "
            f"{row['worst_NRMSE']:.3e} | {row['lowest_cosine']:.9f} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 覆盖范围",
            "",
            "1. 40 个 case 覆盖 `B=1/2`、`H=1/2/4/8`、`K=1/2/4/8` 与共享/逐 head group。",
            "2. 覆盖普通、零衰减、强衰减和小信号数值域。",
            "3. 验证 diagonal scan 与 chunk-state 两条 backward 分支的四个融合输出。",
            "4. 另有 bitwise determinism pytest gate。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(
        f"Total={len(rows)} Passed={passed_count} "
        f"Failed={len(rows) - passed_count}"
    )
    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
