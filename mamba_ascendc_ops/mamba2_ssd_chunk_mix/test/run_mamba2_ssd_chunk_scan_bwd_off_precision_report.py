#!/usr/bin/env python3
"""Run native precision cases and emit JSON/Markdown reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from chunk_scan_bwd_off_precision_common import THRESHOLD, all_cases, evaluate


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
            "mode": case.mode,
            "cube_input_dtype": "float16",
            "accumulation_dtype": "float32",
            "outputs": output_metrics,
            "worst_output": worst_name,
            "worst_MERE": max(v["MERE"] for v in output_metrics.values()),
            "worst_MARE": worst["MARE"],
            "passed": passed,
        }
        rows.append(row)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {case.case_id:02d} "
            f"{case.mode} {case.shape} worst={worst_name} "
            f"MERE={row['worst_MERE']:.3e} MARE={row['worst_MARE']:.3e}"
        )

    report_dir = Path(__file__).resolve().parent
    json_path = report_dir / "mamba2_ssd_chunk_scan_bwd_off_precision_report.json"
    md_path = report_dir / "mamba2_ssd_chunk_scan_bwd_off_precision_report.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    passed_count = sum(row["passed"] for row in rows)
    worst_mere = max(rows, key=lambda row: row["worst_MERE"])
    worst_mare = max(rows, key=lambda row: row["worst_MARE"])
    lines = [
        "# Mamba2SsdChunkScanBwdOff 精度验证报告",
        "",
        f"- 测试时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 平台：Ascend 910B3",
        "- 精度合同：FP16 Cube 输入，FP32 累加与输出",
        "- 参考：显式模拟 FP16 舍入的 PyTorch CPU matmul",
        "- 通过条件：`MERE < 2^-13` 且 `MARE < 10 * 2^-13`",
        "",
        "## 总览",
        "",
        "| 总用例 | 通过 | 失败 | 通过率 | Worst MERE | Worst MARE |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {len(rows)} | {passed_count} | {len(rows) - passed_count} | "
        f"{100 * passed_count / len(rows):.1f}% | "
        f"{worst_mere['worst_MERE']:.3e} | {worst_mare['worst_MARE']:.3e} |",
        "",
        "## 用例结果",
        "",
        "| # | 类别 | Shape `[B,H,K,G]` | 数值模式 | Worst output | MERE | MARE | 结果 |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['category']} | `{row['shape']}` | "
            f"{row['mode']} | {row['worst_output']} | "
            f"{row['worst_MERE']:.3e} | {row['worst_MARE']:.3e} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 覆盖范围",
            "",
            "1. 40 个 case 覆盖 `B=1/2`、`H=1/2/4/8`、`K=1/2/4/8` 和共享/逐 head group。",
            "2. 覆盖普通衰减、零衰减、强衰减和小信号四种数值域。",
            "3. 三个输出逐一统计 MERE、MARE、绝对误差与 cosine similarity。",
            "4. 本报告验证离对角 ChunkScan 反向分支，不代表完整 Mamba2 backward 已完成。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Total={len(rows)} Passed={passed_count} Failed={len(rows) - passed_count}")
    return 0 if passed_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
