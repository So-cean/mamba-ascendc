#!/usr/bin/env python3
"""Run 30 native cases and write JSON/Markdown precision reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from precision_common import FINAL_MODES, TEST_SHAPES, THRESHOLD, run_case


def main():
    results = []
    case_id = 0
    for category, description, shape in TEST_SHAPES:
        for mode in FINAL_MODES:
            case_id += 1
            metrics, passed = run_case(shape, mode, 20260807 + case_id - 1)
            result = {
                "case_id": case_id,
                "category": category,
                "description": description,
                "shape": list(shape),
                "dfinal_mode": mode,
                "dtype": "float32",
                "threshold": THRESHOLD,
                "outputs": metrics,
                "passed": passed,
            }
            results.append(result)
            worst_mere = max(value["MERE"] for value in metrics.values())
            worst_mare = max(value["MARE"] for value in metrics.values())
            print(
                f"[{'PASS' if passed else 'FAIL'}] {case_id:02d} "
                f"{category}/{mode} shape={shape} "
                f"worst_MERE={worst_mere:.3e} worst_MARE={worst_mare:.3e}"
            )

    report_dir = Path(__file__).resolve().parent
    json_path = report_dir / "mamba2_ssd_state_passing_bwd_precision_report.json"
    md_path = report_dir / "mamba2_ssd_state_passing_bwd_precision_report.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed_count = sum(item["passed"] for item in results)
    max_mere = max(
        metric["MERE"]
        for item in results
        for metric in item["outputs"].values()
    )
    max_mare = max(
        metric["MARE"]
        for item in results
        for metric in item["outputs"].values()
    )
    max_abs = max(
        metric["max_abs_err"]
        for item in results
        for metric in item["outputs"].values()
    )
    lines = [
        "# Mamba2SsdStatePassingBwd 精度验证报告",
        "",
        f"- 测试时间: {datetime.now().isoformat(timespec='seconds')}",
        "- 测试平台: Ascend 910B3",
        "- dtype: FP32",
        "- 参考实现: PyTorch CPU reverse chunk recurrence",
        "- 判定标准: `MERE < 2^-13` 且 `MARE < 10 * 2^-13`",
        "",
        "## 总览",
        "",
        "| 总用例 | 通过 | 失败 | 通过率 | Worst MERE | Worst MARE | MaxAbsErr |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {len(results)} | {passed_count} | {len(results) - passed_count} | "
        f"{100.0 * passed_count / len(results):.1f}% | {max_mere:.3e} | "
        f"{max_mare:.3e} | {max_abs:.3e} |",
        "",
        "## 用例结果",
        "",
        "| # | 类别 | Shape `[B,H,K,P,N,T]` | dfinal | Worst MERE | Worst MARE | 结果 |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for item in results:
        worst_mere = max(value["MERE"] for value in item["outputs"].values())
        worst_mare = max(value["MARE"] for value in item["outputs"].values())
        lines.append(
            f"| {item['case_id']} | {item['category']} | `{item['shape']}` | "
            f"{item['dfinal_mode']} | {worst_mere:.3e} | {worst_mare:.3e} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 关键发现",
            "",
            "1. `dfinal=None`、显式全零与随机 final-state gradient 均由同一 native kernel 覆盖。",
            "2. `N/T=64/128`、`K=1..8` 以及最多 `B*H=16` 的并行 stream 均纳入测试。",
            "3. 三个输出分别按 MERE/MARE 判定；报告中的 worst 值不是把零输出隐藏后的聚合值。",
            "4. 本报告只证明 reverse state-passing 子算子精度，不代表完整 Mamba2 backward M1 已完成。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Summary: total={len(results)} passed={passed_count} "
        f"failed={len(results) - passed_count}"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
