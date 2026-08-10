#!/usr/bin/env python3
"""Run the M1 gradient matrix and write public reports."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from bwd_m1_precision_common import all_cases, evaluate


def main():
    rows = []
    for case in all_cases():
        metrics, passed = evaluate(case)
        worst_name, worst = max(metrics.items(), key=lambda item: item[1]["nrmse"])
        row = {
            "case_id": case.case_id,
            "category": case.category,
            "shape": list(case.shape),
            "mode": case.mode,
            "gradients": metrics,
            "worst_gradient": worst_name,
            "worst_nrmse": worst["nrmse"],
            "lowest_cosine": min(v["cosine_sim"] for v in metrics.values()),
            "passed": passed,
        }
        rows.append(row)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {case.case_id:02d} "
            f"{case.category}/{case.mode} {case.shape} worst={worst_name} "
            f"nrmse={row['worst_nrmse']:.3e} cosine={row['lowest_cosine']:.9f}"
        )

    report_dir = Path(__file__).resolve().parent
    platform_name = torch.npu.get_device_name(0)
    default_tag = platform_name.lower().replace("_", "-")
    report_tag = os.environ.get("MAMBA_PRECISION_REPORT_TAG", default_tag)
    json_path = report_dir / (
        f"mamba2_ssd_bwd_m1_precision_report_{report_tag}.json"
    )
    md_path = report_dir / (
        f"mamba2_ssd_bwd_m1_precision_report_{report_tag}.md"
    )
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    passed_count = sum(row["passed"] for row in rows)
    worst = max(rows, key=lambda row: row["worst_nrmse"])
    lowest = min(rows, key=lambda row: row["lowest_cosine"])
    lines = [
        "# Mamba2 SSD backward M1 精度验证报告",
        "",
        f"- 测试时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 平台：{platform_name}",
        "- 参考：PyTorch FP32 `ssd_chunk_scan_ref` autograd",
        "- M1：AscendC off/diag-state/state-passing/dt native-core 集成路径",
        "- 通过条件：每项梯度 NRMSE <= 8e-3 且 cosine >= 0.999",
        "",
        "## 总览",
        "",
        "| 总用例 | 通过 | 失败 | Worst NRMSE | Lowest cosine |",
        "|---:|---:|---:|---:|---:|",
        f"| {len(rows)} | {passed_count} | {len(rows) - passed_count} | "
        f"{worst['worst_nrmse']:.3e} | {lowest['lowest_cosine']:.9f} |",
        "",
        "## 用例结果",
        "",
        "| # | 类别 | Shape `[B,L,H,G]` | 模式 | Worst gradient | NRMSE | Lowest cosine | 结果 |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['category']} | `{row['shape']}` | "
            f"{row['mode']} | {row['worst_gradient']} | "
            f"{row['worst_nrmse']:.3e} | {row['lowest_cosine']:.9f} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 覆盖范围",
            "",
            "- output-only、output+final、final-only 三种上游梯度入口。",
            "- D 的 `[H,P]` 与 `[H]` 两种接口，以及 z、dt_bias、softplus、finite clamp、initial state。",
            "- B/L/H/G scaling 和共享 group 的确定性 head reduction。",
            "- 九项 public 梯度按实际 optional 输入逐项比较。",
            "",
            "该报告验证 M1 native-core 数学与 public autograd 闭环；SSD 核心分支均已原生化，public 层仍保留少量 tensor glue。",
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
