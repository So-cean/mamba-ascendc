"""Profile Mamba2 SSD backward M0 against NPU PyTorch composition."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from mamba2_ssd_bwd_profiler_common import (
    load_cases,
    prepare_paths,
    profile_callable,
    render_report,
    shape_label,
)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-file", default=os.path.join(HERE, "mamba2_ssd_bwd_perf_cases.jsonl")
    )
    parser.add_argument(
        "--trace-root", default=os.path.join(HERE, "profiler_trace")
    )
    parser.add_argument(
        "--report-md",
        default=os.path.join(HERE, "mamba2_ssd_bwd_torch_npu_profiler_report.md"),
    )
    parser.add_argument("--only-case", type=int)
    args = parser.parse_args()

    cases = load_cases(args.case_file)
    indices = [args.only_case] if args.only_case is not None else list(range(len(cases)))
    device = torch.device("npu:0")
    results = []
    for index in indices:
        case = cases[index]
        custom, baseline = prepare_paths(case, device)
        custom_dir = os.path.join(args.trace_root, "mamba2_ssd_bwd", "custom", f"case_{index:03d}")
        baseline_dir = os.path.join(args.trace_root, "mamba2_ssd_bwd", "baseline", f"case_{index:03d}")
        custom_us, custom_csv = profile_callable(custom, custom_dir)
        baseline_us, baseline_csv = profile_callable(baseline, baseline_dir)
        row = {
            "case": index,
            "shape": shape_label(case),
            "dtype": "float32",
            "custom_us": custom_us,
            "baseline_us": baseline_us,
            "speedup": baseline_us / custom_us,
        }
        results.append(row)
        print(
            f"case={index} custom={custom_us:.3f}us baseline={baseline_us:.3f}us "
            f"speedup={row['speedup']:.3f} custom_csv={custom_csv} baseline_csv={baseline_csv}"
        )

    case_label = os.path.relpath(os.path.abspath(args.case_file), REPO_ROOT)
    trace_label = os.path.relpath(os.path.abspath(args.trace_root), REPO_ROOT)
    report = render_report(results, case_label, trace_label)
    with open(args.report_md, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"report={args.report_md}")


if __name__ == "__main__":
    main()
