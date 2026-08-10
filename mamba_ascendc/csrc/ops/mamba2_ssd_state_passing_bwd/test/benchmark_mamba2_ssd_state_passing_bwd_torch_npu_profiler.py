"""Profile native state-passing backward against an NPU tensor baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from mamba2_ssd_state_passing_bwd_profiler_common import (
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
        "--case-file",
        default=os.path.join(HERE, "mamba2_ssd_state_passing_bwd_perf_cases.jsonl"),
    )
    parser.add_argument("--trace-root", default=os.path.join(HERE, "profiler_trace"))
    parser.add_argument(
        "--report-md",
        default=os.path.join(
            HERE, "mamba2_ssd_state_passing_bwd_torch_npu_profiler_report.md"
        ),
    )
    parser.add_argument("--only-case", type=int)
    args = parser.parse_args()

    cases = load_cases(args.case_file)
    indices = [args.only_case] if args.only_case is not None else range(len(cases))
    results = []
    for index in indices:
        custom, baseline = prepare_paths(cases[index], torch.device("npu:0"))
        custom_us, custom_csv = profile_callable(
            custom,
            os.path.join(args.trace_root, "state_passing_bwd", "custom", f"case_{index:03d}"),
        )
        baseline_us, baseline_csv = profile_callable(
            baseline,
            os.path.join(args.trace_root, "state_passing_bwd", "baseline", f"case_{index:03d}"),
        )
        row = {
            "case": index,
            "shape": shape_label(cases[index]),
            "dtype": "float32",
            "custom_us": custom_us,
            "baseline_us": baseline_us,
            "speedup": baseline_us / custom_us,
        }
        results.append(row)
        print(
            f"case={index} custom={custom_us:.3f}us baseline={baseline_us:.3f}us "
            f"speedup={row['speedup']:.3f} custom_csv={custom_csv} "
            f"baseline_csv={baseline_csv}"
        )

    case_label = os.path.relpath(os.path.abspath(args.case_file), REPO_ROOT)
    trace_label = os.path.relpath(os.path.abspath(args.trace_root), REPO_ROOT)
    with open(args.report_md, "w", encoding="utf-8") as handle:
        handle.write(render_report(results, case_label, trace_label))
    print(f"report={args.report_md}")


if __name__ == "__main__":
    main()
