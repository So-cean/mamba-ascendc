#!/usr/bin/env python3
"""Collect a reproducible Level1 NPU profile for Mamba-2 backward-only."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch_npu

import ascend_kernel  # noqa: F401

from mamba2_backward_bench import (
    CASES,
    autograd_grad,
    make_forward,
    make_inputs,
    synchronize,
    trainable_inputs,
)


WARMUP = 5
ACTIVE = 5
PROFILED_STEPS = WARMUP + ACTIVE


def _find_column(names: list[str], *needles: str) -> str:
    for name in names:
        lowered = name.lower()
        if all(needle.lower() in lowered for needle in needles):
            return name
    raise KeyError(f"missing column containing {needles}: {names}")


def _wait_for_csv(root: Path) -> Path:
    deadline = time.time() + 120
    while time.time() < deadline:
        paths = glob.glob(str(root / "**" / "op_statistic.csv"), recursive=True)
        if paths:
            return Path(max(paths, key=os.path.getmtime))
        time.sleep(0.2)
    raise FileNotFoundError(f"op_statistic.csv not found below {root}")


def _active_kernel_mean_us(root: Path) -> float | None:
    paths = glob.glob(str(root / "**" / "kernel_details.csv"), recursive=True)
    if not paths:
        return None
    path = Path(max(paths, key=os.path.getmtime))
    steps: defaultdict[str, float] = defaultdict(float)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        step_key = _find_column(fields, "step", "id")
        duration_key = _find_column(fields, "duration", "us")
        for row in reader:
            step = row.get(step_key, "").strip()
            duration = row.get(duration_key, "").strip()
            if step and duration:
                steps[step] += float(duration)
    if len(steps) != ACTIVE:
        return None
    return sum(steps.values()) / ACTIVE


def _aggregate(
    csv_path: Path, root: Path
) -> tuple[list[dict[str, float | str]], int, float | None]:
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, float] = defaultdict(float)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        name_key = _find_column(fields, "type")
        total_key = _find_column(fields, "total", "us")
        count_key = next(
            (field for field in fields if "count" in field.lower()), None
        )
        for row in reader:
            name = row.get(name_key, "").strip()
            raw_total = row.get(total_key, "").strip()
            if not name or not raw_total:
                continue
            totals[name] += float(raw_total)
            counts[name] += float(row[count_key]) if count_key else 0.0
    active_kernel_us = _active_kernel_mean_us(root)
    raw_total_us = sum(totals.values())
    if active_kernel_us is None:
        divisor = ACTIVE
    else:
        divisor = min(
            (ACTIVE, PROFILED_STEPS),
            key=lambda value: abs(raw_total_us / value - active_kernel_us),
        )
    rows = [
        {
            "op_type": name,
            "mean_us": total / divisor,
            "calls_per_step": counts[name] / divisor,
        }
        for name, total in sorted(totals.items(), key=lambda item: -item[1])
    ]
    return rows, divisor, active_kernel_us


def profile_case(
    case: str, feature: str, trace_root: Path, allow_nonfinite: bool,
    final_state_grad: bool,
) -> dict:
    device = torch.device("npu")
    inputs = make_inputs(case, feature, device)
    _, grad_inputs = trainable_inputs(inputs)
    forward = make_forward(
        "ascendc", case, feature, inputs,
        return_final_state=final_state_grad,
    )
    fixed_result = forward()
    batch, _, nheads, headdim, dstate, _, _ = CASES[case]
    generator = torch.Generator(device="cpu").manual_seed(20260806)
    dout = torch.randn(inputs["x"].shape, generator=generator).to(device)
    dfinal = torch.randn(
        (batch, nheads, headdim, dstate), generator=generator
    ).to(device)

    def backward_only():
        return autograd_grad(
            fixed_result,
            grad_inputs,
            dout,
            dfinal,
            final_state_grad,
            retain_graph=True,
        )

    output_values = fixed_result if isinstance(fixed_result, tuple) else (fixed_result,)
    outputs_finite = all(
        bool(torch.isfinite(value).all().item()) for value in output_values
    )
    probe_grads = backward_only()
    synchronize(device)
    gradients_finite = all(
        bool(torch.isfinite(value).all().item()) for value in probe_grads
    )
    del probe_grads
    if (not outputs_finite or not gradients_finite) and not allow_nonfinite:
        raise RuntimeError(
            f"{case}: non-finite forward/gradient result; refusing to profile"
        )

    case_root = trace_root / case
    shutil.rmtree(case_root, ignore_errors=True)
    case_root.mkdir(parents=True, exist_ok=True)
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )
    schedule = torch_npu.profiler.schedule(
        wait=0, warmup=WARMUP, active=ACTIVE, repeat=1, skip_first=0
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(case_root)
        ),
        experimental_config=config,
        schedule=schedule,
    ) as profiler:
        for _ in range(WARMUP + ACTIVE):
            backward_only()
            profiler.step()
    synchronize(device)
    csv_path = _wait_for_csv(case_root)
    ops, divisor, active_kernel_us = _aggregate(csv_path, case_root)
    return {
        "case": case,
        "shape_b_l_h_p_n_c_g": [
            CASES[case][0], CASES[case][1], CASES[case][2], CASES[case][3],
            CASES[case][4], CASES[case][6], CASES[case][5],
        ],
        "warmup": WARMUP,
        "active": ACTIVE,
        "outputs_finite": outputs_finite,
        "gradients_finite": gradients_finite,
        "final_state_grad": final_state_grad,
        "op_statistic_divisor": divisor,
        "active_kernel_details_ms": (
            active_kernel_us / 1000.0 if active_kernel_us is not None else None
        ),
        "total_device_ms": sum(item["mean_us"] for item in ops) / 1000.0,
        "op_types": ops,
        "op_statistic_csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASES), required=True
    )
    parser.add_argument("--feature", default="full")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-nonfinite",
        action="store_true",
        help="collect a diagnostic profile even when outputs/gradients are non-finite",
    )
    parser.add_argument("--final-state-grad", action="store_true")
    args = parser.parse_args()
    rows = [
        profile_case(
            case, args.feature, args.trace_root, args.allow_nonfinite,
            args.final_state_grad,
        )
        for case in args.cases
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
