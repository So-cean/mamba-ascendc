"""Sanitize NPU decomposition profiles into README-ready aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize_csv(path: Path, profile_root: Path) -> dict:
    relative = path.relative_to(profile_root)
    backend, case = relative.parts[0:2]
    steps: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            step = row.get("Step Id", "").strip()
            if step:
                steps.setdefault(step, []).append(row)
    if len(steps) != 5:
        raise RuntimeError(f"expected 5 active steps in {path}, found {len(steps)}")
    launches = [len(rows) for rows in steps.values()]
    device_ms = [
        sum(float(row["Duration(us)"]) for row in rows) / 1000
        for rows in steps.values()
    ]
    names = sorted({row["Name"] for rows in steps.values() for row in rows})
    return {
        "backend": backend,
        "case": case,
        "active_steps": len(steps),
        "kernel_launches_per_step": statistics.fmean(launches),
        "profiled_device_ms_per_step": statistics.fmean(device_ms),
        "unique_kernel_types": len(names),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = sorted(args.profile_root.rglob("kernel_details.csv"))
    if not files:
        raise RuntimeError(f"no kernel_details.csv under {args.profile_root}")
    results = [summarize_csv(path, args.profile_root) for path in files]
    rendered = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
