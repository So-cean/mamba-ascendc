#!/usr/bin/env python3
"""Public forward precision gate for the Ascend 950 ChunkMix key 9 path."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FWD_TEST_DIR = (
    REPO_ROOT / "mamba_ascendc" / "csrc" / "ops" /
    "mamba2_ssd_fwd" / "test"
)
sys.path.insert(0, str(FWD_TEST_DIR))

from run_mamba2_ssd_fwd_aligned_precision import _run  # noqa: E402


CASES = (
    ((1, 128, 2, 64, 64, 64, 1), "basic"),
    ((1, 128, 2, 64, 64, 64, 1), "all"),
    ((1, 256, 4, 64, 64, 64, 1), "basic"),
    ((2, 256, 8, 64, 64, 64, 4), "initial"),
    ((1, 512, 8, 64, 64, 64, 2), "all"),
    ((2, 512, 8, 64, 64, 64, 1), "all"),
)


def main() -> None:
    results = []
    for case_id, (shape, variant) in enumerate(CASES, 1):
        print(json.dumps({
            "case_id": case_id,
            "shape": shape,
            "variant": variant,
            "stage": "launch",
        }), flush=True)
        result = _run(case_id, shape, variant)
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
        if not result["passed"]:
            raise AssertionError(
                f"public forward key9 precision failed: {shape} {variant}"
            )
    print(json.dumps({
        "summary": "PASS",
        "passed": len(results),
        "total": len(results),
    }), flush=True)


if __name__ == "__main__":
    main()
