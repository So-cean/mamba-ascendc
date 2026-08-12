"""Repository-level checks for the shared forward benchmark matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

from mamba2_shape_matrix import SHAPE_CASES, SUITES  # noqa: E402


def test_shape_matrix_is_valid_and_covers_all_public_axes():
    assert len(SHAPE_CASES) >= 50
    assert len(SUITES["standard"]) >= 24
    for case in SHAPE_CASES.values():
        assert all(value > 0 for value in case.public_tuple)
        assert case.nheads % case.ngroups == 0
        assert case.full_feature_input_mib > 0

    axes = {case.axis for case in SHAPE_CASES.values()}
    assert {
        "dispatch",
        "tail",
        "sequence",
        "batch",
        "heads",
        "groups",
        "headdim",
        "dstate",
        "chunk",
        "stress",
        "legacy",
    } <= axes
    assert {case.expected_910b3_path for case in SHAPE_CASES.values()} == {
        "generic",
        "aligned",
        "cube_mix",
    }


def test_every_suite_references_known_unique_cases():
    for names in SUITES.values():
        assert names
        assert len(names) == len(set(names))
        assert set(names) <= set(SHAPE_CASES)


def test_profiler_jsonl_matches_the_canonical_matrix():
    path = (
        ROOT
        / "mamba_ascendc"
        / "csrc"
        / "ops"
        / "mamba2_ssd_fwd"
        / "test"
        / "mamba2_ssd_fwd_perf_cases.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) >= 8
    for row in rows:
        case = SHAPE_CASES[row["case"]]
        inputs = {item["name"]: item for item in row["inputs"]}
        x_shape = inputs["x"]["shape"]
        b_shape = inputs["B"]["shape"]
        assert [
            x_shape[0],
            x_shape[1],
            x_shape[2],
            x_shape[3],
            b_shape[3],
            inputs["chunk_size"]["value"],
            b_shape[2],
        ] == list(case.public_tuple)
        assert row["expected_910b3_path"] == case.expected_910b3_path
