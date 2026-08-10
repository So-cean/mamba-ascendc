"""Thirty-case FP32 precision gate for state-passing backward."""

from __future__ import annotations

import pytest

from precision_common import FINAL_MODES, TEST_SHAPES, THRESHOLD, run_case


CASES = [
    (category, description, shape, mode)
    for category, description, shape in TEST_SHAPES
    for mode in FINAL_MODES
]


@pytest.mark.parametrize(
    "category,description,shape,mode",
    CASES,
    ids=[f"{category}-{mode}-{shape}" for category, _, shape, mode in CASES],
)
def test_precision(category, description, shape, mode):
    metrics, passed = run_case(shape, mode, seed=20260807 + CASES.index(
        (category, description, shape, mode)
    ))
    assert passed, (
        f"{category}/{description}/{mode}: {metrics}; "
        f"limits MERE<{THRESHOLD:.3e}, MARE<{10 * THRESHOLD:.3e}"
    )
