"""Pytest gate for all 45 Mamba2 SSD precision cases."""

import pytest

from precision_common import THRESHOLD, evaluate_case, iter_case_specs


SPECS = list(iter_case_specs())


@pytest.mark.parametrize(
    "spec",
    SPECS,
    ids=[f"case_{spec[0]:02d}_{spec[1]}_{spec[4] or spec[3]}" for spec in SPECS],
)
def test_precision(spec):
    result = evaluate_case(spec)
    assert result["passed"], (
        f"case={result['case_id']} {result['category']}/{result['description']} "
        f"shape={result['shape']} threshold={THRESHOLD:.8e} "
        f"out={result['out']} final_state={result['final_state']}"
    )
