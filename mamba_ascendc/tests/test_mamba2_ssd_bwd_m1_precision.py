"""Formal 40-case end-to-end M1 backward precision suite."""

import pytest

from bwd_m1_precision_common import all_cases, evaluate


@pytest.mark.parametrize("case", all_cases(), ids=lambda case: f"case_{case.case_id:02d}")
def test_mamba2_ssd_bwd_m1_precision(case):
    metrics, passed = evaluate(case)
    assert passed, metrics
