"""Precision and determinism gates for the fused diagonal/state backward."""

from __future__ import annotations

import pytest
import torch

from chunk_scan_bwd_diag_state_precision_common import (
    all_cases,
    evaluate,
    make_inputs,
    npu_call,
)


@pytest.mark.parametrize(
    "case", all_cases(), ids=lambda case: f"case_{case.case_id:02d}"
)
def test_precision(case):
    output_metrics, passed = evaluate(case)
    assert passed, output_metrics


def test_bitwise_deterministic():
    inputs = make_inputs(all_cases()[-1])
    first = npu_call(inputs)
    second = npu_call(inputs)
    for name, lhs, rhs in zip(
        ("d_xdt", "d_b_group", "d_c_diag_group", "g_dA_cs"),
        first,
        second,
    ):
        assert torch.equal(lhs, rhs), f"{name} is not bitwise deterministic"
