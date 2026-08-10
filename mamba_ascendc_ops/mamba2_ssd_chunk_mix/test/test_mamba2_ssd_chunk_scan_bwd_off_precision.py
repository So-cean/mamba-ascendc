"""Precision and determinism gates for Mamba2SsdChunkScanBwdOff."""

from __future__ import annotations

import pytest
import torch

from chunk_scan_bwd_off_precision_common import (
    all_cases,
    evaluate,
    make_inputs,
    npu_call,
)


@pytest.mark.parametrize("case", all_cases(), ids=lambda case: f"case_{case.case_id:02d}")
def test_precision(case):
    output_metrics, passed = evaluate(case)
    assert passed, output_metrics


def test_bitwise_deterministic():
    case = all_cases()[-1]
    inputs = make_inputs(case)
    first = npu_call(inputs)
    second = npu_call(inputs)
    for name, lhs, rhs in zip(
        ("d_states_start", "d_c_head", "g_dA_cs_off"), first, second
    ):
        assert torch.equal(lhs, rhs), f"{name} is not bitwise deterministic"
