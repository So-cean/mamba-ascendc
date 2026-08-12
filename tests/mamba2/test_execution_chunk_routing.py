"""CPU-only tests for logical chunk to Cube/MIX micro-tile routing."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "mamba_ascendc"
    / "python"
    / "ascend_kernel"
    / "ascend_kernel"
    / "mamba2.py"
)
SPEC = spec_from_file_location("mamba2_source_routing", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MAMBA2 = module_from_spec(SPEC)
SPEC.loader.exec_module(MAMBA2)


def _inputs(dstate=128, headdim=64, seqlen=1024):
    x = torch.empty((1, seqlen, 128, headdim), device="meta")
    b = torch.empty((1, seqlen, 8, dstate), device="meta")
    return x, b


@pytest.mark.parametrize(
    ("logical_chunk", "dstate", "chunk128", "expected"),
    [
        (256, 128, "1", 128),
        (256, 128, "0", 64),
        (512, 64, "1", 64),
        (192, 128, "1", 64),
        (128, 128, "1", 128),
    ],
)
def test_select_execution_chunk_size(
    monkeypatch, logical_chunk, dstate, chunk128, expected
):
    monkeypatch.setenv("MAMBA_ASCENDC_CHUNK_MIX", "1")
    monkeypatch.setenv("MAMBA_ASCENDC_CHUNK128", chunk128)
    x, b = _inputs(dstate=dstate)
    assert MAMBA2._select_execution_chunk_size(x, b, logical_chunk) == expected


def test_select_execution_chunk_preserves_unsupported_shape(monkeypatch):
    monkeypatch.setenv("MAMBA_ASCENDC_CHUNK_MIX", "1")
    monkeypatch.setenv("MAMBA_ASCENDC_CHUNK128", "1")
    x, b = _inputs(headdim=32)
    assert MAMBA2._select_execution_chunk_size(x, b, 256) == 256
