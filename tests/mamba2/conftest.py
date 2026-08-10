# Copyright (c) 2026, mamba-ascendc authors.
# pytest fixtures for Mamba-2 tests.

import pytest
import torch


@pytest.fixture(scope="function", autouse=True)
def fixed_seed():
    """Fix random seed for deterministic tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def small_config():
    """Small config for fast smoke tests."""
    return dict(
        batch=2, seqlen=16, nheads=4, headdim=8,
        dstate=8, chunk_size=8, ngroups=1,
    )


@pytest.fixture
def medium_config():
    """Medium config for correctness tests."""
    return dict(
        batch=2, seqlen=32, nheads=4, headdim=16,
        dstate=16, chunk_size=16, ngroups=1,
    )


@pytest.fixture
def large_config():
    """Larger config for stress / accuracy tests."""
    return dict(
        batch=2, seqlen=64, nheads=4, headdim=32,
        dstate=32, chunk_size=32, ngroups=1,
    )
