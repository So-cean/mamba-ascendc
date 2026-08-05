# Copyright (c) 2024, mamba-triton-ascend authors.
# pytest fixtures for NPU test suite.

import os
import shutil
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "mamba2_gpu: tests requiring CUDA and the official mamba_ssm package"
    )
    config.addinivalue_line(
        "markers", "mamba2_npu: tests requiring an Ascend NPU and Triton-Ascend"
    )


@pytest.fixture(scope="function", autouse=True)
def fixed_seed():
    """Fix random seed for deterministic tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture(scope="function")
def clear_triton_cache():
    """Clear Triton cache before each test to ensure clean compile."""
    cache_dir = os.path.expanduser("~/.triton/cache")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
        except OSError:
            pass  # NFS lock or in-use files
    yield


def _is_npu_available():
    try:
        import torch_npu
        return torch_npu.npu.is_available()
    except (ImportError, AttributeError):
        return False


@pytest.fixture(scope="session")
def npu_available():
    """Check if NPU is available."""
    return _is_npu_available()


# Skip all NPU tests if NPU not available
npu_skip = pytest.mark.skipif(
    not _is_npu_available(),
    reason="NPU not available"
)


@pytest.fixture
def device():
    """Return NPU device string if available, else CPU."""
    return "npu" if _is_npu_available() else "cpu"


@pytest.fixture
def small_config():
    """Small config for fast smoke tests."""
    return {"batch": 1, "dim": 4, "dstate": 4, "seqlen": 8}


@pytest.fixture
def medium_config():
    """Medium config for correctness tests."""
    return {"batch": 2, "dim": 16, "dstate": 8, "seqlen": 32}
