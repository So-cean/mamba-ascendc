"""Public Python API for the packaged Mamba-2 AscendC forward operator."""

from ._runtime import (
    configure_runtime,
    get_custom_opp_path,
    get_op_api_library_path,
    runtime_info,
)

# Configure custom OPP discovery and preload its ACLNN library before torch_npu
# initializes the operator runtime.
configure_runtime()

import torch
import torch_npu  # noqa: F401

torch.ops.load_library(runtime_info()["extension_library"])

from .mamba2 import mamba_chunk_scan_combined, mamba2_ssd_fwd

__all__ = [
    "configure_runtime",
    "get_custom_opp_path",
    "get_op_api_library_path",
    "mamba_chunk_scan_combined",
    "mamba2_ssd_fwd",
    "runtime_info",
]
