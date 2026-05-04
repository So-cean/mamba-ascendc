# Copyright (c) 2024, mamba-triton-ascend authors.
# Selective state update (single-step inference) - top-level interface.

from mamba_triton_ascend.ops.triton_kernels.selective_state_update import (
    selective_state_update,
)
from mamba_triton_ascend.ops.reference import (
    selective_state_update_ref,
)

__all__ = ["selective_state_update", "selective_state_update_ref"]
