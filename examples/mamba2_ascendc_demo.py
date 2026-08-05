"""Minimal Mamba-2 SSD forward demo on an Ascend NPU."""

from __future__ import annotations

import ctypes
import os

import torch
import torch_npu  # noqa: F401  # Registers the ``npu`` device with PyTorch.


def load_custom_op_api() -> None:
    """Load the generated ACLNN API library when the job exports its path."""
    library_path = os.environ.get("MAMBA_CHUNK_MIX_OP_API_LIB")
    if library_path:
        ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)


def main() -> None:
    load_custom_op_api()
    import ascend_kernel

    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is available")

    torch.manual_seed(20260803)
    device = torch.device("npu:0")

    # A compact configuration matching a standard Mamba-2 SSD layer.
    batch, seqlen = 1, 128
    nheads, headdim = 2, 64
    ngroups, dstate = 1, 128
    chunk_size = 128

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.float32, device=device)

    x = randn(batch, seqlen, nheads, headdim)
    dt = 0.01 + 0.1 * torch.rand(
        batch, seqlen, nheads, dtype=torch.float32, device=device
    )
    A = -(0.1 + 0.4 * torch.rand(nheads, dtype=torch.float32, device=device))
    B = randn(batch, seqlen, ngroups, dstate) / 5
    C = randn(batch, seqlen, ngroups, dstate) / 5
    D = randn(nheads, headdim)
    z = randn(batch, seqlen, nheads, headdim)
    dt_bias = randn(nheads) * 0.1

    with torch.no_grad():
        out, final_state = ascend_kernel.mamba2_ssd_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            return_final_state=True,
        )
        torch.npu.synchronize()

    assert out.shape == (batch, seqlen, nheads, headdim)
    assert final_state.shape == (batch, nheads, headdim, dstate)
    assert torch.isfinite(out).all().item()
    assert torch.isfinite(final_state).all().item()

    print(f"device={torch.npu.get_device_name(0)}")
    print(f"out.shape={tuple(out.shape)}")
    print(f"final_state.shape={tuple(final_state.shape)}")
    print(f"out.checksum={out.float().sum().item():.6f}")
    print(f"final_state.checksum={final_state.float().sum().item():.6f}")
    print("Mamba-2 AscendC forward: PASS")


if __name__ == "__main__":
    main()
