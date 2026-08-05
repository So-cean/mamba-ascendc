"""Minimal Mamba-2 SSD forward demo using Triton-Ascend."""

import torch
import torch_npu  # noqa: F401  # Register the NPU backend.

from mamba_triton_ascend.mamba2 import mamba_chunk_scan_combined


def main() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("No Ascend NPU is available")

    torch.manual_seed(7)
    device = torch.device("npu:0")
    batch, seqlen, nheads, headdim = 1, 128, 2, 64
    ngroups, dstate, chunk_size = 1, 128, 128

    x = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt = 0.01 + 0.1 * torch.rand(batch, seqlen, nheads, device=device)
    A = -(0.1 + 0.4 * torch.rand(nheads, device=device))
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device) / 5
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device) / 5

    with torch.no_grad():
        out, final_state = mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            return_final_states=True,
            backend="triton",
        )
        torch.npu.synchronize()

    print(f"out.shape={tuple(out.shape)}")
    print(f"final_state.shape={tuple(final_state.shape)}")


if __name__ == "__main__":
    main()
