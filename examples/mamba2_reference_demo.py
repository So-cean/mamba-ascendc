"""Minimal Mamba-2 SSD forward demo using the pure PyTorch reference."""

import torch

from mamba_torch import ssd_chunk_scan_ref


def main() -> None:
    torch.manual_seed(7)
    batch, seqlen, nheads, headdim = 1, 128, 2, 64
    ngroups, dstate, chunk_size = 1, 128, 128

    x = torch.randn(batch, seqlen, nheads, headdim)
    dt = 0.01 + 0.1 * torch.rand(batch, seqlen, nheads)
    A = -(0.1 + 0.4 * torch.rand(nheads))
    B = torch.randn(batch, seqlen, ngroups, dstate) / 5
    C = torch.randn(batch, seqlen, ngroups, dstate) / 5

    with torch.no_grad():
        out, final_state = ssd_chunk_scan_ref(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            return_final_state=True,
        )

    print(f"out.shape={tuple(out.shape)}")
    print(f"final_state.shape={tuple(final_state.shape)}")


if __name__ == "__main__":
    main()
