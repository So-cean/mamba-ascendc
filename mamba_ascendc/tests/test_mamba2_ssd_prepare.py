import pytest
import torch

import ascend_kernel


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Ascend NPU is required",
)


@pytest.mark.parametrize(
    "batch,nheads,nchunks,ngroups,chunk_size,headdim",
    [
        (1, 1, 2, 1, 16, 16),
        (1, 4, 3, 2, 32, 32),
        (2, 8, 4, 2, 64, 64),
        (1, 16, 2, 4, 128, 64),
    ],
)
def test_prepare_matches_torch(
    batch, nheads, nchunks, ngroups, chunk_size, headdim
):
    torch.manual_seed(20260802)
    cb = (
        torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size,
                    device="npu", dtype=torch.float32)
        .mul_(0.2)
        .to(torch.float16)
    )
    increments = -torch.rand(
        batch, nheads, nchunks, chunk_size, device="npu", dtype=torch.float32
    ).mul_(0.08)
    d_a = increments.cumsum(-1).contiguous()
    x = (
        torch.randn(batch, nheads, nchunks, chunk_size, headdim,
                    device="npu", dtype=torch.float32)
        .mul_(0.3)
        .to(torch.float16)
    )

    w, weighted_x = torch.ops.mamba_ascend.mamba2_ssd_prepare(cb, d_a, x)

    heads_per_group = nheads // ngroups
    cb_head = (
        cb.permute(0, 2, 1, 3, 4)
        .repeat_interleave(heads_per_group, dim=1)
        .float()
    )
    diff = d_a.unsqueeze(-1) - d_a.unsqueeze(-2)
    causal = torch.tril(torch.exp(diff))
    w_ref = (cb_head * causal).to(torch.float16)
    weighted_x_ref = (
        x.float() * torch.exp(d_a[..., -1:] - d_a).unsqueeze(-1)
    ).to(torch.float16)

    torch.testing.assert_close(w.float(), w_ref.float(), rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(
        weighted_x.float(), weighted_x_ref.float(), rtol=4e-3, atol=4e-3
    )
