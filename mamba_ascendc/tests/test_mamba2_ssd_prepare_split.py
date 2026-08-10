import pytest
import torch

import ascend_kernel  # noqa: F401


@pytest.mark.parametrize(
    "batch,nheads,nchunks,ngroups",
    [(1, 4, 3, 2), (2, 8, 4, 2), (2, 32, 8, 4)],
)
def test_prepare_split_matches_torch(batch, nheads, nchunks, ngroups):
    torch.manual_seed(20260808)
    cb = (
        torch.randn(batch, nchunks, ngroups, 64, 64,
                    device="npu", dtype=torch.float32)
        .mul_(0.2)
        .to(torch.float16)
    )
    increments = -torch.rand(
        batch, nheads, nchunks, 64, device="npu", dtype=torch.float32
    ).mul_(0.08)
    d_a = increments.cumsum(-1).contiguous()
    x = (
        torch.randn(batch, nheads, nchunks, 64, 64,
                    device="npu", dtype=torch.float32)
        .mul_(0.3)
        .to(torch.float16)
    )

    w, weighted_x = torch.ops.mamba_ascend.mamba2_ssd_prepare_split(
        cb, d_a, x
    )
    heads_per_group = nheads // ngroups
    cb_head = (
        cb.permute(0, 2, 1, 3, 4)
        .repeat_interleave(heads_per_group, dim=1)
        .float()
    )
    diff = d_a.unsqueeze(-1) - d_a.unsqueeze(-2)
    w_ref = (cb_head * torch.tril(torch.exp(diff))).to(torch.float16)
    weighted_x_ref = (
        x.float() * torch.exp(d_a[..., -1:] - d_a).unsqueeze(-1)
    ).to(torch.float16)

    torch.testing.assert_close(w.float(), w_ref.float(), rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(
        weighted_x.float(), weighted_x_ref.float(), rtol=4e-3, atol=4e-3
    )
