import os

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

candidate_library = os.environ.get("MAMBA2_TEST_EXTENSION_LIB")
if candidate_library:
    torch.ops.load_library(candidate_library)
else:
    import ascend_kernel  # noqa: F401


@pytest.mark.parametrize(
    "seqlen,chunk_size,dstate",
    [(128, 64, 64), (2048, 64, 64), (256, 128, 128), (4096, 128, 128)],
)
def test_mamba2_ssd_preprocess_precision(seqlen, chunk_size, dstate):
    torch.manual_seed(20260801)
    batch, nheads, headdim = 2, 8, 64
    ngroups = 2
    x = torch.randn(batch, seqlen, nheads, headdim, device="npu")
    dt = 0.1 * torch.randn(batch, seqlen, nheads, device="npu")
    A = -(0.2 + torch.rand(nheads, device="npu"))
    B = 0.2 * torch.randn(batch, seqlen, ngroups, dstate, device="npu")
    C = 0.2 * torch.randn(batch, seqlen, ngroups, dstate, device="npu")
    dt_bias = 0.1 * torch.randn(nheads, device="npu")

    x_cube, da_cumsum, b_cube, c_cube = (
        torch.ops.mamba_ascend.mamba2_ssd_preprocess(
            x, dt, A, B, C, dt_bias, chunk_size, True, 0.001, 2.0
        )
    )
    nchunks = seqlen // chunk_size
    dt_ref = F.softplus(dt + dt_bias.view(1, 1, nheads)).clamp(0.001, 2.0)
    da_ref = (dt_ref * A.view(1, 1, nheads)).reshape(
        batch, nchunks, chunk_size, nheads
    ).permute(0, 3, 1, 2).cumsum(-1)
    x_ref = (x * dt_ref.unsqueeze(-1)).reshape(
        batch, nchunks, chunk_size, nheads, headdim
    ).permute(0, 3, 1, 2, 4).to(torch.float16)
    b_ref = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).permute(
        0, 1, 3, 4, 2
    ).to(torch.float16)
    c_ref = C.reshape(batch, nchunks, chunk_size, ngroups, dstate).permute(
        0, 1, 3, 2, 4
    ).to(torch.float16)

    torch.testing.assert_close(x_cube, x_ref, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(da_cumsum, da_ref, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(b_cube, b_ref, rtol=0, atol=0)
    torch.testing.assert_close(c_cube, c_ref, rtol=0, atol=0)
