import ctypes
import os

import pytest
import torch
import torch.nn.functional as F

library_path = os.environ.get("MAMBA_CHUNK_MIX_OP_API_LIB")
if library_path:
    library = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
    assert getattr(library, "aclnnMamba2SsdOffEpilogue")
import ascend_kernel  # noqa: E402,F401


@pytest.mark.parametrize(
    "batch,heads,groups,chunks,state_dim",
    [(1, 2, 1, 1, 64), (1, 8, 2, 4, 128), (2, 16, 4, 8, 128)],
)
def test_mamba2_ssd_off_epilogue(
    batch, heads, groups, chunks, state_dim
):
    torch.manual_seed(20260802)
    tile = 64
    heads_per_group = heads // groups
    c = (0.2 * torch.randn(batch, chunks, groups, tile, state_dim,
                           device="npu")).half()
    states = 0.2 * torch.randn(
        batch, heads, chunks, tile, state_dim, device="npu"
    )
    da = -0.4 * torch.rand(batch, heads, chunks, tile, device="npu")
    y_diag = 0.2 * torch.randn(
        batch, heads, chunks, tile, tile, device="npu"
    )
    x = torch.randn(batch, chunks * tile, heads, tile, device="npu")
    d = torch.randn(heads, tile, device="npu")
    z = torch.randn_like(x)

    actual = torch.ops.mamba_ascend.mamba2_ssd_off_epilogue(
        c, states, da, y_diag, x, d, z
    )
    c_grouped = c.permute(0, 2, 1, 3, 4).unsqueeze(2)
    states_t = states.reshape(
        batch, groups, heads_per_group, chunks, tile, state_dim
    ).transpose(-1, -2).half()
    y_off = torch.matmul(c_grouped, states_t).float()
    y_off = y_off * torch.exp(
        da.reshape(batch, groups, heads_per_group, chunks, tile)
    ).unsqueeze(-1)
    y_local = y_diag.reshape(
        batch, groups, heads_per_group, chunks, tile, tile
    )
    expected = (y_local + y_off).permute(0, 3, 4, 1, 2, 5)
    expected = expected.reshape(batch, chunks * tile, heads, tile)
    expected = (expected + x * d.view(1, 1, heads, tile)) * F.silu(z)

    error = actual.float() - expected.float()
    nrmse = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    ).item()
    cosine = F.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    max_abs = error.abs().max().item()
    print(
        f"case={(batch, heads, groups, chunks, state_dim)} "
        f"max_abs={max_abs:.6e} nrmse={nrmse:.6e} cosine={cosine:.9f}"
    )
    assert torch.isfinite(actual).all()
    assert max_abs <= 5e-2
    assert nrmse <= 5e-3
    assert cosine >= 0.999
