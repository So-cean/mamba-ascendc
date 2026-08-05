import ctypes
import os

import pytest
import torch
import torch.nn.functional as F


library_path = os.environ.get("MAMBA_CHUNK_MIX_OP_API_LIB")
if library_path:
    library = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
    assert getattr(library, "aclnnMamba2SsdStateEpilogue")
import ascend_kernel  # noqa: E402,F401


def _reference(chunk_states, da, c, y_diag, x, d, z, initial):
    state_np_layout = chunk_states.shape[-1] == 64
    if state_np_layout:
        batch, heads, chunks, state_dim, head_dim = chunk_states.shape
    else:
        batch, heads, chunks, head_dim, state_dim = chunk_states.shape
    chunk_size = da.shape[-1]
    groups = c.shape[2]
    heads_per_group = heads // groups
    if initial is None:
        state = torch.zeros(
            (batch, heads, state_dim, head_dim) if state_np_layout else
            (batch, heads, head_dim, state_dim),
            dtype=chunk_states.dtype,
            device=chunk_states.device,
        )
    else:
        state = (initial.transpose(-1, -2).contiguous()
                 if state_np_layout else initial.clone())

    outputs = []
    for chunk in range(chunks):
        c_chunk = c[:, chunk].repeat_interleave(heads_per_group, dim=1)
        state_for_cube = state if state_np_layout else state.transpose(-1, -2)
        y_off = torch.matmul(c_chunk, state_for_cube.half()).float()
        y = y_diag[:, :, chunk] + y_off * torch.exp(
            da[:, :, chunk]
        ).unsqueeze(-1)
        outputs.append(y)
        decay = torch.exp(da[:, :, chunk, -1]).unsqueeze(-1).unsqueeze(-1)
        state = decay * state + chunk_states[:, :, chunk]

    out = torch.stack(outputs, dim=2).permute(0, 2, 3, 1, 4)
    out = out.reshape(batch, chunks * chunk_size, heads, head_dim)
    out = (out + x * d.view(1, 1, heads, head_dim)) * F.silu(z)
    return out, state.transpose(-1, -2) if state_np_layout else state


def _metrics(actual, expected):
    error = actual.float() - expected.float()
    nrmse = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    ).item()
    cosine = F.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    return error.abs().max().item(), nrmse, cosine


@pytest.mark.parametrize(
    "batch,heads,groups,chunks,chunk_size,state_dim,has_initial",
    [
        (1, 2, 1, 1, 64, 64, False),
        (1, 8, 2, 4, 64, 128, True),
        (2, 16, 4, 8, 64, 128, False),
        # 32 paired tasks: exercises the two-head 64x128 Cube path while
        # retaining enough tasks to fill all 20 AIC cores.
        (4, 16, 4, 4, 64, 128, True),
        (1, 4, 2, 2, 128, 128, False),
        (2, 16, 4, 8, 128, 128, True),
    ],
)
def test_mamba2_ssd_state_epilogue(
    batch, heads, groups, chunks, chunk_size, state_dim, has_initial
):
    torch.manual_seed(20260802)
    head_dim = 64
    state_shape = ((batch, heads, chunks, state_dim, head_dim)
                   if state_dim == 64 or chunk_size == 128 else
                   (batch, heads, chunks, head_dim, state_dim))
    chunk_states = 0.05 * torch.randn(*state_shape, device="npu")
    da_step = -(0.001 + 0.01 * torch.rand(
        batch, heads, chunks, chunk_size, device="npu"
    ))
    da = torch.cumsum(da_step, dim=-1)
    c = (0.1 * torch.randn(
        batch, chunks, groups, chunk_size, state_dim, device="npu"
    )).half()
    y_diag = 0.1 * torch.randn(
        batch, heads, chunks, chunk_size, head_dim, device="npu"
    )
    x = 0.2 * torch.randn(
        batch, chunks * chunk_size, heads, head_dim, device="npu"
    )
    d = 0.2 * torch.randn(heads, head_dim, device="npu")
    z = 0.2 * torch.randn_like(x)
    initial = None
    if has_initial:
        initial = 0.05 * torch.randn(
            batch, heads, head_dim, state_dim, device="npu"
        )

    actual_out, actual_final = (
        torch.ops.mamba_ascend.mamba2_ssd_state_epilogue(
            chunk_states, da, c, y_diag, x, d, z, initial
        )
    )
    expected_out, expected_final = _reference(
        chunk_states, da, c, y_diag, x, d, z, initial
    )

    out_metrics = _metrics(actual_out, expected_out)
    final_metrics = _metrics(actual_final, expected_final)
    print(
        f"case={(batch, heads, groups, chunks, state_dim, has_initial)} "
        f"out(max_abs,nrmse,cos)={out_metrics} "
        f"final(max_abs,nrmse,cos)={final_metrics}",
        flush=True,
    )
    assert torch.isfinite(actual_out).all()
    assert torch.isfinite(actual_final).all()
    for max_abs, nrmse, cosine in (out_metrics, final_metrics):
        assert max_abs <= 5e-2
        assert nrmse <= 5e-3
        assert cosine >= 0.999
