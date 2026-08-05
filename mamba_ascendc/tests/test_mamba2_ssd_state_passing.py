import torch

import ascend_kernel


def _reference(chunk_states, da_cumsum, initial_states=None):
    batch, nheads, nchunks, headdim, dstate = chunk_states.shape
    if initial_states is None:
        state = torch.zeros(
            batch, nheads, headdim, dstate,
            dtype=chunk_states.dtype,
            device=chunk_states.device,
        )
    else:
        state = initial_states.clone()
    starts = []
    for chunk in range(nchunks):
        starts.append(state)
        state = (
            torch.exp(da_cumsum[:, :, chunk, -1]).unsqueeze(-1).unsqueeze(-1)
            * state
            + chunk_states[:, :, chunk]
        )
    return torch.stack(starts, dim=2), state


def test_mamba2_ssd_state_passing_precision():
    torch.manual_seed(20260801)
    chunk_states = torch.randn(2, 8, 16, 64, 128, device="npu") * 0.1
    da = -(0.01 + 0.1 * torch.rand(2, 8, 16, 128, device="npu"))
    da_cumsum = torch.cumsum(da, dim=-1)
    initial = torch.randn(2, 8, 64, 128, device="npu") * 0.1
    expected_starts, expected_final = _reference(chunk_states, da_cumsum, initial)
    actual_starts, actual_final = torch.ops.mamba_ascend.mamba2_ssd_state_passing(
        chunk_states, da_cumsum, initial
    )
    torch.testing.assert_close(actual_starts, expected_starts, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual_final, expected_final, rtol=2e-4, atol=2e-5)
