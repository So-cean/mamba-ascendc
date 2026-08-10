"""Small native-kernel workload for mssanitizer."""

import torch
import torch_npu

import ascend_kernel  # noqa: F401


def main():
    torch.manual_seed(20260807)
    shape = (1, 2, 2, 64, 128)
    states_start = torch.rand(*shape, dtype=torch.float32, device="npu") * 0.1
    d_states_start = torch.rand_like(states_start) * 0.1
    d_a_cumsum = -torch.rand(1, 2, 2, 128, device="npu") * 0.2
    dfinal_state = torch.rand(1, 2, 64, 128, device="npu") * 0.1
    outputs = torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd(
        states_start, d_states_start, d_a_cumsum, dfinal_state
    )
    torch.npu.synchronize()
    assert all(torch.isfinite(output).all().item() for output in outputs)
    print("mamba2_ssd_state_passing_bwd mssanitizer workload passed")


if __name__ == "__main__":
    main()
