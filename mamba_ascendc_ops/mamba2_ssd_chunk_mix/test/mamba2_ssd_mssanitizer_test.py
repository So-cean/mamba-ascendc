"""Minimal deterministic launch used by the production mssanitizer job."""

import ctypes
import os

import torch
import torch_npu  # noqa: F401


library = ctypes.CDLL(
    os.environ["MAMBA_CHUNK_MIX_OP_API_LIB"], mode=ctypes.RTLD_GLOBAL
)
assert getattr(library, "aclnnMamba2SsdChunkMix")
assert getattr(library, "aclnnMamba2SsdChunkScanBwdOff")
assert getattr(library, "aclnnMamba2SsdChunkScanBwdDiagState")
assert getattr(library, "aclnnMamba2SsdOffEpilogue")
assert getattr(library, "aclnnMamba2SsdStateEpilogue")
import ascend_kernel  # noqa: E402,F401


def run_chunk_mix():
    torch.manual_seed(20260802)
    x = torch.randn(1, 4, 2, 64, 64, device="npu").half()
    da = -torch.cumsum(
        0.005 + 0.04 * torch.rand(1, 4, 2, 64, device="npu"), dim=-1
    )
    b = (0.1 * torch.randn(1, 2, 1, 64, 128, device="npu")).half()
    c = (0.1 * torch.randn(1, 2, 1, 64, 128, device="npu")).half()
    outputs = torch.ops.mamba_ascend.mamba2_ssd_chunk_mix(x, da, b, c)
    torch.npu.synchronize()
    assert all(torch.isfinite(value).all().item() for value in outputs)


def run_off_epilogue():
    torch.manual_seed(20260803)
    c = (0.1 * torch.randn(1, 2, 1, 64, 128, device="npu")).half()
    states = 0.1 * torch.randn(1, 4, 2, 64, 128, device="npu")
    da = -torch.rand(1, 4, 2, 64, device="npu")
    y = 0.1 * torch.randn(1, 4, 2, 64, 64, device="npu")
    x = torch.randn(1, 128, 4, 64, device="npu")
    d = torch.randn(4, 64, device="npu")
    z = torch.randn_like(x)
    output = torch.ops.mamba_ascend.mamba2_ssd_off_epilogue(
        c, states, da, y, x, d, z
    )
    torch.npu.synchronize()
    assert torch.isfinite(output).all().item()


def run_chunk_scan_bwd_off():
    torch.manual_seed(20260808)
    batch, heads, chunks, groups = 2, 8, 4, 2
    gy = 0.02 + 0.08 * torch.rand(
        batch, heads, chunks, 64, 64, device="npu"
    )
    states = 0.02 + 0.08 * torch.rand(
        batch, heads, chunks, 64, 64, device="npu"
    )
    da = -0.01 - 0.2 * torch.rand(
        batch, heads, chunks, 64, device="npu"
    )
    c = (0.02 + 0.08 * torch.rand(
        batch, chunks, groups, 64, 64, device="npu"
    )).half()
    outputs = torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_off(
        gy, states, da, c
    )
    torch.npu.synchronize()
    # A direct D2H load lets initcheck verify that every output byte was
    # written by the filtered custom kernel.  An NPU-side isfinite/all chain
    # would itself be uninstrumented under --kernel-name and produce false
    # one-byte reports for its boolean outputs.
    host_outputs = tuple(value.cpu() for value in outputs)
    assert all(torch.isfinite(value).all().item() for value in host_outputs)


def run_chunk_scan_bwd_diag_state():
    torch.manual_seed(20260820)
    batch, heads, chunks, groups = 1, 4, 2, 1
    gy = 0.02 + 0.08 * torch.rand(
        batch, heads, chunks, 64, 64, device="npu"
    )
    x = (0.02 + 0.08 * torch.rand(
        batch, heads, chunks, 64, 64, device="npu"
    )).half()
    steps = 0.001 + 0.01 * torch.rand(
        batch, heads, chunks, 64, device="npu"
    )
    da = -steps.cumsum(dim=-1)
    b = (0.02 + 0.08 * torch.rand(
        batch, chunks, groups, 64, 64, device="npu"
    )).half()
    c = (0.02 + 0.08 * torch.rand(
        batch, chunks, groups, 64, 64, device="npu"
    )).half()
    d_chunk = 0.02 + 0.08 * torch.rand(
        batch, heads, chunks, 64, 64, device="npu"
    )
    outputs = torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_diag_state(
        gy, x, da, b, c, d_chunk
    )
    torch.npu.synchronize()
    host_outputs = tuple(value.cpu() for value in outputs)
    assert all(torch.isfinite(value).all().item() for value in host_outputs)


def run_state_epilogue():
    torch.manual_seed(20260804)
    batch, heads, groups, chunks, tile, state_dim = 1, 4, 1, 3, 64, 128
    chunk_states = 0.05 * torch.randn(
        batch, heads, chunks, tile, state_dim, device="npu"
    )
    da = torch.cumsum(
        -(0.001 + 0.01 * torch.rand(
            batch, heads, chunks, tile, device="npu"
        )),
        dim=-1,
    )
    c = (0.1 * torch.randn(
        batch, chunks, groups, tile, state_dim, device="npu"
    )).half()
    y = 0.1 * torch.randn(
        batch, heads, chunks, tile, tile, device="npu"
    )
    x = 0.2 * torch.randn(
        batch, chunks * tile, heads, tile, device="npu"
    )
    d = 0.2 * torch.randn(heads, tile, device="npu")
    z = 0.2 * torch.randn_like(x)
    initial = 0.05 * torch.randn(
        batch, heads, tile, state_dim, device="npu"
    )
    outputs = torch.ops.mamba_ascend.mamba2_ssd_state_epilogue(
        chunk_states, da, c, y, x, d, z, initial
    )
    torch.npu.synchronize()
    assert all(torch.isfinite(value).all().item() for value in outputs)


if __name__ == "__main__":
    mode = os.environ.get("MAMBA_MSSAN_TARGET", "chunk_mix")
    if mode == "chunk_mix":
        run_chunk_mix()
    elif mode == "chunk_scan_bwd_off":
        run_chunk_scan_bwd_off()
    elif mode == "chunk_scan_bwd_diag_state":
        run_chunk_scan_bwd_diag_state()
    elif mode == "off_epilogue":
        run_off_epilogue()
    elif mode == "state_epilogue":
        run_state_epilogue()
    else:
        raise ValueError(f"unsupported MAMBA_MSSAN_TARGET={mode}")
