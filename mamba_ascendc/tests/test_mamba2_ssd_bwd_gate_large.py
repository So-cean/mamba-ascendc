"""Large-shape determinism and precision regression for the fused gate bwd."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest
import torch_npu  # noqa: F401


_ROOT = Path(__file__).resolve().parents[2]
if not hasattr(torch.ops.mamba_ascend, "mamba2_ssd_bwd_gate"):
    torch.ops.load_library(
        str(
            _ROOT
            / "mamba_ascendc/python/ascend_kernel/ascend_kernel/lib/"
            "libascend_kernel.so"
        )
    )


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-30)
    return (torch.linalg.vector_norm(error) / denominator).item()


def _run_case(batch: int, seqlen: int, heads: int, seed: int):
    headdim = 64
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu_inputs = [
        torch.randn(
            batch, seqlen, heads, headdim,
            generator=generator,
            dtype=torch.float32,
        )
        for _ in range(4)
    ]
    npu_inputs = [value.npu() for value in cpu_inputs]
    torch.npu.synchronize()

    runs = []
    for _ in range(2):
        outputs = torch.ops.mamba_ascend.mamba2_ssd_bwd_gate(*npu_inputs)
        torch.npu.synchronize()
        runs.append(tuple(value.cpu() for value in outputs))

    for first, second in zip(runs[0], runs[1]):
        assert torch.equal(first, second)

    dout, z, y_pre, x = cpu_inputs
    sigmoid = torch.sigmoid(z)
    gy = dout * z * sigmoid
    dz = dout * y_pre * sigmoid * (1.0 + z * (1.0 - sigmoid))
    d_d = (x * gy).sum(dim=(0, 1))
    gy_head = gy.reshape(
        batch, seqlen // 64, 64, heads, headdim
    ).permute(0, 3, 1, 2, 4).contiguous()

    for name, actual, expected, limit in (
        ("gy", runs[0][0], gy_head, 5.0e-4),
        ("dz", runs[0][1], dz, 2.0e-6),
        ("dD", runs[0][2], d_d, 2.0e-6),
    ):
        assert _nrmse(actual, expected) <= limit, name
    assert runs[0][0].dtype == torch.float16


@pytest.mark.parametrize("heads", [1, 7, 8, 9, 32])
def test_mamba2_ssd_bwd_gate_head_block_boundaries(heads: int):
    # Cover both full eight-head blocks and the non-aligned tail path.
    _run_case(batch=1, seqlen=128, heads=heads, seed=20260807 + heads)


def test_mamba2_ssd_bwd_gate_large_reused_ub_is_deterministic():
    # B*K=512 exercises the maximum dD reduction and gives every AIV many
    # consecutive chunks.  The latter catches missing MTE3->MTE2 protection
    # when a core reuses its UB tiles on the next task.
    _run_case(batch=4, seqlen=8192, heads=1, seed=20260807)


def test_mamba2_ssd_bwd_gate_reduce_reuses_sum_after_mte3():
    # 910B3 exposes fewer AIVs than this head count, so every participating
    # core reduces multiple heads and reuses the same sum UB.  This catches a
    # missing MTE3->Vector dependency after writing one head's dD.
    _run_case(batch=1, seqlen=128, heads=96, seed=20260903)


def test_mamba2_ssd_bwd_gate_extreme_z_is_finite_and_accurate():
    """Cover saturation values after switching the kernel to native Sigmoid."""
    batch, seqlen, heads, headdim = 1, 128, 3, 64
    generator = torch.Generator(device="cpu").manual_seed(20260904)
    dout = torch.randn(
        batch, seqlen, heads, headdim, generator=generator, dtype=torch.float32
    )
    y_pre = torch.randn(
        batch, seqlen, heads, headdim, generator=generator, dtype=torch.float32
    )
    x = torch.randn(
        batch, seqlen, heads, headdim, generator=generator, dtype=torch.float32
    )
    saturation = torch.tensor(
        [-1000.0, -100.0, -80.0, -20.0, 0.0, 20.0, 80.0, 100.0, 1000.0],
        dtype=torch.float32,
    )
    z = saturation.repeat(
        (batch * seqlen * heads * headdim + saturation.numel() - 1)
        // saturation.numel()
    )[: batch * seqlen * heads * headdim].reshape(
        batch, seqlen, heads, headdim
    )

    actual = tuple(
        value.cpu()
        for value in torch.ops.mamba_ascend.mamba2_ssd_bwd_gate(
            *(value.npu() for value in (dout, z, y_pre, x))
        )
    )
    torch.npu.synchronize()

    sigmoid = torch.sigmoid(z)
    gy = dout * z * sigmoid
    dz = dout * y_pre * sigmoid * (1.0 + z * (1.0 - sigmoid))
    d_d = (x * gy).sum(dim=(0, 1))
    gy_head = gy.reshape(
        batch, seqlen // 64, 64, heads, headdim
    ).permute(0, 3, 1, 2, 4).contiguous()

    for value in actual:
        assert torch.isfinite(value).all()
    for name, value, expected, limit in (
        ("gy", actual[0], gy_head, 5.0e-4),
        ("dz", actual[1], dz, 2.0e-6),
        ("dD", actual[2], d_d, 2.0e-6),
    ):
        assert _nrmse(value, expected) <= limit, name


def test_mamba2_ssd_bwd_gate_nodd_half_ypre_pipeline_precision():
    """Cover the production H256 no-dD/FP16-y_pre specialization."""
    batch, seqlen, heads, headdim = 1, 128, 9, 64
    generator = torch.Generator(device="cpu").manual_seed(20260905)
    dout = torch.randn(
        batch, seqlen, heads, headdim,
        generator=generator, dtype=torch.float32,
    )
    y_pre = torch.randn(
        batch, seqlen, heads, headdim,
        generator=generator, dtype=torch.float32,
    ).half()
    z = torch.randn(
        batch, seqlen, heads, headdim,
        generator=generator, dtype=torch.float32,
    ) * 8.0

    actual_gy, actual_dz = (
        value.cpu()
        for value in torch.ops.mamba_ascend.mamba2_ssd_bwd_gate_nodd(
            dout.npu(), z.npu(), y_pre.npu()
        )
    )
    torch.npu.synchronize()

    sigmoid = torch.sigmoid(z)
    expected_gy = (dout * z * sigmoid).reshape(
        batch, seqlen // 64, 64, heads, headdim
    ).permute(0, 3, 1, 2, 4).contiguous().half()
    expected_dz = (
        dout * y_pre.float() * sigmoid * (1.0 + z * (1.0 - sigmoid))
    )
    assert _nrmse(actual_gy.float(), expected_gy.float()) <= 5.0e-4
    assert _nrmse(actual_dz, expected_dz) <= 2.0e-6
