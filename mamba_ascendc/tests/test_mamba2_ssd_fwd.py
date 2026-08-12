"""Functional and first-gate precision tests for native Mamba-2 SSD forward."""

import pytest
import torch

from mamba_torch.ssd_reference import ssd_chunk_scan_ref


def _make_case(
    batch=1,
    seqlen=8,
    nheads=2,
    headdim=4,
    dstate=8,
    ngroups=1,
    optional=False,
    initial=False,
):
    generator = torch.Generator().manual_seed(
        1000 + batch + seqlen + nheads + headdim + dstate + ngroups
    )
    x = torch.randn(batch, seqlen, nheads, headdim, generator=generator)
    dt = 0.1 * torch.randn(batch, seqlen, nheads, generator=generator)
    A = -(0.2 + torch.rand(nheads, generator=generator))
    B = 0.2 * torch.randn(batch, seqlen, ngroups, dstate, generator=generator)
    C = 0.2 * torch.randn(batch, seqlen, ngroups, dstate, generator=generator)
    kwargs = {}
    if optional:
        kwargs.update(
            D=0.1 * torch.randn(nheads, headdim, generator=generator),
            z=torch.randn(batch, seqlen, nheads, headdim, generator=generator),
            dt_bias=0.1 * torch.randn(nheads, generator=generator),
            dt_softplus=True,
            dt_limit=(0.001, 2.0),
        )
    if initial:
        kwargs["initial_states"] = 0.1 * torch.randn(
            batch, nheads, headdim, dstate, generator=generator
        )
    return (x, dt, A, B, C), kwargs


def _to_npu(value):
    if isinstance(value, torch.Tensor):
        return value.npu().contiguous()
    return value


def _assert_close(actual, expected, name):
    actual_cpu = actual.cpu().float()
    expected_cpu = expected.cpu().float()
    diff = (actual_cpu - expected_cpu).abs()
    assert torch.allclose(actual_cpu, expected_cpu, rtol=2e-4, atol=2e-5), (
        f"{name}: max_abs={diff.max().item():.6e}, "
        f"mean_abs={diff.mean().item():.6e}"
    )


def test_functional_call():
    import ascend_kernel

    args, kwargs = _make_case(batch=1, seqlen=2, nheads=1, headdim=2, dstate=4)
    npu_args = tuple(_to_npu(value) for value in args)
    out, final_state = ascend_kernel.mamba2_ssd_fwd(
        *npu_args, chunk_size=2, return_final_state=True
    )
    assert out.shape == args[0].shape
    assert final_state.shape == (1, 1, 2, 4)
    assert out.device.type == "npu"
    assert final_state.device.type == "npu"
    assert torch.isfinite(out).all().item()
    assert torch.isfinite(final_state).all().item()


@pytest.mark.parametrize(
    "case",
    [
        dict(batch=1, seqlen=8, nheads=2, headdim=4, dstate=8, ngroups=1),
        dict(batch=2, seqlen=7, nheads=4, headdim=3, dstate=5, ngroups=2),
        dict(batch=1, seqlen=9, nheads=2, headdim=8, dstate=8, ngroups=1,
             optional=True),
        dict(batch=2, seqlen=5, nheads=4, headdim=4, dstate=6, ngroups=2,
             optional=True, initial=True),
    ],
)
def test_precision(case):
    import ascend_kernel

    args, kwargs = _make_case(**case)
    ref_out, ref_final = ssd_chunk_scan_ref(
        *args,
        chunk_size=4,
        return_final_state=True,
        **kwargs,
    )
    npu_args = tuple(_to_npu(value) for value in args)
    npu_kwargs = {key: _to_npu(value) for key, value in kwargs.items()}
    npu_out, npu_final = ascend_kernel.mamba2_ssd_fwd(
        *npu_args,
        chunk_size=4,
        return_final_state=True,
        **npu_kwargs,
    )
    _assert_close(npu_out, ref_out, "out")
    _assert_close(npu_final, ref_final, "final_state")


@pytest.mark.parametrize("logical_chunk,seqlen", [(256, 256), (512, 512)])
def test_large_logical_chunk_uses_validated_microtiles(
    monkeypatch, logical_chunk, seqlen
):
    """A logical chunk may be executed as multiple 128-token micro-tiles."""
    import ascend_kernel

    monkeypatch.setenv("MAMBA_ASCENDC_CHUNK_MIX", "1")
    monkeypatch.setenv("MAMBA_ASCENDC_CHUNK128", "1")
    args, kwargs = _make_case(
        batch=1,
        seqlen=seqlen,
        nheads=8,
        headdim=64,
        dstate=128,
        ngroups=2,
        optional=True,
        initial=True,
    )
    # The official comparison does not include z gating.
    kwargs.pop("z")
    ref_out, ref_final = ssd_chunk_scan_ref(
        *args,
        chunk_size=logical_chunk,
        return_final_state=True,
        **kwargs,
    )
    npu_args = tuple(_to_npu(value) for value in args)
    npu_kwargs = {key: _to_npu(value) for key, value in kwargs.items()}
    npu_out, npu_final = ascend_kernel.mamba2_ssd_fwd(
        *npu_args,
        chunk_size=logical_chunk,
        return_final_state=True,
        **npu_kwargs,
    )
    assert torch.allclose(npu_out.cpu(), ref_out, rtol=1e-2, atol=3e-3)
    assert torch.allclose(npu_final.cpu(), ref_final, rtol=1e-2, atol=3e-3)
