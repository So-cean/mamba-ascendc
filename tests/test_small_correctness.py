# Copyright (c) 2024, mamba-triton-ascend authors.
# Small-config correctness tests against CPU PyTorch reference.
# This is the ONLY file that uses CPU reference; all other tests are reference-free.

import pytest
import torch

from mamba_triton_ascend.ops.reference import selective_scan_ref
from mamba_triton_ascend.ops.selective_scan import selective_scan_fn


class TestSmallCorrectness:
    """
    Golden-standard validation on tiny configs.
    Forward diff < 1e-3, backward diff < 1e-2.
    """

    @pytest.fixture
    def tiny_inputs(self):
        """Tiny config: (1, 2, 2, 4)."""
        batch, dim, dstate, seqlen = 1, 2, 2, 4
        torch.manual_seed(42)
        return {
            "batch": batch, "dim": dim, "dstate": dstate, "seqlen": seqlen,
            "u": torch.randn(batch, dim, seqlen),
            "delta": torch.rand(batch, dim, seqlen) * 0.1 + 0.01,
            "A": torch.randn(dim, dstate) * 0.1,
            "B": torch.randn(batch, dstate, seqlen),
            "C": torch.randn(batch, dstate, seqlen),
            "D": torch.randn(dim),
            "z": torch.randn(batch, dim, seqlen),
            "delta_bias": torch.randn(dim) * 0.01,
        }

    @pytest.mark.usefixtures("clear_triton_cache")
    def test_forward_basic(self, npu_available, tiny_inputs):
        if not npu_available:
            pytest.skip("NPU not available")
        inp = tiny_inputs
        # CPU reference
        out_ref = selective_scan_ref(
            inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"],
            delta_softplus=True
        )
        # NPU Triton
        out_tri = selective_scan_fn(
            inp["u"].to("npu"),
            inp["delta"].to("npu"),
            inp["A"].to("npu"),
            inp["B"].to("npu"),
            inp["C"].to("npu"),
            delta_softplus=True
        )
        diff = (out_ref - out_tri.cpu()).abs().max().item()
        assert diff < 1e-3, f"Forward diff too large: {diff}"

    @pytest.mark.usefixtures("clear_triton_cache")
    def test_forward_with_all_params(self, npu_available, tiny_inputs):
        if not npu_available:
            pytest.skip("NPU not available")
        inp = tiny_inputs
        out_ref = selective_scan_ref(
            inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"],
            D=inp["D"], z=inp["z"],
            delta_bias=inp["delta_bias"],
            delta_softplus=True
        )
        out_tri = selective_scan_fn(
            inp["u"].to("npu"),
            inp["delta"].to("npu"),
            inp["A"].to("npu"),
            inp["B"].to("npu"),
            inp["C"].to("npu"),
            D=inp["D"].to("npu"),
            z=inp["z"].to("npu"),
            delta_bias=inp["delta_bias"].to("npu"),
            delta_softplus=True
        )
        diff = (out_ref - out_tri.cpu()).abs().max().item()
        assert diff < 1e-3, f"Forward diff too large: {diff}"

    @pytest.mark.usefixtures("clear_triton_cache")
    def test_backward_basic(self, npu_available, tiny_inputs):
        if not npu_available:
            pytest.skip("NPU not available")
        inp = tiny_inputs
        # CPU reference with autograd
        u_r = inp["u"].clone().requires_grad_(True)
        d_r = inp["delta"].clone().requires_grad_(True)
        A_r = inp["A"].clone().requires_grad_(True)
        B_r = inp["B"].clone().requires_grad_(True)
        C_r = inp["C"].clone().requires_grad_(True)
        out_ref = selective_scan_ref(u_r, d_r, A_r, B_r, C_r, delta_softplus=True)
        out_ref.sum().backward()

        # NPU Triton with autograd
        u_t = inp["u"].clone().to("npu").requires_grad_(True)
        d_t = inp["delta"].clone().to("npu").requires_grad_(True)
        A_t = inp["A"].clone().to("npu").requires_grad_(True)
        B_t = inp["B"].clone().to("npu").requires_grad_(True)
        C_t = inp["C"].clone().to("npu").requires_grad_(True)
        out_tri = selective_scan_fn(u_t, d_t, A_t, B_t, C_t, delta_softplus=True)
        out_tri.sum().backward()

        diffs = {
            "du": (u_r.grad - u_t.grad.cpu()).abs().max().item(),
            "ddelta": (d_r.grad - d_t.grad.cpu()).abs().max().item(),
            "dA": (A_r.grad - A_t.grad.cpu()).abs().max().item(),
            "dB": (B_r.grad - B_t.grad.cpu()).abs().max().item(),
            "dC": (C_r.grad - C_t.grad.cpu()).abs().max().item(),
        }
        max_diff = max(diffs.values())
        for name, diff in diffs.items():
            print(f"  {name} diff: {diff:.2e}")
        assert max_diff < 1e-2, f"Backward diff too large: {max_diff}"

    @pytest.mark.usefixtures("clear_triton_cache")
    def test_backward_with_all_params(self, npu_available, tiny_inputs):
        if not npu_available:
            pytest.skip("NPU not available")
        inp = tiny_inputs
        # CPU reference
        u_r = inp["u"].clone().requires_grad_(True)
        d_r = inp["delta"].clone().requires_grad_(True)
        A_r = inp["A"].clone().requires_grad_(True)
        B_r = inp["B"].clone().requires_grad_(True)
        C_r = inp["C"].clone().requires_grad_(True)
        D_r = inp["D"].clone().requires_grad_(True)
        z_r = inp["z"].clone().requires_grad_(True)
        db_r = inp["delta_bias"].clone().requires_grad_(True)
        out_ref = selective_scan_ref(
            u_r, d_r, A_r, B_r, C_r,
            D=D_r, z=z_r, delta_bias=db_r, delta_softplus=True
        )
        out_ref.sum().backward()

        # NPU Triton
        u_t = inp["u"].clone().to("npu").requires_grad_(True)
        d_t = inp["delta"].clone().to("npu").requires_grad_(True)
        A_t = inp["A"].clone().to("npu").requires_grad_(True)
        B_t = inp["B"].clone().to("npu").requires_grad_(True)
        C_t = inp["C"].clone().to("npu").requires_grad_(True)
        D_t = inp["D"].clone().to("npu").requires_grad_(True)
        z_t = inp["z"].clone().to("npu").requires_grad_(True)
        db_t = inp["delta_bias"].clone().to("npu").requires_grad_(True)
        out_tri = selective_scan_fn(
            u_t, d_t, A_t, B_t, C_t,
            D=D_t, z=z_t, delta_bias=db_t, delta_softplus=True
        )
        out_tri.sum().backward()

        diffs = {
            "du": (u_r.grad - u_t.grad.cpu()).abs().max().item(),
            "ddelta": (d_r.grad - d_t.grad.cpu()).abs().max().item(),
            "dA": (A_r.grad - A_t.grad.cpu()).abs().max().item(),
            "dB": (B_r.grad - B_t.grad.cpu()).abs().max().item(),
            "dC": (C_r.grad - C_t.grad.cpu()).abs().max().item(),
            "dD": (D_r.grad - D_t.grad.cpu()).abs().max().item(),
            "dz": (z_r.grad - z_t.grad.cpu()).abs().max().item(),
            "ddelta_bias": (db_r.grad - db_t.grad.cpu()).abs().max().item(),
        }
        max_diff = max(diffs.values())
        for name, diff in diffs.items():
            print(f"  {name} diff: {diff:.2e}")
        assert max_diff < 1e-2, f"Backward diff too large: {max_diff}"
