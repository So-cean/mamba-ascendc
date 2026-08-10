# Mamba2 SSD backward diagonal finalize test cases

## Test configuration

- `SUPPORTED_DTYPES = [float16 inputs with float32 d_a/output]`
- `TEST_SHAPES = [(1,1,1,1), (1,3,2,1), (1,4,2,2),
  (1,8,2,2), (1,5,2,1), (2,8,4,2)]`
- `GENERAL_SHAPES = [(1,2,8,1), (2,4,1,4), (4,16,8,4)]`
- `(B,H,K,G)` requires positive values and `H % G == 0`.
- Matrix dimensions are fixed to 64.

Each shape covers nominal, zero-decay, strong-decay, small-signal and signed
random values. The performance suite additionally covers
`(B,H,K,G)=(8,32,64,8)`.

Head-block-specific coverage is:

- `(1,1,1,1)`: small-shape single-head fallback;
- `(1,3,2,1)`: one short tail block;
- `(1,8,2,2)`: exact four-head block per group;
- `(1,5,2,1)`: one full block followed by a one-head tail.

## Baseline

The NPU baseline is the PyTorch tensor composition from `design.md`: FP32
casts, elementwise Mul/Add, row/column sums, head-to-group ReduceSum and
layout conversion. The custom call is
`torch.ops.mamba_ascend.mamba2_ssd_bwd_diag_finalize`.

Precision gates use NRMSE `<2e-3` and cosine similarity `>0.99999`, matching
the parent DiagState operator's mixed-precision contract. Repeated execution
must be bitwise deterministic.
