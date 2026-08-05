# mamba2_ssd_fwd test cases

This is the unified source for Phase 4 smoke tests, Phase 7 precision evaluation and
Phase 8 performance evaluation.

## 1. Configuration

```python
SUPPORTED_DTYPES = [torch.float32]

# Tuple order used below:
# (B, L, H, P, N, chunk_size, G)
```

Common feature presets:

```python
BASIC = dict(D=False, z=False, dt_bias=False, dt_softplus=False,
             initial_states=False, dt_limit=(0.0, float("inf")))
D_ONLY = {**BASIC, "D": True}
Z_ONLY = {**BASIC, "z": True}
INITIAL = {**BASIC, "initial_states": True}
SOFTPLUS = {**BASIC, "dt_bias": True, "dt_softplus": True,
            "dt_limit": (0.005, 0.2)}
ALL = dict(D=True, z=True, dt_bias=True, dt_softplus=True,
           initial_states=True, dt_limit=(0.005, 0.2))
```

The signed engineering stress suite uses CPU
`torch.Generator().manual_seed(42)`, FP32, with:

```text
x, z, D, initial_state ~ normal
dt ~ uniform(0.01, 0.11)
A  ~ -uniform(0.1, 0.5)
B, C ~ normal / 5
dt_bias ~ normal * 0.1
```

The strict MERE/MARE suite uses bounded positive `x/B/C/D/z/initial_state` inputs for
the same shapes and features. This avoids random signed reduction cancellation near
zero, where a sub-micro absolute error can produce an unbounded pointwise relative
error. Signed inputs remain mandatory in the engineering `assert_close` suite and are
not replaced by the positive-domain evaluation.

## 2. TEST_CASES

Regular tensors keep `x.numel() <= 200K`.

| ID | Category | Shape `(B,L,H,P,N,C,G)` | Features | Purpose |
|---|---|---|---|---|
| R01 | small | (1,32,2,8,8,16,1) | BASIC | minimum regular recurrence |
| R02 | small | (1,32,2,8,8,16,1) | D_ONLY | vector D epilogue |
| R03 | small | (1,32,2,8,8,16,1) | Z_ONLY | SiLU gate |
| R04 | small | (1,32,2,8,8,16,1) | INITIAL | nonzero initial state |
| R05 | small | (1,32,2,8,8,16,1) | SOFTPLUS | bias/softplus/clamp |
| R06 | small | (1,32,2,8,8,16,1) | ALL | all feature interaction |
| R07 | tail | (1,65,4,16,16,32,2) | BASIC | tail token mask |
| R08 | tail | (1,65,4,16,16,32,2) | D_ONLY | tail plus D |
| R09 | tail | (1,65,4,16,16,32,2) | Z_ONLY | tail plus z |
| R10 | tail | (1,65,4,16,16,32,2) | INITIAL | tail plus initial state |
| R11 | tail | (1,65,4,16,16,32,2) | SOFTPLUS | tail preprocessing |
| R12 | tail | (1,65,4,16,16,32,2) | ALL | tail all features |
| R13 | multi-group | (2,128,8,32,32,64,2) | BASIC | head-to-group mapping |
| R14 | multi-group | (2,128,8,32,32,64,2) | D_ONLY | grouped D epilogue |
| R15 | multi-group | (2,128,8,32,32,64,2) | Z_ONLY | grouped z gate |
| R16 | multi-group | (2,128,8,32,32,64,2) | INITIAL | grouped initial state |
| R17 | multi-group | (2,128,8,32,32,64,2) | SOFTPLUS | grouped preprocessing |
| R18 | multi-group | (2,128,8,32,32,64,2) | ALL | grouped all features |

## 3. GENERAL_CASES

| ID | Category | Shape `(B,L,H,P,N,C,G)` | Features | Purpose |
|---|---|---|---|---|
| G01 | chunk128 | (1,256,8,32,64,128,2) | BASIC | user chunk larger than micro-chunk |
| G02 | chunk128 | (1,256,8,32,64,128,2) | ALL | chunk128 all features |
| G03 | aligned | (1,128,2,64,64,64,1) | BASIC | mixed Key 1 |
| G04 | aligned | (1,128,2,64,64,64,1) | ALL | mixed Key 1 all features |
| G05 | aligned-128 | (1,128,2,64,64,128,1) | BASIC | chunk API partition invariance |
| G06 | aligned-128 | (1,128,2,64,64,128,1) | ALL | partition invariance with features |
| G07 | dstate128 | (1,128,2,64,128,128,1) | BASIC | 32 KiB persistent state |
| G08 | dstate128 | (1,128,2,64,128,128,1) | ALL | maximum V1 state with features |
| G09 | single-token | (1,1,1,8,8,16,1) | BASIC | smallest valid L |
| G10 | odd-L | (1,31,2,16,16,16,1) | ALL | L smaller than one micro-chunk |
| G11 | head-group-ratio | (1,96,16,16,32,32,1) | BASIC | 16 heads sharing one B/C group |
| G12 | many-groups | (1,96,16,16,32,32,8) | ALL | two heads per group |

Coverage count is `(18 regular + 12 general) * 1 dtype = 30` cases before boundary
variants. Feature expansion is explicit in the table rather than implicit test nesting.

## 4. Boundary cases

Boundary cases run on R01 unless another shape is specified.

| ID | Input modification | Expected property |
|---|---|---|
| B01 | x = 0 | output is zero except no hidden contribution; final state decays |
| B02 | B = 0 | no new state contribution |
| B03 | C = 0, D absent | output zero after z gate |
| B04 | A = -1e-4 | near-identity decay |
| B05 | A = -10 | strong but finite decay |
| B06 | raw dt = -20, softplus enabled | stable positive softplus, no overflow |
| B07 | raw dt = 20, softplus enabled | stable large-input softplus then clamp |
| B08 | dt_limit min=max=0.05 | constant discretization interval |
| B09 | z = -20 | near-zero SiLU gate |
| B10 | z = 0 | exact zero gate |
| B11 | z = 20 | large positive finite gate |
| B12 | initial_state = 0 | identical to absent initial state |
| B13 | initial_state = 1 | deterministic nonzero state propagation |
| B14 | D shape `[H]` | scalar-per-head broadcast |
| B15 | D shape `[H,P]` | channel-wise D broadcast |

Invalid-input host tests:

- H not divisible by G;
- B/C shape mismatch;
- wrong rank/dtype/device;
- noncontiguous input in the low-level op;
- chunk_size <= 0;
- dt_limit_min > dt_limit_max;
- optional tensor shape mismatch;
- zero sequence length (outside V1).

## 5. Baselines

### 5.1 Custom NPU call

```python
out, final_state = torch.ops.mamba_ascend.mamba2_ssd_fwd(
    x, dt, A, B, C, D, z, dt_bias, initial_states,
    chunk_size, dt_softplus, dt_limit[0], dt_limit[1]
)
```

The public Python wrapper converts `None` to the registered optional arguments and drops
`final_state` when `return_final_state=False`.

### 5.2 CPU oracle

```python
out_ref, final_ref = mamba_torch.ssd_reference.ssd_chunk_scan_ref(
    x.cpu(), dt.cpu(), A.cpu(), B.cpu(), C.cpu(), chunk_size,
    D=cpu(D), z=cpu(z), dt_bias=cpu(dt_bias),
    dt_softplus=dt_softplus, dt_limit=dt_limit,
    initial_states=cpu(initial_states), return_final_state=True,
)
```

### 5.3 NPU performance baselines

1. frozen Triton-Ascend `mamba_chunk_scan_combined(..., backend="triton")`;
2. NPU PyTorch composition `ssd_chunk_scan_ref` for diagnostic comparison;
3. Official `mamba_ssm` on A100 for the final GPU reference result.

## 6. Precision metrics and pass rules

For both output and final state, report:

- MaxAbsErr / MeanAbsErr;
- MERE / MARE;
- NRMSE;
- cosine similarity;
- NaN/Inf count.

Engineering compatibility Gate:

```text
torch.testing.assert_close(rtol=1e-2, atol=3e-3)
output NRMSE <= 2.401e-3
final-state NRMSE <= 2.900e-4
NaN/Inf = 0
```

MERE/MARE are reported separately against the ecosystem FP32 threshold. Passing the
engineering Gate must not be described as strict pure-FP32 accuracy when BF16 Cube
operands are selected.

## 7. Performance cases

All use ALL features, FP32 API, seed 20260801.

| Case | Shape `(B,L,H,P,N,C,G)` | Triton 910B | A30 median | AscendC Gate |
|---|---|---:|---:|---:|
| tiny | (1,128,2,64,64,64,1) | 0.166596 ms | 0.601600 ms | first record, no regression hiding |
| small | (2,512,8,64,64,64,1) | 0.537070 ms | 0.605184 ms | first record, no regression hiding |
| medium | (4,2048,16,64,128,128,4) | 4.245598 ms | 0.920576 ms | <=4.245598 ms after optimization |

Profiler schedule is `warmup=5, active=5` for skill compliance, with a second 10-repeat
msprof run for continuity with the Triton report. Report kernel-only device duration,
wrapper end-to-end device chain and workspace bytes.

## 8. Coverage Gate

- [x] every supported dtype is covered by every listed case;
- [x] `(TEST_CASES + GENERAL_CASES) * dtypes = 30`;
- [x] regular shapes respect the <=200K input-element guideline;
- [x] D/z/bias/softplus/initial/final/tail/multi-group are covered;
- [x] CPU, Triton NPU, PyTorch NPU, A30 and future A100 baselines are named.
