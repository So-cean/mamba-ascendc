# Mamba-2 H256 Forward Profiling Architecture Report

## 1. Configuration Context

| Item | Value |
|---|---|
| Device | Ascend 910B3 |
| Operator | Mamba-2 SSD training forward |
| Shape `[B,L,H,P,N,chunk,G]` | `[8,4096,256,64,64,64,64]` |
| Public dtype | FP32; Cube operands FP16 with FP32 accumulation |
| Capture | `wait=0, warmup=5, active=5, repeat=1` |
| Profiling scope | Five complete active forward steps |
| Device streams | One active compute stream, `stream 2` |
| Capture span | `169.835250 ms` |
| Device busy union | `169.816000 ms` |
| Device gap | `0.019250 ms` (`0.011335%`) |
| Formal latency | Device-event p50 `31.888220 ms` |

This is a custom SSD operator trace rather than a Transformer layer trace. The report therefore treats the four public forward stages as architecture boundaries. There is no FIA marker, tensor-parallel communication, prefill/decode split, or multi-layer model structure in this capture.

## 2. Model Architecture Determination

| Evidence | Value | Interpretation |
|---|---:|---|
| Complete active steps | `5` | Five independent executions of one Mamba-2 SSD forward |
| `mamba2_ssd_preprocess` per step | `1` | Public-layout preprocessing boundary |
| `Mamba2SsdChunkMix` per step | `1` | Chunk-local SSD compute boundary |
| `Transpose` per step | `1` | Explicit producer-to-consumer layout boundary |
| `Mamba2SsdStateEpilogueTrain` per step | `1` | Recurrent state, projection, gate and final-state boundary |
| Repeated stage order | `5/5` identical | Stable four-stage forward architecture |

The profiler evidence identifies one operator invocation as:

```text
Preprocess → ChunkMix → Transpose → StateEpilogueTrain
```

No inference about a surrounding model's layer count is made because the trace intentionally contains only the SSD operator.

## 3. Forward Pass Boundaries

The five active steps occupy a contiguous device interval. Absolute timestamps are normalized to the first active kernel because public reports remove machine-specific absolute timestamps.

| Pass | Stage sequence | Service wall time | Device busy time | Internal gaps |
|---:|---|---:|---:|---:|
| 1 | Preprocess → ChunkMix → Transpose → StateEpilogue | `33.9230 ms` | `33.9205 ms` | `0.0025 ms` |
| 2 | Same | `34.0105 ms` | `34.0069 ms` | `0.0036 ms` |
| 3 | Same | `33.9543 ms` | `33.9512 ms` | `0.0031 ms` |
| 4 | Same | `34.0103 ms` | `34.0071 ms` | `0.0031 ms` |
| 5 | Same | `33.9340 ms` | `33.9306 ms` | `0.0034 ms` |
| Mean | Same | `33.9664 ms` | `33.9633 ms` | `0.0031 ms` |

The pass-to-pass service range is `0.0875 ms`, or `0.26%` of the mean. The formal device-event latency is lower than the profiler mean by about `6.5%`; cross-platform performance tables therefore use device events, while this trace is used only for decomposition and hardware analysis.

## 4. Stage Classification

| Stage type | Count per pass | Engine mapping | Characteristics |
|---|---:|---|---|
| Vector preprocess | `1` | AIV + MTE2/MTE3 | softplus, clamp, exp/cumsum and layout production |
| Mixed chunk compute | `1` | AIC + AIV | chunk-local matrix products plus causal/vector work |
| Layout transform | `1` | DMA/vector runtime path | converts an intermediate to the state consumer layout |
| Mixed state epilogue | `1` | AIC + AIV | recurrent state, projection, D/z gate and final-state store |

## 5. Cross-Verification Table

| Operation | Passes 1–5 count | Expected count | Verification |
|---|---:|---:|---|
| `mamba2_ssd_preprocess` | `5` | `5` | PASS |
| `Mamba2SsdChunkMix` | `5` | `5` | PASS |
| `Transpose` | `5` | `5` | PASS |
| `Mamba2SsdStateEpilogueTrain` | `5` | `5` | PASS |
| Communication kernels | `0` | `0` | PASS |
| Host-visible device bubbles ≥ `0.1 ms` | `0` | `0` | PASS |

## 6. Per-Stage Sub-Structure

### Preprocess

```text
FP32 public tensors
├─ dt bias, softplus and clamp
├─ exp / cumulative decay terms
├─ x, B and C consumer-layout production
└─ aligned GM workspace stores
```

| Metric | Value |
|---|---:|
| Mean duration | `6.666683 ms` |
| AIV vector ratio | `42.800%` |
| AIV scalar ratio | `54.822%` |
| AIV MTE2 ratio | `26.533%` |
| AIV MTE3 ratio | `8.778%` |

### ChunkMix

```text
Preprocessed GM workspaces
├─ AIV causal/decay preparation
├─ AIC chunk-local GEMMs
├─ AIV post-processing
└─ y_diag and chunk-state workspace stores
```

| Metric | Value |
|---|---:|
| Mean duration | `14.849678 ms` |
| Cube active ratio | `99.108%` |
| AIC MAC ratio | `16.100%` |
| AIC MTE2 ratio | `23.133%` |
| AIV scalar ratio | `56.400%` |

### Transpose

```text
ChunkMix output → consumer-major state workspace
```

The representative step spends `0.117480 ms` in the explicit transpose. Its small latency does not include the larger cost of writing and rereading producer workspaces around the transform.

### StateEpilogueTrain

```text
Chunk-local states and y_diag
├─ recurrent state propagation
├─ state-to-output projection
├─ D residual and z/silu gate
└─ FP32 output + final recurrent state
```

| Metric | Value |
|---|---:|
| Mean duration | `12.302741 ms` |
| Cube active ratio | `98.336%` |
| AIC MAC ratio | `2.900%` |
| AIV MTE2 ratio | `76.467%` |
| AIV MTE3 ratio | `22.456%` |

**The largest architecture-level opportunity is StateEpilogue data movement and consumer layout, not additional kernel fusion for launch reduction.** Cube is continuously scheduled, but effective MAC work is sparse and AIV input movement dominates.

## 7. Decode Phase Analysis

Not applicable. This capture profiles a fixed-length training forward of the SSD operator. It contains neither autoregressive decode iterations nor KV-cache operations. A decode-specific conclusion would require a separate incremental-state workload and trace.

## 8. Communication Pipeline Structure

| Stream | Purpose | Overlap |
|---|---|---|
| `stream 2` | All four public forward stages | AIC/AIV/MTE overlap occurs inside MIX kernels; no cross-stream communication overlap is present |

```text
stream 2: [Preprocess 6.609 ms][ChunkMix 14.985 ms][T 0.117 ms][StateEpilogue 12.210 ms]
           └──── AIV ────────┘└──── internal AIC/AIV pipeline ───┘└─ AIC/AIV/MTE ─┘
device:   busy ─────────────────────────────────────────────────────────────────────
gaps:                           1.2 µs              0.61 µs             0.77 µs
```

There is no HCCL or inter-device communication. Cross-kernel stages execute serially because each consumes the previous stage's GM workspace. The measured internal gaps total only `0.0031 ms` per `33.9664 ms` step.

## 9. Stage-to-Stage Variation

| Metric | Preprocess | ChunkMix | Transpose | StateEpilogue |
|---|---:|---:|---:|---:|
| Mean Level2 duration | `6.667 ms` | `14.850 ms` | `0.117 ms` representative | `12.303 ms` |
| Share of profiled stage time | `19.6%` | `43.8%` | `0.3%` | `36.3%` |
| Cube active | `0.0%` | `99.1%` | N/A | `98.3%` |
| AIC MAC | `0.0%` | `16.1%` | N/A | `2.9%` |
| AIV MTE2 | `26.5%` | `27.8%` | N/A | `76.5%` |
| Primary constraint | Vector/control | mixed compute + movement | layout boundary | movement + low-effective-MAC projection |

The five-pass service-time variation is only `0.26%`; there is no isolated slow pass or recurring bubble. Optimization should therefore target per-stage work rather than scheduler stability.

## 10. Model Architecture Summary

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Mamba-2 SSD training forward · [8,4096,256,64,64,64,64]            │
├──────────────────────────────────────────────────────────────────────┤
│ FP32 inputs                                                          │
│   └─ Preprocess (AIV)                                                │
│       └─ ChunkMix (AIC + AIV)                                        │
│           └─ Transpose / consumer layout                             │
│               └─ StateEpilogueTrain (AIC + AIV)                     │
│                   ├─ FP32 output                                     │
│                   └─ final recurrent state                           │
├──────────────────────────────────────────────────────────────────────┤
│ No HCCL · one compute stream · internal gap ratio 0.0092%            │
│ Main bottleneck: cross-kernel materialization and StateEpilogue MTE2 │
└──────────────────────────────────────────────────────────────────────┘

0 ms          6.61 ms                    21.59 ms   21.71 ms          33.92 ms
|──────────────|────────────────────────────|──────────|─────────────────|
[ Preprocess ][          ChunkMix          ][Transpose][ StateEpilogue ]
```

The anomaly record is stored in [`mamba2_h256_fwd_anomaly_20260810.json`](mamba2_h256_fwd_anomaly_20260810.json). It reports no material host underfeed or device bubble and assigns high confidence to a device-kernel-bound diagnosis.
