# Mamba2 SSD backward OffPrepare design

## Interface and math

Inputs use the internal head-major layout:

- `gy`: FP16 `[B,H,K,64,64]`;
- `states_start`: FP16 or FP32 `[B,H,K,64,64]`;
- `dA_cumsum`: FP32 `[B,H,K,64]`.

The operator produces FP16 Cube operands:

```text
q[b,h,k,t,p] = gy[b,h,k,t,p] * exp(dA_cumsum[b,h,k,t])
state_half = cast_fp16(states_start)
```

When `states_start` is already FP16, `state_half` aliases the input and the kernel does
not read or write it. Exp and Mul always execute in FP32 before the final FP16 cast.

## Inter-core tiling

One logical task owns consecutive chunks of one `(b,h)` pair. The host queries the UB
capacity and chooses the largest legal `chunks_per_task`, capped at 3 because the
Broadcast row dimension must stay below 256:

```text
chunk_tiles = ceil_div(K, chunks_per_task)
task_count = B * H * chunk_tiles
block_dim = min(task_count, GetCoreNumAiv())
```

For the heavy training path (`states_start` already FP16), three chunks make each main
`gy/q` DMA 24 KiB instead of 8 KiB and reduce per-task synchronization by about 3x.
FP32-state compatibility consumes additional UB, so the host derives a smaller tile
from the same queried capacity when necessary.

## UB allocation

Per chunk in the normal FP16-state path:

| Buffer | Dtype | Elements/chunk | Bytes/chunk |
|---|---|---:|---:|
| gy input | FP16 | 4096 | 8192 |
| gy compute | FP32 | 4096 | 16384 |
| dA | FP32 | 64 | 256 |
| decay | FP32 | 4096 | 16384 |
| q output | FP16 | 4096 | 8192 |
| Broadcast workspace | byte | 8192 | 8192 |
| **Total** |  |  | **57600** |

Three chunks require 172800 bytes (168.75 KiB), below the 192 KiB 910B3 UB limit.
FP32-state compatibility adds 24576 bytes per chunk; the host computes the legal tile
count from `GetCoreMemSize(UB)` instead of hardcoding a device capacity.

## Data movement and synchronization

- GM/UB traffic uses `DataCopyPad` only.
- Main tiles are contiguous in `K` for a fixed `(b,h)`, so one DMA descriptor moves all
  active chunks.
- `MTE2_V` orders input visibility, `V_MTE3` orders output production, and the existing
  final full-pipeline fence protects UB reuse across logical tasks.
- Tail tiles use the exact active chunk count and never read outside the tensor.

## Correctness and performance gates

- Independent OffPrepare cases cover FP16/FP32 state input, `K=1/2/3/4/64/65`, multiple
  batches and heads, deterministic repeat execution, and numerical boundaries.
- Full backward public gradient matrix must pass 41/41 on 910B3 and 950PR.
- H256 uses `[B,L,H,P,N,C,G]=[8,4096,256,64,64,64,64]`, device Event,
  `warmup=10`, `repeat=50`, p50.
- The candidate is accepted only if both platforms are non-regressing and at least one
  platform improves end-to-end BWD by 1%; otherwise it is reverted.

## Accepted dual-platform result

The consecutive-chunk implementation passed the independent 8-case operator test and
the full 41-case public-gradient matrix on both Ascend 910B3 and Ascend 950PR.

| Platform | OffPrepare before | OffPrepare after | BWD before | BWD after |
|---|---:|---:|---:|---:|
| Ascend 910B3 | 4.909 ms | 4.582 ms | 89.617 ms | 89.333 ms |
| Ascend 950PR | 2.861 ms | 1.879 ms | 71.967 ms | 71.157 ms |

The Level1 stage reduction is 6.7% on 910B3 and 34.3% on 950PR. End-to-end BWD
improves 0.32% and 1.14%, respectively. The result satisfies the shared-source gate:
neither platform regresses and 950PR exceeds the 1% acceptance threshold.
