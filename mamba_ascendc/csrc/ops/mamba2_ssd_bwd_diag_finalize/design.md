# Mamba2 SSD backward diagonal finalize design

## Interface

```text
mamba2_ssd_bwd_diag_finalize(
    dx_diag, dr, db_diag_group, db_state, dc_diag_group, dw, w, r,
    d_a_cumsum, groups
) -> (dx, db_group, dc_group, g_d_a_cumsum)
```

`dx_diag`, `dr`, `db_state`, `dw`, `w`, and `r` are contiguous FP16
`[B,H,K,64,64]` tensors.  The already reduced `db_diag_group` and
`dc_diag_group` are contiguous FP16 `[B,G,K,64,64]`. `d_a_cumsum` is
contiguous FP32 `[B,H,K,64]`; `groups` divides `H`. Outputs use FP32
accumulation:

- `dx`: `[B,H,K,64,64]`
- `db_group`, `dc_group`: `[B,K,G,64,64]`
- `g_d_a_cumsum`: `[B,H,K,64]`

This is an internal fixed-shape Mamba-2 M1 backward operator for Ascend
910B. It has no equivalent single PyTorch API.

## Mathematics

For every head task:

```text
decay[t] = exp(d_a[63] - d_a[t])
dx       = fp32(dx_diag) + fp32(dr) * decay[:, None]

g_diag   = rowsum(fp32(dw) * fp32(w))
           - colsum(fp32(dw) * fp32(w))
state    = rowsum(fp32(dr) * fp32(r))
g        = g_diag - state
g[63]   += sum(state)
```

For every `(B,G,K)` owner task:

```text
db_group = fp32(db_diag_group) + sum_heads(fp32(db_state))
dc_group = fp32(dc_diag_group)
```

## Execution and tiling

This is a standalone AIV kernel. Block tiling assigns one `(B,G,K)` task to
an AIV and loops strided by the runtime queried AIV core count. A group owner
streams all heads in the group, so dB/dC need no atomics or framework
`ReduceSum`/`Transpose` kernels.

Within one group owner, heads are tiled in blocks of at most four. Public
head-owned inputs are laid out `[B,H,K,...]`, so adjacent heads at a fixed chunk are
strided in GM. One `DataCopyPad` descriptor uses `blockCount=headBlock`, an
8-KiB matrix `blockLen`, and a `(K-1)*matrixBytes` source gap to compact the
head block in UB. Thus the H256 stress shape (`H/G=4`) replaces four 8-KiB
matrix DMA commands with one 32-KiB burst for every FP16 input tensor. `dA`
uses the same head-block copy and the four `g` vectors are emitted by one
strided store.

`headBlock=min(H/G,4)`. `H/G=1` is the unchanged small-shape fallback.
When `H/G` is not divisible by four, the last descriptor uses its exact tail
`blockCount`; no padded head participates in compute or output.

The two group-owned diagonal inputs are loaded once before the head loop.
Only `db_state` is accumulated per head, eliminating two head-owned matrix
reads, casts and reductions. Each task is fixed to a 64x64 tile. GM matrices are naturally 512-byte
aligned. All GM/UB transfers use `DataCopyPad`. FP16 inputs are cast to FP32
before Mul/Add/reduction. Row reductions use `WholeReduceSum`; the product is
backed up before the row reduction when it is also needed by the column tree.

### UB allocation

Worst case is `headBlock=4`:

| Buffer | Count | Bytes each | Total |
|---|---:|---:|---:|
| FP16 head-block matrix scratch | 3 | 32,768 | 98,304 |
| FP32 matrix scratch/product | 3 | 16,384 | 49,152 |
| FP32 dB/dC accumulators | 2 | 16,384 | 32,768 |
| Block-local dA/state-row/g vectors | 3 | 1,024 | 3,072 |
| Single-head decay/diag vectors | 2 | 256 | 512 |
| Broadcast temporary | 1 | 8,192 | 8,192 |
| **Total** | | | **192,000 B** |

Ascend 910B provides 192 KiB (196,608 B) UB, leaving 4,608 B. The host
computes the required size from the selected head block instead of always
reserving the maximum:

| `headBlock` | Required UB | Remaining from 192 KiB |
|---:|---:|---:|
| 1 | 115,968 B | 80,640 B |
| 2 | 141,312 B | 55,296 B |
| 3 | 166,656 B | 29,952 B |
| 4 | 192,000 B | 4,608 B |

The host queries UB size and rejects the launch when the selected working set
does not fit. A single slot is intentional: every owner task performs several
dependent reductions. Double-slot task pipelining cannot coexist with the
four-head block in 192 KiB and remains a separate future experiment.

## Precision and determinism

- Batched Cube operands/results are FP16; all finalize arithmetic is FP32.
- No atomics are used; each group output has exactly one writer.
- dB/dC heads are accumulated in a deterministic increasing-head order.
- Head blocking changes DMA issue granularity only; compute and accumulation
  retain the original increasing-head order bit for bit.
- Reused UB buffers have explicit MTE2/Vector/MTE3 hard-event ordering.

## Performance gate

1. Compare against the exact NPU tensor composition with fixed
   `warmup=5, active=5` profiler steps.
2. The recorded H256 profile is approximately 15.3 ms per step. Re-profile the identical
   case after build; the expected gain must come from fewer/larger MTE2 DMA
   commands, not a changed formula or output layout.
3. Verify the single-head fallback and head-block tails `3`, `4`, and `4+1`
   before accepting the H256 result.
