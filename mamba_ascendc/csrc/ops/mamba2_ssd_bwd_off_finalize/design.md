# Mamba2 SSD backward off-diagonal finalize

## Scope

This AIV operator finalizes the two global FP16 BatchMatMul outputs used by
the high-task-count backward path.  It reduces the per-head `dC` contribution
to the public group-owned layout and forms the per-token `dA_cumsum`
contribution.

The first BatchMatMul result, `d_states_half`, is already in the exact
head-major logical order required by the next state-passing stage.  The
operator therefore returns a reshaped alias of that tensor.  It must not read,
cast, or copy this result.

## Interface

```text
(d_states, d_c_group, g_da) = mamba2_ssd_bwd_off_finalize(
    d_states_half, d_c_head_half, c_cube)
```

- `d_states_half`: FP16 `[B,G,R,K,64,64]`, contiguous, where `H=G*R`.
- `d_c_head_half`: FP16 `[B,G,R,K,64,64]`, contiguous.
- `c_cube`: FP16 `[B,K,G,64,64]`, contiguous.
- `d_states`: FP16 view `[B,H,K,64,64]` aliasing `d_states_half`.
- `d_c_group`: FP32 public layout `[B,K,64,G,64]`.
- `g_da`: FP32 `[B,H,K,64]`.

Only the latter two outputs are written by the device kernel.  Returning the
FP16 view preserves the exact BMM values and allows the state-passing consumer
to perform one FP16-to-FP32 cast after its GM read.

## Tiling and architecture mapping

- Inter-core task: one `(batch, group, chunk)`.
- `blockDim = min(B*G*K, GetCoreNumAiv())`.
- Each task loads the shared `C[64,64]` once and loops over `R` heads.
- The algorithm is Vector-only and therefore uses the available AIV count on
  both 910B3 and 950PR.  No fixed AIC:AIV ratio is assumed.

## UB allocation

| Buffer | dtype | elements | bytes |
|---|---:|---:|---:|
| `dCHeadHalf` | FP16 | 4096 | 8192 |
| `cHalf` | FP16 | 4096 | 8192 |
| `dCHead` | FP32 | 4096 | 16384 |
| `dCAcc` | FP32 | 4096 | 16384 |
| `c` | FP32 | 4096 | 16384 |
| `product` | FP32 | 4096 | 16384 |
| `gDa` | FP32 | 64 | 256 |
| **Total** | | | **82,176 bytes** |

The removed `dStatesHalf` and `dStates` buffers previously consumed another
24,576 bytes and forced a full GM read/cast/write for every head.

## Compute sequence

For every `(b,g,k)` task:

1. Load `C[b,k,g]`, cast once to FP32, and zero the FP32 `dC` accumulator.
2. For each head in the group, load and cast `dC_head`.
3. Accumulate `dC_group += dC_head`.
4. Compute `g_da[t] = sum_n(dC_head[t,n] * C[t,n])` in FP32.
5. Store `g_da` for the head.
6. After all heads, store the group-owned `dC_group` with one strided DMA.

## H256 traffic gate

For `[B,H,K,G]=[8,256,64,64]`, `d_states` contains 536,870,912
elements.  Relative to the old finalize-plus-state-passing chain, aliasing the
FP16 result removes:

- 1 GiB FP16 read in finalize;
- 2 GiB FP32 write in finalize;
- 1 GiB of the following state-passing read (FP32 2 GiB becomes FP16 1 GiB).

The optimization must pass the same precision suite and improve end-to-end
backward latency on both Ascend 910B3 and Ascend 950PR.  Otherwise it is
reverted.
