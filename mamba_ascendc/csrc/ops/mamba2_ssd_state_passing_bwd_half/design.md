# Mamba2 SSD state-passing backward with FP16 boundaries

## Scope

This AIV recurrence propagates the state gradient from the last chunk to the
first.  All recurrence, reductions, and exponential scaling remain FP32.  The
large state inputs may be FP16 at stage boundaries and are cast exactly once
after GM-to-UB transfer.

## Interface

```text
(d_chunk_states_half, d_initial_state, d_a_chunk_last) =
    mamba2_ssd_state_passing_bwd_half(
        states_start, d_states_start, d_a_cumsum, d_final_state=None)
```

- `states_start`: FP16 or FP32 `[B,H,K,P,N]`.
- `d_states_start`: FP16 or FP32 `[B,H,K,P,N]`.
- `d_a_cumsum`: FP32 `[B,H,K,T]`.
- `d_final_state`: optional FP32 `[B,H,P,N]`.
- `d_chunk_states_half`: FP16 `[B,H,K,P,N]`.
- remaining outputs: FP32.

`P=64`, `N in {64,128}`, and `T in {64,128}`.  Input dtypes are independent;
the heavy H256 path uses FP16 for both state tensors.

## Tiling and architecture mapping

- Inter-core task: one `(batch, head)` recurrence.
- `blockDim = min(B*H, GetCoreNumAiv())`.
- Chunks remain serial within a task because `gstate[k-1]` depends on
  `gstate[k]`.
- For `N=64` with FP16 `states_start`, two-deep queues prefetch the next chunk.
  `d_states_start` uses a two-deep FP16 or FP32 queue according to its dtype.
- This is AIV-only and queries the platform AIV count; it does not assume the
  910B3 1AIC:2AIV or 950PR 1AIC:1AIV topology.

## UB allocation for P=N=64 pipelined path

| Buffer | bytes |
|---|---:|
| FP32 `gstate`, product, reduction work, input, output | 81,920 |
| scalar workspace | 256 |
| FP16 output/cast buffer | 8,192 |
| two-deep FP16 state prefetch | 16,384 |
| two-deep dState prefetch, FP16 / FP32 | 16,384 / 32,768 |
| **Total, FP16 dState / FP32 dState** | **123,136 / 139,520 bytes** |

The host queries UB capacity and rejects unsupported configurations.  Queues
are initialized only for the active input dtype.

## Compute sequence

For each `(b,h)`, traverse chunks in reverse:

1. Store the current FP32 recurrent gradient as the FP16 chunk-state output.
2. Load/cast `states_start`, multiply by `gstate`, and reduce to the scalar
   `dA` contribution.
3. Read the last decay value and compute the scalar exponential.
4. Scale `gstate` in FP32.
5. Load `d_states_start`; if FP16, cast it into the now-free FP32 product
   buffer, then add it to `gstate`.
6. Prefetch the next chunk through the dtype-specific two-deep queue.

No FP16/BF16 value participates in recurrence math; all non-copy operations
consume FP32 tensors.

## Validation gate

The FP16-gradient path must be numerically equivalent to the old path that
cast the same FP16 BMM output to FP32 in the preceding finalize operator.  It
must pass all M1 backward cases and improve the H256 end-to-end backward on
both Ascend 910B3 and Ascend 950PR.
