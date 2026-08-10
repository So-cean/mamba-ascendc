# Mamba2 SSD backward gate design

## Interface and math

Input tensors are contiguous FP32 tensors in public layout `[B,L,H,P]`, with
`L % 64 == 0` and `P == 64`.

```text
s     = sigmoid(z)
gy    = dout * z * s
dz    = dout * y_pre * s * (1 + z * (1 - s))
dD[h] = sum_{b,l}(x[b,l,h,:] * gy[b,l,h,:])
```

Outputs are internal FP16 `gy_head[B,H,K,64,P]`, FP32 `dz[B,L,H,P]`, and
FP32 `dD[H,P]`.  Gate arithmetic and the `dD` reduction remain FP32; only the
consumer-owned `gy_head` workspace is cast on write.  The Diag Cube path
already consumes FP16, while OffPrepare and DtBwdD explicitly up-cast the
workspace to FP32 in UB before their elementwise math.

训练 heavy path 另提供内部 `mamba2_ssd_bwd_gate_nodd(dout,z,y_pre)` 接口，只
返回 `gy_head` 和 `dz`。它与完整接口使用同一个 device kernel，但通过标量参数
关闭 `x` 的 GM→UB 搬运、`x*gy`、dD partial 写回与第二阶段归约。dD 改由
DtBwdD 在已经持有 `x` 和 `gy_head` 的 UB tile 中随路生成；完整 Gate 接口仍
保留，用于不满足融合条件的 shape 和独立测试。

## Baseline mapping

The baseline assigns one `(B,H,K)` task to one AIV. Each 32-row copy moves a
256-byte head row with source/destination stride `(H-1)*P*4` bytes. At H=128,
Level1 profiling shows Gate MTE2/MTE3 growing about 21x while useful work grows
4x.

## Head-block mapping

Round 1 assigns one `(B,K,ceil(H/8))` task to one AIV:

- `HEAD_BLOCK=8`, with a legal tail block for arbitrary positive H;
- `TILE_ROWS=8`, so the maximum tile is `8*8*64=4096` FP32 elements;
- each input DMA moves 16 KB total, in eight bursts of up to 2 KB;
- pointwise math treats `[T,H_block,P]` as one contiguous Vector tile;
- dD reduces the T dimension for all heads in the block together;
- dz is written in public layout with one strided DMA;
- gy is scattered to the existing head-major consumer layout, one DMA per head;
- x stays intact in UB; a separate scratch tile removes the baseline x reload.

## UB allocation

| Buffer | Elements | Bytes |
|---|---:|---:|
| dout/z/y_pre/x | 4 * 4096 FP32 | 64 KB |
| sigmoid/tmp/scratch | 3 * 4096 FP32 | 48 KB |
| dD accumulator | 8 * 64 FP32 | 2 KB |
| gy output staging | 4096 FP16 | 8 KB |
| **Total** |  | **122 KB** |

This is below the 192 KB UB budget on Ascend 910B3.

`nodd` 路径不分配完整 `x` 和 dD accumulator buffer，仅保留 32B 占位；其余
sigmoid/Newton/dz/gy buffer 与完整路径相同。Host 仍通过 `GetCoreNumAiv()`
取得 Vector 核数，因此 910B3 与 950PR 不硬编码不同核数。

## Correctness and performance gates

- Preserve bitwise determinism between repeated executions.
- Direct Gate precision: NRMSE <= 5e-4 for FP16 gy and <= 2e-6 for FP32 dz/dD.
- Public backward: all existing 40 gradient cases pass.
- H128 Gate MTE2+MTE3 decreases by at least 50%.
- H128 complete backward decreases by at least 15%.

## Round 1 results

The head-block path was built and measured on Ascend 910B3 with the same
`B=8, L=4096, P=N=C=64` workload as the baseline. Pipeline figures below are
the mean of `mamba2_ssd_bwd_gate` rows in profiler `kernel_details.csv`.

| H/G | Gate old | Gate head-block | Gate speedup | MTE2 old/new | MTE3 old/new |
|---:|---:|---:|---:|---:|---:|
| 32/8 | 1.953 ms | 1.836 ms | 1.06x | 0.797/0.655 ms | 0.193/0.396 ms |
| 128/32 | 24.267 ms | 6.985 ms | 3.47x | 16.311/2.428 ms | 4.160/1.516 ms |
| 256/64 | 30.848 ms | 15.561 ms | 1.98x | 15.696/5.834 ms | 6.309/3.543 ms |

At H=128, combined MTE2+MTE3 time decreases by 80.7%; at H=256 it decreases
by 57.4%. The useful tensor bytes still scale with H, but DMA descriptor count,
burst fragmentation, and the extra x reload no longer scale in the baseline
way. H=32 exposes the trade-off: the new head-major gy scatter doubles MTE3,
but lower MTE2/Vector/Scalar time still makes the whole Gate 6.4% faster.

Complete backward median latency (`warmup=10`, `repeat=50`) changes as follows:

| H/G | Baseline | Head-block | Speedup |
|---:|---:|---:|---:|
| 32/8 | 16.508 ms | 16.365 ms | 1.01x |
| 64/16 | 34.392 ms | 32.447 ms | 1.06x |
| 128/32 | 82.908 ms | 65.358 ms | 1.27x |
| 256/64 | 158.739 ms | 142.686 ms | 1.11x |

Dedicated Gate precision and bitwise determinism pass for H=1/7/8/9/32,
including both full and tail head blocks, plus a `B*K=512` reused-UB stress
case. The public backward matrix passed cases 1--15 before an unrelated
`Mamba2SsdChunkScanBwdDiagState` watchdog timeout at case 16; that case has no
z or D and does not dispatch this Gate kernel.

After this change the large-shape backward is no longer dominated by Gate
alone. At H=256 the main profiler means are BMM 27.674 ms, DtBwdD 24.245 ms,
Gate 15.561 ms, and DiagFinalize 15.394 ms. The next round should therefore
optimize Dt layout/pipeline and the remaining Gate MTE/Vector overlap rather
than further changing Gate head tiling in isolation.

## Gate→Dt dD fusion result

The accepted heavy path calls `mamba2_ssd_bwd_gate_nodd`, eliminating Gate's
second full public-layout x read and dD partial production. Dt consumes the x
and gy tiles already resident in UB and reuses the deterministic Gate reduce.

| Platform | Gate before/after | Dt before/after | Gate+Dt saved | H256 BWD |
|---|---:|---:|---:|---:|
| Ascend 910B3 | 14.350/12.217 ms | 14.659/15.627 ms | 1.164 ms | 93.481 ms |
| Ascend 950PR | 9.465/8.050 ms | 10.855/11.765 ms | 0.505 ms | 74.536 ms |

Both platforms pass the 41-case M1 gradient matrix. The complete gates are
92/92 on 910B3 and 95/95 on 950PR. The fusion is enabled by default for T64,
`H>=2`, and two-dimensional D; `MAMBA_ASCENDC_FUSE_DD_IN_DT=0` restores the
diagnostic legacy path.

## Rejected native Sigmoid candidate

The accepted Gate kernel originally expanded sigmoid into `Muls`, two clamps,
`Exp`, `Adds`, `Reciprocal`, and two Newton-refinement rounds. Every operation
spans the complete head-block tile and most pairs require an explicit Vector
barrier. Both target toolchains expose the AscendC high-level `Sigmoid` API;
the implementation uses the native Vector `Div` path and counter masking.

The experiment replaced only that instruction sequence with:

```cpp
Sigmoid<float, false>(sigmoid, z, scratch_bytes, active_elements);
```

The existing scratch tile was reused as API workspace, so public inputs,
FP16 `gy_head`, FP32 `dz`, the Gate-to-Dt dD fusion, task mapping, and UB
capacity remained unchanged. Dedicated Gate tests, including `z` values from
`-1000` to `1000`, passed 8/8 on both devices. Formal H256 Event results were:

| Platform | Accepted BWD | Native Sigmoid BWD | Improvement |
|---|---:|---:|---:|
| Ascend 910B3 | 90.394 ms | 89.959 ms | 0.48% |
| Ascend 950PR | 72.623 ms | 72.065 ms | 0.77% |

Both improvements are below the predeclared 1% end-to-end BWD gate. The
candidate is therefore rejected and the manually refined reciprocal remains
the production implementation. This also shows that local Gate sigmoid
instruction count is not the dominant remaining H256 bottleneck.
