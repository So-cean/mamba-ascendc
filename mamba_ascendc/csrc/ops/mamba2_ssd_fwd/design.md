# Mamba-2 SSD Forward 设计

## 1. 范围

公开接口为 `ascend_kernel.mamba2_ssd_fwd`，实现固定长度 Mamba-2 SSD forward。
当前支持 FP32 API、variable B/C、多 head/group、D、z、dt bias、softplus、dt
clamp、initial state 和 final state；不支持 varlen 和 backward。

数值定义：

```text
u_t      = dt_t + dt_bias
dt'_t    = clamp(softplus(u_t), min, max)
dA_t     = A_h * dt'_t
xdt_t    = x_t * dt'_t
S_t      = exp(dA_t) * S_(t-1) + B_t outer xdt_t
y_t      = C_t @ S_t + D * x_t
out_t    = y_t * silu(z_t)
```

## 2. 实现分层

公开 Python wrapper 根据 shape 选择：

| 路径 | 条件 | 实现 |
|---|---|---|
| Cube/MIX | `P=64`、`N=64/128`、chunk 64/128、`L%64=0` | preprocess + chunk_mix + state epilogue |
| aligned | 维度 16 对齐、`P*N<=8192` | preprocess/prepare + torch matmul + state passing |
| generic | 其余合法 FP32 shape | direct recurrence kernel |

Cube/MIX 算子来自独立工程
`mamba_ascendc_ops/mamba2_ssd_chunk_mix`，本工程通过 aclnn bridge 注册到
`torch.ops.mamba_ascend`。

## 3. Cube/MIX 算法

对每个 chunk：

```text
CB         = C[T,N] @ B^T[N,T]
W[i,j]     = CB[i,j] * exp(dA_cs[i] - dA_cs[j]), j <= i
Y_diag     = W[T,T] @ xdt[T,P]
chunkState = B^T[N,T] @ (xdt * exp(dA_last - dA_cs))[T,P]
Y_off      = C[T,N] @ stateStart[N,P] * exp(dA_cs)
stateEnd   = exp(dA_last) * stateStart + chunkState
```

Cube 使用 FP16 operands 和 FP32 accumulate。Vector 负责 softplus/exp、cumsum、
causal mask、transpose、state recurrence、D/z epilogue。

## 4. 并行和 tiling

- Preprocess：1D grid，在 `(B,H,chunk)` 和 `(B,G,chunk)` 任务间分发。
- ChunkMix：`KERNEL_TYPE_MIX_AIC_1_2`，每个 MIX block 为 1 AIC + 2 AIV。
- 当 `(B,G,chunk)` 已能占满 AIC 时使用 group-task mode，在同一核复用 B/CB。
- StateEpilogue：一个 MIX task 负责完整 `(B,H)` stream，state 跨 chunk 常驻 UB。
- TilingKey：chunk64/N64=`2`，chunk64/N128=`3`，chunk128/N128=`7`；
  StateEpilogue chunk64=`4`、chunk128=`8`。
- core 数和 workspace 均由 platform API 查询，不硬编码设备核数。

## 5. UB 预算

### ChunkMix 每个 AIV

| Buffer | T64 大小 | T128 大小 | 生命周期 |
|---|---:|---:|---|
| transpose queues | 1 KiB | 4 KiB | B/x layout |
| dA row | 256 B | 512 B | 每 head |
| CB input | 8 KiB | 32 KiB | build W |
| FP16 output | 4 KiB | 16 KiB | W store |
| two FP32 work blocks | 16 KiB | 64 KiB | decay/CB 临时 |
| causal mask | 8 KiB | 32 KiB | 核生命周期 |
| FP16 causal mask | — | 16 KiB | T128 核生命周期 |
| decay buffers | 640 B | 2.25 KiB | 每 head |
| broadcast temporary | 4 KiB | 16 KiB | broadcast API |
| 合计 | 约 41.9 KiB | 约 182.8 KiB | T128 接近 192 KiB 上限 |

T128 不增加双缓冲，因为会越过 UB 上限。causal mask 和 broadcast 临时空间只在
需要释放 UB 以容纳新阶段时才考虑压缩；单独压缩不会减少 Cube MAC。

### Preprocess

Preprocess 使用共享 single-depth input/output queues、16x16 transpose queues、
`dA/dt/cumsum` buffers、完整 `T*P` dt broadcast matrix。T128/N128 时可选 32 KiB
`B^T` 聚合 buffer，将大量 512 B strided stores 合并为一次 32 KiB GM store。
该规格接近 UB 上限，因此不使用通用双缓冲。

### StateEpilogue

每个 AIV 保存自身负责的半个 FP32 state，并通过两个 GM workspace slots 流水化
state cast、Cube projection 和 epilogue。D 当前广播为 token matrix；这是可释放
UB 的维护项，但不是已证明的性能优化。

## 6. Workspace 和同步

AIC 与 AIV 的本地存储不共享，使用每个 MIX core 独立 workspace：

```text
2 * weightedX FP16 slots
2 * W FP16 slots
1 * CB FP32 slot
legacy B slot（仅 chunk64/N128 legacy layout）
```

同步使用固定 event ID 和细粒度 hard events。禁止用 `PIPE_ALL` 替代跨 MTE 阶段
依赖。workspace 大小由 host tiling 根据 T/P/N 计算。

## 7. 精度和验证

- Public API：31-case aligned/general gate。
- 强制 Cube/MIX：35-case gate。
- 指标：finite、allclose/max-abs、NRMSE、cosine。
- Memory：chunk_mix/off_epilogue/state_epilogue 分别执行 mssanitizer。
- 性能：torch_npu profiler；事件计时只用于延迟分布，不替代 profiler 结论。

正式入口见 `scripts/jobs/fwd/` 和 `docs/fwd/README.md`。
