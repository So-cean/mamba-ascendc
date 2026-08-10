# Mamba2SsdBwd AscendC 设计文档

## 1. 接口与实现边界

### 1.1 Public autograd 接口

Public Python 签名继续使用 `mamba2_ssd_fwd(...)`。输入需要梯度时由
`torch.autograd.Function` 保存实际 forward dispatch、micro-chunk 和必要中间量，
`backward(dout, dfinal_state)` 返回：

```text
dx, ddt, dA, dB, dC, None(chunk_size),
dD, dz, ddt_bias, dinitial_states, None(return_final_state), ...
```

不存在的 optional 输入必须返回 `None`，不能用空 Tensor 冒充 public 梯度。

### 1.2 Native AscendC 内部接口

首个 correctness Gate 使用一个原生 `mamba2_ssd_bwd` kernel：

```cpp
std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_bwd(
    const at::Tensor &x,                 // [B,L,H,P]
    const at::Tensor &dt,                // [B,L,H]
    const at::Tensor &A,                 // [H]
    const at::Tensor &B,                 // [B,L,G,N]
    const at::Tensor &C,                 // [B,L,G,N]
    const at::Tensor &dout,              // [B,L,H,P]
    const at::Tensor &final_state,        // [B,H,P,N]
    const c10::optional<at::Tensor> &D,  // [H,P]
    const c10::optional<at::Tensor> &z,  // [B,L,H,P]
    const c10::optional<at::Tensor> &dt_bias,
    const c10::optional<at::Tensor> &initial_states,
    const c10::optional<at::Tensor> &dfinal_state,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);
```

Native 输出为：

```text
dx [B,L,H,P]
ddt [B,L,H]
dA_partial [B,H]
dB [B,L,G,N]
dC [B,L,G,N]
dD_partial [B,H,P]
dz [B,L,H,P] or empty sentinel
dinitial [B,H,P,N] or empty sentinel
```

Python 层做确定性 `sum(dim=0)` 得到 `dA/dD`，并从 `ddt` 归约
`ddt_bias`。这些归约后续由 `Mamba2SsdDtBwd` 合并到原生性能路径。

### 1.3 首版 native 规格

| 项目 | M0 correctness | M1 performance |
|---|---|---|
| dtype | FP32 | FP32 public；Cube FP16 operand/FP32 accumulate |
| layout | contiguous fixed length | contiguous fixed length |
| shape | `P=N=64`, `L % 64 == 0`, `chunk=64` | `P=64`, `N/chunk in {64,128}` |
| optional | basic，`dfinal_state` 可选 | D/z/bias/softplus/clamp/initial 全覆盖 |
| algorithm | direct reverse recurrence | chunk algebra + Cube/MIX |
| fallback | unsupported shape 明确报错 | generic shape 可显式 fallback |

M0 只作为 native 梯度闭环，不发布性能结论。README 性能只在 M1 完成后更新。

## 2. 数学定义

记：

```text
u_t       = dt_t + dt_bias
q_t       = clamp(softplus(u_t), q_min, q_max)
a_t       = exp(A * q_t)
v_t       = x_t * q_t
S_t       = a_t * S_(t-1) + v_t[:,None] * B_t[None,:]
Ypre_t    = S_t @ C_t + D * x_t
out_t     = Ypre_t * silu(z_t)
```

没有 optional 参数时跳过对应项。给定 `gout=dout`，逆序梯度为：

```text
gy_t      = gout_t * silu(z_t)                         # or gout_t
dz_t      = gout_t * Ypre_t * silu'(z_t)
dD       += gy_t * x_t
dx_D      = gy_t * D

gS_t     += gy_t[:,None] * C_t[None,:]
dC_t     += sum_p gy_t[p] * S_t[p,n]

da_t      = sum_(p,n) gS_t[p,n] * S_(t-1)[p,n]
dB_t     += sum_p gS_t[p,n] * v_t[p]
dv_t[p]   = sum_n gS_t[p,n] * B_t[n]
gS_(t-1)  = a_t * gS_t

dx_t      = q_t * dv_t + dx_D
dq_t      = sum_p dv_t[p] * x_t[p] + da_t * a_t * A
dA       += da_t * a_t * q_t
ddt_t     = dq_t * clamp_grad * softplus_grad
```

同一 group 的 B/C 被多个 head 共享，`dB/dC` 必须按 head 确定性相加。

## 3. M0 AscendC API 调用序列

M0 每个 `(batch, group)` 为一个任务，组内 head 顺序执行，避免 dB/dC atomic。
每个 active AI Vector core 独占一段可复用的 `[L,P,N]` workspace：

```cpp
for (head in group):
  Duplicate(state, 0, P*N)
  for (t = 0; t < L; ++t):
    state = exp(A*q[t]) * state + (x[t]*q[t])[:,None] * B[t][None,:]
    DataCopy(states_work[t], state, P*N)

  Duplicate(gstate, 0, P*N);                // or DataCopy(dfinal_state)
  for (t = L - 1; t >= 0; --t):
    DataCopy(xLocal, x[t], P);
    DataCopy(doutLocal, dout[t], P);
    DataCopy(bLocal, B[t], N);
    DataCopy(cLocal, C[t], N);

    q = preprocess_dt_scalar(dt[t], bias, softplus, clamp);
    a = Exp(A[head] * q);

    DataCopy(state, states_work[t], P*N)
    DataCopy(statePrev, states_work[t-1], P*N)  // t=0 时清零

    for p:
      yState[p] = ReduceSum(state[p,:] * cLocal[:]);
      gy[p] = doutLocal[p];
    for p,n:
      gstate[p,n] += gy[p] * cLocal[n];
      dC[n] += gy[p] * state[p,n];
      dB[n] += gstate[p,n] * x[p] * q;
      dv[p] += gstate[p,n] * bLocal[n];
      da += gstate[p,n] * statePrev[p,n];
    Muls(gstate, gstate, a, P*N);

    dx[p] = q * dv[p];
    ddt = ReduceSum(dv * x) + da * a * A[head];
    dA_partial += da * a * q;
    DataCopy(dx/dt/dB/dC, local outputs);
```

早期 M0 曾尝试跨完整序列用逆变换恢复 state；严格精度报告显示 `dA/dC` 会因
final-state 舍入误差放大而失败。当前 M0 改为 FP32 重算并暂存 token state。
M1 不保留这个大 workspace，而改用 chunk 分解：

```text
ChunkScanBwd
  Cube: dW=gY@X^T, dX=W^T@gY, dC=dCB@B, dB=dCB^T@C
  Vector: decay/mask/dcs/local reductions
        ↓ dS_start
StatePassingBwd
  reverse chunk recurrence; state gradient stays in UB
        ↓ dU
ChunkStateBwd
  Cube: dB_state=R@dU^T, dR=B@dU
        ↓ dcs_state, dX_state
DtBwd
  reverse-cumsum; dt/A/bias/softplus/clamp; deterministic reductions
```

## 4. Tiling

### 4.1 Block 级

M0：

```text
task_count  = B * G
block_dim   = min(task_count, GetCoreNumAiv())
task        = block_idx; task < task_count; task += block_dim
```

一个 task 独占 `[B,L,G,N]` 的一组 dB/dC 输出，组内 head 串行累加，因此无
跨核写冲突。M1 的 Cube 分支按 `(B,H,chunk,tile)` 并行；共享 group 梯度先写
`[B,H,chunk,...]` partial，再由固定树归约，禁止不确定性 atomic。

### 4.2 UB 级与对齐

M0 production gate `P=N=64` 的 FP32 UB 预算：

| Buffer | 元素 | 数量 | 字节 |
|---|---:|---:|---:|
| state | `P*N=4096` | 1 | 16,384 |
| gstate | `P*N=4096` | 1 | 16,384 |
| state scratch | `P*N=4096` | 1 | 16,384 |
| x/dout/gy/dv/dx | `P=64` | 5 | 1,280 |
| B/C/dB/dC | `N=64` | 4 | 1,024 |
| scalar/reduction scratch | aligned | 1 | 4,096 |
| queues and padding | aligned | - | <= 16,384 |
| **总计** | | | **<= 71,936 B** |

小于 Ascend 910B 的 192 KiB UB。所有 queue 长度按 32B 对齐；GM 连续大块
按 512B 对齐。M1 `N=128` 时 state/gstate 为各 32 KiB，仍只让一个 head 的
state 驻留 UB；Cube tile 使用现有 64×64 dynamic matmul。

## 5. Workspace 与保存/重算

M0 host 临时申请 `[used_core_num,L,P,N]` FP32 workspace，各 core 在连续处理
不同 `(batch,group,head)` 时复用自身分片；该方案仅用于 correctness。Autograd
保存输入引用和 `final_state`。M1 默认：

| 中间量 | extreme 大小 | 策略 |
|---|---:|---|
| `dA_cumsum` | 4 MiB | 保存 |
| processed q | 4 MiB | 重算 |
| xCube | 128 MiB | 重算 |
| B/C Cube layout | 各 64 MiB | 重算 |
| CB | 128 MiB | 重算 |
| chunk state / states_start | 各 256 MiB | 重算 |
| Ypre | 256 MiB | z 存在时首版保存，后续可重算 |

不得同时保存 chunk state 和 states_start；extreme 会额外占用 512 MiB。

## 6. 精度与错误处理

- Public 输入和所有 Vector 计算统一 FP32；M1 Cube 为 FP16 operand、FP32 累加。
- M0 只接受 basic/non-softplus、`P=N=chunk=64`；其他输入必须报错，
  不静默走未验证路径。
- M1 的误差门：每项梯度 finite、NRMSE `<=5e-3`、cosine `>=0.999`；FP64
  directional derivative 相对误差 `<=1e-2`。
- `dB/dC/dA/dD/dt_bias` 使用确定性归约；相同 seed 重复 5 次必须 bitwise
  一致或在报告中明确非确定来源。

## 7. 性能计划

M0 仅统计功能 latency。M1 目标：

1. 复用 `mamba2_dynamic_matmul.h` 的 64×64 Cube tile；
2. Vector 生产衰减/mask，Cube 消费 GEMM，MIX pipeline 以 event 连接；
3. state passing 的正向 state 与反向 gstate 常驻 UB；
4. 不物化 full token state，不使用跨完整序列 reverse reconstruction；
5. profiler 同时报 kernel chain、Cube active、MAC ratio、AIV scalar/MTE2/MTE3；
6. 与 A100 80GB 同 shape 报 bwd-only、fwd+bwd、peak memory 和 saved tensor。

## 8. 工程检查清单

- [x] 标准目录 `csrc/ops/mamba2_ssd_bwd/`
- [x] Host 参数、shape、dtype 和 UB 检查
- [x] Kernel M0 basic recurrence
- [x] `ops.h` 声明
- [x] `register.cpp` schema / PrivateUse1 注册
- [x] `csrc/CMakeLists.txt` host/device 源文件
- [x] Python `torch.autograd.Function`
- [x] dtype-preserving FP64 oracle
- [x] native / fallback case 分开报告
- [x] 32 例 public autograd 精度
- [x] M1 子算子 directional derivative、mssanitizer、device-event benchmark、profiler

## 9. M1 native-core 集成状态

当前 public autograd 对 `P=N=T=64` 的 Cube/MIX 规格已形成完整 M1 数学闭环：

```text
gate/D tensor ops
  -> native ChunkScanBwdOff MIX
  -> native ChunkScanBwdDiagState MIX (diagonal + chunk-state)
  -> native StatePassingBwd
  -> native DtBwd
  -> deterministic shared-group reduction
```

40/40 端到端 gradient cases 通过，覆盖 output-only、output+final、final-only、
`D=[H,P]/[H]`、z、dt_bias、softplus、finite clamp 和 initial state；worst NRMSE
为 `7.116e-4`，lowest cosine 为 `0.999997914`。DiagState 子算子另有 40/40
精度、bitwise determinism 与 kernel-filtered memcheck/racecheck/initcheck 门禁。

M1 的 SSD 核心分支均已由 AscendC 实现；public autograd 仍保留 y/z 重算、分支
相加和 Off dC shared-group reduction 等少量 NPU tensor glue，因此不把整个
backward 描述成单一 fused kernel。DtBwd 的 `T*P` 标量内循环已替换为 Vector
Mul + WholeReduceSum；DiagState 已使用 group-owned 调度复用 `C@B`/`B^T`，并
直接输出 group dB/dC。40/40 端到端精度继续通过。

A100 80GB 与 910B3 的同 shape benchmark：large/extreme 从原始
`10.556/87.101 ms` 降至 `5.530/45.384 ms`。训练前向已把 UB 中的 pre-gate
写出并由 backward 复用，40/40 精度继续通过；extreme saved tensor 为
`904.008 MiB`，A100 同 benchmark 为 `908.008 MiB`。当前 extreme 吞吐仍只有
A100 的 18.3%。DiagState 使用四 head `64x256` 宽 GEMM，并在 L0C 中累计 group
dB/dC；校正 profiler 调用数后约 `18.07 ms`，remaining
public tensor glue 约 `9.25 ms`、DtBwd `5.86 ms`、Off `4.85 ms`、forward
recompute 约 `5.50 ms`。下一轮必须做跨 task 双槽流水，以及
z/D/g_cs/Off group epilogue 融合。

## 10. 参考

- `mamba_torch/ssd_reference.py`
- `mamba_triton_ascend/mamba2/ops/triton/ssd_combined.py`
- `mamba_ascendc_ops/mamba2_ssd_chunk_mix/op_kernel/mamba2_dynamic_matmul.h`
- `mamba_ascendc/csrc/ops/mamba2_ssd_state_passing/`
