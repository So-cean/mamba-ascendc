# Mamba2SsdStatePassingBwd AscendC 设计文档

## 1. 算子接口

### 1.1 函数签名

```cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_state_passing_bwd(
    const at::Tensor &states_start,
    const at::Tensor &d_states_start,
    const at::Tensor &dA_cumsum,
    const c10::optional<at::Tensor> &dfinal_state);
```

### 1.2 参数说明

| 参数 | I/O | dtype | Shape | 描述与约束 |
|---|---|---|---|---|
| `states_start` | 输入 | FP32 | `[B,H,K,P,N]` | forward 每个 chunk 的输入 state，contiguous |
| `d_states_start` | 输入 | FP32 | `[B,H,K,P,N]` | output branch 对 chunk-start state 的梯度 |
| `dA_cumsum` | 输入 | FP32 | `[B,H,K,T]` | chunk 内 `A*dt` 的 prefix sum；最后一个元素决定 chunk decay |
| `dfinal_state` | 可选输入 | FP32 | `[B,H,P,N]` | final state 的上游梯度；缺省视为零 |
| `d_chunk_states` | 输出 | FP32 | `[B,H,K,P,N]` | chunk contribution `U[k]` 的梯度 |
| `dinitial_state` | 输出 | FP32 | `[B,H,P,N]` | initial state 梯度 |
| `dA_chunk_last` | 输出 | FP32 | `[B,H,K]` | 每个 chunk 最后一个 cumsum 位置的梯度贡献 |

M1 首版固定 `P=64`，支持 `N,T in {64,128}`，`K>=1`。所有输入必须在同一
NPU、FP32、contiguous。该内部接口没有对应的 PyTorch 单算子，语义由 Mamba2
chunk state recurrence 定义。

### 1.3 支持的数据类型

- [x] float32
- [ ] float16
- [ ] bfloat16

## 2. 计算逻辑

### 2.1 数学定义

Forward chunk recurrence：

```text
alpha[k] = exp(dA_cumsum[..., k, T-1])
S[k+1]   = alpha[k] * states_start[k] + U[k]
```

Backward 对每个 `(B,H)` stream 从最后一个 chunk 逆序执行：

```text
g = dfinal_state if present else 0
for k = K-1 ... 0:
    d_chunk_states[k] = g
    dA_chunk_last[k]  = alpha[k] * sum(g * states_start[k])
    g = d_states_start[k] + alpha[k] * g
dinitial_state = g
```

`dA_chunk_last` 随后与 output/state branch 的其他 `dcs` 项合并，不能在本算子
内提前做 reverse-cumsum。

### 2.2 AscendC API 伪代码

```cpp
gstate = dfinal_state ? DataCopyPad(dfinal_state) : Duplicate(0)
for chunk = K - 1 ... 0:
    DataCopyPad(d_chunk_states[chunk], gstate)
    DataCopyPad(state_local, states_start[chunk])
    Mul(product, gstate, state_local, P*N)
    ReduceSum(dot, product, shared_tmp, P*N)
    alpha = ScalarExp(dA_cumsum[chunk, T-1])
    dA_chunk_last[chunk] = dot[0] * alpha
    DataCopyPad(dstart_local, d_states_start[chunk])
    Muls(gstate, gstate, alpha, P*N)
    Add(gstate, gstate, dstart_local, P*N)
DataCopyPad(dinitial_state, gstate)
```

实现路径选择：**AscendC Vector kernel**。本算子只有逐元素状态更新和一个连续
FP32 dot reduction，不包含矩阵乘法；Cube 用于相邻的 ChunkScanBwd 和
ChunkStateBwd，而不是本递推。

## 3. Tiling 策略

### 3.1 Block 级

```text
task_count   = B * H
used_core    = min(task_count, GetCoreNumAiv())
task         = block_idx; task < task_count; task += used_core
```

一个 task 独占完整 `(B,H)` stream，保证跨 chunk 逆序依赖，同时没有跨核写冲突。
每个 state 行为 `P*N*4` 字节；当前规格为 16 KiB 或 32 KiB，天然满足 512B 对齐。

### 3.2 UB 级

递推要求完整 `gstate` 常驻 UB；每次只加载一个 chunk 的 `states_start` 或
`d_states_start`。`N=128` 的最坏 FP32 预算：

| Buffer | 元素 | 数量 | 字节 |
|---|---:|---:|---:|
| `gstate` | `P*N=8192` | 1 | 32,768 |
| input queue | `P*N` | 1 | 32,768 |
| output queue | `P*N` | 1 | 32,768 |
| reduction product | `P*N` | 1 | 32,768 |
| reduction shared tmp | `P*N` | 1 | 32,768 |
| scalar/result | aligned | 1 | 256 |
| **总计** | | | **164,096 B** |

Host 使用平台 API 获取 UB 大小并验证 `5*P*N*sizeof(float)+4096 <= ubSize`；
Ascend 910B 192 KiB UB 下有不少于 28 KiB 余量。因为 chunk recurrence 串行且
完整 state 占比较大，首版不对 state queue 做 double buffer；后续 profiler 证明
MTE 是瓶颈后再用 event 连接单 input/output queue。

### 3.3 Tiling 参数

该工程使用 no-workspace kernel，shape 参数直接作为 kernel 标量参数：

```text
B, H, K, P, N, T, has_dfinal, used_core_num
```

`state_elements=P*N`，`chunk_stride=T`。没有尾 tile；不满足固定 P/N/T 约束的
输入由 Host 明确拒绝。

## 4. Workspace

不申请 GM workspace。输出 tensor 由 Host 分配；所有临时量只存在于每个 AIV 的
UB 中。

## 5. 性能优化

1. `gstate` 在整个 reverse chunk loop 中常驻 UB，不在 chunk 间回写/重读。
2. `(B,H)` 独立 stream 并行，避免 M0 的 `(B,G)` ownership 导致组内 head 串行。
3. state、dstart 和 dU 都是连续 16/32 KiB 大块传输。
4. dot 使用 Vector `Mul + ReduceSum`，禁止 `GetValue` 遍历 `P*N`。
5. `dA_chunk_last` 每个 task 独占，无 atomic；结果可确定性复现。

该算子预计以 Vector reduction 和 GM copy 为主；最终性能必须由
`torch_npu.profiler` 判断，不能根据指令数量推断。

## 6. Kernel 实现要点

- `DataCopyPad` 用于所有 GM↔UB 搬运。
- `ReduceSum` 可能修改 source，因此只对专用 `product` buffer 归约。
- `ScalarExp` 与 forward state-passing 使用同一近似，保证 recurrence 一致。
- `d_chunk_states[k]` 必须在更新 `gstate` 之前写出。
- `dA_chunk_last` 的 dot 使用更新前的 incoming `gstate`。
- FP16/BF16 不在首版支持域；未来支持时必须先 Cast 到 FP32 再递推和归约。

## 7. 实现检查清单

- [x] 接口、数学定义和依赖顺序确定
- [x] `(B,H)` Block tiling 和 UB 预算确定
- [x] 无 GM workspace、无 atomic
- [x] `op_host/mamba2_ssd_state_passing_bwd.cpp`
- [x] `op_kernel/mamba2_ssd_state_passing_bwd.cpp`
- [x] `ops.h` / `register.cpp` / `CMakeLists.txt`
- [x] public functional test 和 30-case precision report
- [x] mssanitizer 与 8-case profiler

实测结果：30/30 FP32 精度 case 通过，worst MERE/MARE 为
`1.515e-7 / 7.300e-7`；kernel-filter memcheck、racecheck、initcheck 通过；
8-case profiler 为 `5.480–19.834 us`，相对同 NPU eager tensor composition
平均加速 `11.139x`。

## 8. 参考实现

- `mamba_ascendc/csrc/ops/mamba2_ssd_state_passing/`
- `mamba_triton_ascend/mamba2/ops/triton/ssd_combined.py`
- `mamba_ascendc/csrc/ops/mamba2_ssd_bwd/design.md`
