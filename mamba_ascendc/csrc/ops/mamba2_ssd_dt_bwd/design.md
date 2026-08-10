# Mamba2SsdDtBwd AscendC 设计文档

## 1. 算子接口

```cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mamba2_ssd_dt_bwd(
    const at::Tensor &x,                 // [B,L,H,P]
    const at::Tensor &d_xdt_total,       // [B,H,K,T,P]
    const at::Tensor &g_dA_cs_total,     // [B,H,K,T]
    const at::Tensor &dt,                // [B,L,H]
    const at::Tensor &A,                 // [H]
    const c10::optional<at::Tensor> &dt_bias,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

mamba2_ssd_dt_bwd_d(
    const at::Tensor &x,                 // [B,L,H,P]
    const at::Tensor &d_xdt_total,       // [B,H,K,T,P]
    const at::Tensor &g_dA_cs_total,     // [B,H,K,T]
    const at::Tensor &dt,                // [B,L,H]
    const at::Tensor &A,                 // [H]
    const at::Tensor &gy_head,           // FP16 [B,H,K,T,P]
    const at::Tensor &D,                 // [H,P]
    const c10::optional<at::Tensor> &dt_bias,
    bool dt_softplus,
    double dt_limit_min,
    double dt_limit_max);

mamba2_ssd_dt_bwd_d_dd(
    // 输入与 mamba2_ssd_dt_bwd_d 相同，额外返回 dD[H,P]
    ...);
```

输出依次为最终 `dx_core[B,L,H,P]`、`ddt[B,L,H]`、`dA[H]`、
`ddt_bias[H]`。无 bias 时第四项由 public autograd 忽略并映射为 `None`。
`mamba2_ssd_dt_bwd_d_dd` 额外返回 `dD[H,P]`，只在 T64、`H>=2`、二维 D 的
head-block 路径使用。

数值输入与 public 输出保持 contiguous FP32；内部 `gy_head` workspace 为
contiguous FP16，`P=64`、`T in {64,128}`、`L=K*T`。DtBwdD 在 UB 中先把
`gy_head` 升为 FP32，再执行 `gy*D`，不允许 FP16 直接参与梯度算术。

### Gate→Dt dD 融合

T64 head-block 已把 public-layout `x[T,2,P]` 搬入 `xHeadBlockBuf`，并在 D
分支读取 FP16 `gy_head[T,P]`。融合路径在这两个 tile 均驻留 UB 时计算：

```text
dD_partial[h,b,k,p] = sum_t float(gy_head[h,b,k,t,p]) * x[b,k,t,h,p]
dx_D                = float(gy_head) * D[h,p]
```

实现先把当前 head 的原始 x 从 head-block staging 解包到工作 buffer，与已经
升为 FP32 的 gy 相乘，再用固定 64-row tree reduction 得到一行 dD partial；
随后复用同一 buffer broadcast D 并完成 `dx_D`。这只增加 UB↔UB 解包和 Vector
乘加，不增加 x 的 HBM 读取。partial 保持 `[H,B,K,P]`，复用 Gate 已验证的
固定顺序第二阶段归约，最终 dD 仍为 FP32。

## 2. 计算逻辑

```text
u       = dt + dt_bias
q_pre   = softplus(u) if enabled else u
q       = clamp(q_pre, min, max)
xdt     = x * q
cs[t]   = sum_(j<=t) A*q[j]

gw[t]        = sum_(s>=t) g_dA_cs_total[s]
dq_direct[t] = sum_p d_xdt_total[t,p] * x[t,p]
dq[t]        = dq_direct[t] + A * gw[t]
dx_core      = q * d_xdt_total
dA           = sum_(b,k,t) gw * q
ddt          = dq * clamp_grad(q_pre) * softplus_grad(u)
ddt_bias     = sum_(b,k,t) ddt
```

Clamp 的等号边界梯度为 1；不允许从 `dA_cumsum/A` 反推 q，因此 `A=0` 仍有
定义良好的梯度。

## 3. Tiling 与确定性归约

### Phase 0：普通路径

- task：`B*H*K`，每个 task 独占一个 chunk；
- 读取 public-layout x/dt，并读取连续 `d_xdt/g_cs`；
- chunk 内 reverse inclusive cumsum；
- 直接写 public-layout `dx/ddt`；
- 写 `[H,B,K]` 的 `dA/dbias` partial。

### Phase 0：T64 `D` 路径的 head-block tiling

H256 profiler 基线中 `mamba2_ssd_dt_bwd_d` 为 **24.245 ms**。原实现把
`(b,h,k)` 作为 task，导致 public `x/dx` 的 `[B,L,H,P]` 布局每次只搬一个
head：一个 descriptor 有 64 个 256B block，block 间跨过其余 `H-1` 个
head；同时每个 chunk 都重新读取同一行 `A/bias/D`。

T64 `D` 路径采用固定 `headBlock=2`：

- task 改为 `(b, head_block)`，共 `B*ceil(H/2)` 个 task；task 内按顺序遍历 K；
- 每个 chunk 用一个二维 DMA 搬连续两个 head，`x/dx` 每个 descriptor 的有效
  payload 由 16 KiB 增至 32 KiB，blockLen 由 256B 增至 512B；
- UB 内用二维 DataCopy 将 token-major 的 `[T,2,P]` 解包成单 head
  `[T,P]`，沿用已验证的数学计算，再打包回 `[T,2,P]`；
- `dt` 以 `[T,2]` 一次搬入，每个 token 补齐至 32B，Gather 通过
  `localHead` 选择对应列；
- `A/bias/D` 在 task 开始时一次搬入并跨全部 K chunk 复用；H256、K64 时，
  每个 `(b,h)` 的 D 行读取次数从 64 降为 1；
- 奇数 H 的最后一个 task 令 `validHeads=1`。x/dx 的 blockLen、GM stride
  以及 UB 解包/打包 stride 均由 `validHeads` 计算，不读取或写入虚拟 head；
- T128 与无 D 路径保持原来的 `(b,h,k)` tiling，避免扩大 UB 工作集并控制
  本轮改动风险。

### Phase 1

- task：H；
- 每个 head 按固定 `(b,k)` 顺序归约 partial；
- 不使用 atomic，同一输入重复执行 bitwise 一致。

当前 correctness kernel 已完成最终接口、chunk 并行与原生固定顺序归约。chunk
内部仍使用 FP32 标量 `GetValue/SetValue`；后续性能轮次将替换为 Vector reverse
scan、`WholeReduceSum`、Compare/Select 和 broadcast，不改变接口或归约顺序。

## 4. UB 规划

T=128、P=64 最坏路径：

| Buffer | 字节 |
|---|---:|
| x | 32 KiB |
| d_xdt | 32 KiB |
| dx output queue | 32 KiB |
| padded dt | 4 KiB |
| g_cs + scalar | < 1 KiB |
| padded ddt queue | 4 KiB |
| **合计** | **约 105 KiB** |

低于 910B 的 192 KiB UB。phase 1 只使用 32B scalar queue。

T64、P64、headBlock=2 的 `_d` 路径静态预算：

| Buffer group | 字节 |
|---|---:|
| 原单-head 计算工作集（x、dXdt、dx queue、dt/ddt、scan、broadcast 等） | 64,800 B |
| x head-block staging `[T,2,P]` | 32 KiB |
| dx head-block staging `[T,2,P]` | 32 KiB |
| dt head-block staging `[T,8]` | 2 KiB |
| A/bias staging 与 D `[2,P]` | 576 B |
| FP16 gy staging `[T,P]` | 8 KiB |
| **合计** | **141,152 B（137.844 KiB）** |

预算低于 192 KiB，保留 63,648 B（62.156 KiB）余量。head-block staging 仅在 T64 `_d`
路径分配；T128/普通路径只为这些 buffer 分配 32B 占位，不改变其主要 UB 预算。

融合 dD 仅新增一个 256B VECOUT queue；乘积复用 `xBuf`，gy 复用
`dXdtBuf`，partial reduction 不新增矩阵级 buffer。总 UB 仍低于 192 KiB。
Host 继续通过 `GetCoreNumAiv()` 选择纯 Vector blockDim；该路径不改变相邻 MIX
kernel 在 910B3 的 1AIC:2AIV 与 950PR 的 1AIC:1AIV 编译拓扑。

## 5. 预期收益与验收

- 直接收益：H256 上 x/dx 公共布局 descriptor 数量约减半，D 的 GM 读取
  descriptor 数量按 K 倍下降，phase-0 task 数从 `B*H*K` 降为
  `B*ceil(H/2)`；
- 保守目标：H256 `_d` kernel 从 24.245 ms 降至 18 ms 以下（至少 1.35x）；
- 精度门槛：现有 FP32 40-case、D/gate 大 shape 与 odd-H 定向用例全通过；
- 性能门槛：H32/H128 不回退超过 5%，H256 至少改善 20%；
- profiler 验收：确认 x/dx 的 MTE2/MTE3 blockLen=512B（尾块 256B），并检查
  Vector/MTE overlap、任务长尾以及 GM 带宽变化。
- Gate→Dt 融合验收：Gate 不再出现 x 的 MTE2 和 dD partial MTE3；Dt 的 x
  MTE2 不增加，双平台 public M1 精度门禁全部通过，H256 BWD 端到端必须有
  超出噪声的改善，否则完整回退。

## 6. 已完成验证

- [x] 编译和 4-case public UT；
- [x] 40/40 FP32 精度用例；
- [x] clamp 边界、ULP、softplus `u=±30`、`A=0`、chunk scan 定向用例；
- [x] 5 次 `dA/dbias` bitwise determinism；
- [x] FP64 directional derivative；
- [ ] mssanitizer；
- [x] T64 Vector reverse scan、WholeReduceSum、Compare/Select 和 broadcast；
- [x] head-block host/kernel 静态布局与 UB 预算审查；
- [x] head-block 与 Gate→Dt dD 融合在 910B3/CANN 8.2、950PR/CANN 9.0 构建；
- [x] 双平台 M1 41-case，完整 910B3 92-case、950PR 95-case 回归；
- [x] 双平台 H256 device Event 与 Level1 profiler；
- [ ] 本轮融合路径的独立 mssanitizer 复核。

Gate→Dt 融合的 H256 正式 BWD 为 910B3 `93.481 ms`、950PR
`74.536 ms`，相对融合前正式值改善 `1.47%/0.68%`。Level1 中 Gate 分别减少
`2.133/1.415 ms`，Dt 因 dD 解包与归约增加 `0.968/0.910 ms`，Gate+Dt 净省
`1.164/0.505 ms`。该路径默认开启，环境变量
`MAMBA_ASCENDC_FUSE_DD_IN_DT=0` 仅用于诊断回退。

## 7. 参考

- `mamba2_ssd_preprocess/op_kernel/mamba2_ssd_preprocess.cpp`
- `mamba_triton_ascend/mamba2/ops/triton/ssd_combined.py::_chunk_cumsum_bwd`
