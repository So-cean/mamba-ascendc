# torch.ops.mamba_ascend.mamba2_ssd_bwd

```text
torch.ops.mamba_ascend.mamba2_ssd_bwd(
    x, dt, A, B, C, dout, final_state,
    D=None, z=None, dt_bias=None, initial_states=None,
    dfinal_state=None, dt_softplus=False,
    dt_limit_min=0.0, dt_limit_max=3.402823466e+38,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
```

Mamba-2 SSD fixed-length forward 的内部 M0 AscendC 反向算子。它以 FP32 顺序
重算 token state，再逆序传播状态梯度，计算 `dx`、`ddt`、`dA_partial`、`dB`
和 `dC`，用于 basic 规格的 correctness fallback。Public 用户通常不直接调用
此接口，而是对 `ascend_kernel.mamba2_ssd_fwd` 的输出调用 `backward()` 或
`torch.autograd.grad()`。

当前高性能训练路径是 M1 native-core public autograd：ChunkScan off-diagonal、
diagonal/chunk-state、StatePassing 和 dt/A 链均由独立 AscendC 子算子执行；Python
层只保留 y/z 重算、分支相加和 shared-group reduction 等少量 tensor glue。M1
支持完整 optional 参数，但不经过本页这个窄规格 M0 内部接口。

状态更新和反向依赖为：

$$
S_t=e^{A\Delta_t}S_{t-1}+(x_t\Delta_t)B_t^T,
\qquad y_t=S_tC_t.
$$

## 参数说明

- **x** (*Tensor*) – 输入，形状 `(B, L, H, P)`。
- **dt** (*Tensor*) – 离散化步长，形状 `(B, L, H)`。
- **A** (*Tensor*) – 每个 head 的衰减参数，形状 `(H,)`。
- **B** (*Tensor*) – group-shared 输入状态参数，形状 `(B, L, G, N)`。
- **C** (*Tensor*) – group-shared 输出状态参数，形状 `(B, L, G, N)`。
- **dout** (*Tensor*) – output 的上游梯度，形状 `(B, L, H, P)`。
- **final_state** (*Tensor*) – forward 最终状态，形状 `(B, H, P, N)`。
- **D** (*Tensor, optional*) – skip 参数。M0 必须为 `None`。
- **z** (*Tensor, optional*) – gate 输入。M0 必须为 `None`。
- **dt_bias** (*Tensor, optional*) – dt bias。M0 必须为 `None`。
- **initial_states** (*Tensor, optional*) – 初始状态。M0 必须为 `None`。
- **dfinal_state** (*Tensor, optional*) – final state 的上游梯度，形状
  `(B, H, P, N)`。默认值：`None`。
- **dt_softplus** (*bool*) – 是否对 dt 使用 softplus。M0 必须为 `False`。
- **dt_limit_min** (*float*) – dt clamp 下界。M0 必须为 `0.0`。
- **dt_limit_max** (*float*) – dt clamp 上界。M0 必须为默认最大 FP32 值。

## 支持的数据类型

`torch.float32`

## Shape

- `x`, `dout`, `dx`: `(B, L, H, P)`
- `dt`, `ddt`: `(B, L, H)`
- `A`: `(H,)`
- `B`, `C`, `dB`, `dC`: `(B, L, G, N)`
- `final_state`, `dfinal_state`: `(B, H, P, N)`
- `dA_partial`: `(B, H)`；public autograd 对 batch 归约得到 `(H,)`

其中 `B` 为 batch，`L` 为序列长度，`H` 为 head 数，`P` 为 head dimension，
`G` 为 group 数，`N` 为 state dimension。

## 约束条件

- 所有必选输入必须是 contiguous FP32 NPU Tensor。
- 当前 native M0 要求 `P=64`、`N=64`、`L % 64 == 0`，且 `H % G == 0`。
- 当前只支持 basic 路径：`D/z/dt_bias/initial_states=None`、
  `dt_softplus=False` 和默认 `dt_limit`。
- M0 使用每个 active AI Vector core 私有的 FP32 token-state workspace，目标是
  correctness，不代表最终内存和性能方案；完整 softplus/clamp 和大 shape 将由
  chunk/Cube M1 backward 提供。
- 不支持的训练规格会显式抛出 `NotImplementedError` 或 `TORCH_CHECK`，不会
  静默标记为 native 通过。

## 使用示例

```python
>>> import torch
>>> from ascend_kernel import mamba2_ssd_fwd
>>> Bsz, L, H, P, N, G = 1, 64, 1, 64, 64, 1
>>> x = torch.randn(Bsz, L, H, P, device="npu", dtype=torch.float32, requires_grad=True)
>>> dt = (0.005 + 0.01 * torch.rand(Bsz, L, H, device="npu")).requires_grad_()
>>> A = (-0.1 * torch.ones(H, device="npu")).requires_grad_()
>>> B = (0.1 * torch.randn(Bsz, L, G, N, device="npu")).requires_grad_()
>>> C = (0.1 * torch.randn(Bsz, L, G, N, device="npu")).requires_grad_()
>>> out, final_state = mamba2_ssd_fwd(
...     x, dt, A, B, C, chunk_size=64, return_final_state=True
... )
>>> loss = out.square().mean() + final_state.square().mean()
>>> loss.backward()
>>> [tensor.grad.shape for tensor in (x, dt, A, B, C)]
[torch.Size([1, 64, 1, 64]), torch.Size([1, 64, 1]), torch.Size([1]),
 torch.Size([1, 64, 1, 64]), torch.Size([1, 64, 1, 64])]
```

## 返回值

内部算子返回 8 个 Tensor：

1. `dx`，形状 `(B,L,H,P)`；
2. `ddt`，形状 `(B,L,H)`；
3. `dA_partial`，形状 `(B,H)`；
4. `dB`，形状 `(B,L,G,N)`；
5. `dC`，形状 `(B,L,G,N)`；
6. `dD_partial`；M0 为空 Tensor；
7. `dz`；M0 为空 Tensor；
8. `dinitial_states`；M0 为空 Tensor。

Public autograd 将 `dA_partial` 归约为 `(H,)`，并对不存在的 optional 输入返回
Python `None`。

## 验证

- public autograd pytest：7/7 通过，覆盖 output、output+final 和 final-only。
- native M0 精度矩阵：32/32 通过；每项梯度满足 `MERE < 2^-13`、
  `MARE < 10 * 2^-13`，且无 NaN/Inf。
- 完整结果见 `test/mamba2_ssd_bwd_precision_report.md`。
- M1 public gradient：40/40 full-feature case 通过，worst NRMSE `7.116e-4`，
  lowest cosine `0.999997914`。
- M1 fused DiagState：40/40 精度和 bitwise determinism 通过；8-case profiler
  相对同 NPU tensor composition 平均 `7.020×`。
