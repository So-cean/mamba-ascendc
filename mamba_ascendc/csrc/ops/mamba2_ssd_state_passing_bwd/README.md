# torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd

```text
torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd(
    states_start, d_states_start, dA_cumsum, dfinal_state=None
) -> tuple[Tensor, Tensor, Tensor]
```

计算 Mamba2 SSD 在 chunk 维度上的反向 state recurrence。算子从最后一个
chunk 向前传播 state gradient，并同时计算每个 chunk contribution、initial
state 和 chunk-last decay 的梯度。

$$
\begin{aligned}
g_K &= dS_{\mathrm{final}}, \\
dU_k &= g_{k+1}, \\
d\alpha_k &= \langle g_{k+1}, S_k \rangle, \\
g_k &= dS_k + \alpha_k g_{k+1},
\quad \alpha_k=\exp(c_{k,T-1}).
\end{aligned}
$$

## 参数说明

- **states_start** (*Tensor*) – Forward 各 chunk 的输入 state，形状为
  `(B, H, K, P, N)`。
- **d_states_start** (*Tensor*) – Output branch 产生的 chunk-start state
  梯度，形状与 `states_start` 相同。
- **dA_cumsum** (*Tensor*) – 每个 chunk 内 `A * dt` 的 prefix sum，形状为
  `(B, H, K, T)`；本算子读取每个 chunk 的最后一个元素。
- **dfinal_state** (*Tensor, optional*) – Final state 的上游梯度，形状为
  `(B, H, P, N)`。默认值为 `None`，等价于全零张量。

## 支持的数据类型

`torch.float32`

## Shape

- **输入**:
  - `states_start`, `d_states_start`: `(B, H, K, P, N)`
  - `dA_cumsum`: `(B, H, K, T)`
  - `dfinal_state`: `(B, H, P, N)`，可选
- **输出**:
  - `d_chunk_states`: `(B, H, K, P, N)`
  - `dinitial_state`: `(B, H, P, N)`
  - `dA_chunk_last`: `(B, H, K)`

其中 `B` 是 batch size，`H` 是 head 数，`K` 是 chunk 数，`P` 是 head
dimension，`N` 是 state dimension，`T` 是 chunk size。

## 约束条件

- 所有输入必须位于同一 Ascend NPU，dtype 为 `torch.float32` 且连续。
- `B`、`H`、`K` 必须大于零。
- 当前 native 规格固定 `P=64`。
- `N` 仅支持 `64` 或 `128`。
- `T` 仅支持 `64` 或 `128`。
- `states_start` 与 `d_states_start` 的 shape 必须完全一致。
- `dA_cumsum` 的前三维必须与 `(B, H, K)` 一致。
- 提供 `dfinal_state` 时，其 shape 必须为 `(B, H, P, N)`。
- 这是完整 Mamba2 backward 的内部算子；单独调用不会生成 `dx/ddt/dA/dB/dC`。

## 使用示例

```python
>>> import torch
>>> import torch_npu
>>> import ascend_kernel
>>> B, H, K, P, N, T = 1, 2, 4, 64, 64, 64
>>> states_start = torch.randn(B, H, K, P, N, device="npu", dtype=torch.float32)
>>> d_states_start = torch.randn_like(states_start)
>>> dA_cumsum = -torch.rand(B, H, K, T, device="npu", dtype=torch.float32)
>>> dfinal_state = torch.randn(B, H, P, N, device="npu", dtype=torch.float32)
>>> dU, dinitial, dA_last = (
...     torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd(
...         states_start, d_states_start, dA_cumsum, dfinal_state
...     )
... )
>>> dU.shape, dinitial.shape, dA_last.shape
(torch.Size([1, 2, 4, 64, 64]), torch.Size([1, 2, 64, 64]), torch.Size([1, 2, 4]))
```

## 返回值

*tuple[Tensor, Tensor, Tensor]* – 依次返回：

1. `d_chunk_states`：每个 chunk contribution 的梯度，FP32；
2. `dinitial_state`：initial state 的梯度，FP32；
3. `dA_chunk_last`：每个 chunk 最后一个 cumsum 位置的梯度贡献，FP32。
