# Mamba2SsdBwd 用例设计文档

## 1. 算子标杆

PyTorch 参考实现：

```python
from mamba_torch.ssd_reference import ssd_chunk_scan_ref

out, final_state = ssd_chunk_scan_ref(
    x, dt, A, B, C, chunk_size,
    D=D, z=z, dt_bias=dt_bias,
    dt_softplus=dt_softplus, dt_limit=dt_limit,
    initial_states=initial_states,
    return_final_state=True,
)
grads = torch.autograd.grad(
    (out, final_state),
    (x, dt, A, B, C, D, z, dt_bias, initial_states),
    (dout, dfinal_state),
    allow_unused=True,
)
```

小 shape 同时使用不强制 `.float()` 的 FP64 sequential recurrence 做方向导数标杆。
GPU 对照使用官方
`mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`。

---

## 2. 用例说明

### 2.1 测试配置

```python
SUPPORTED_DTYPES = [torch.float32]

TEST_SHAPES = [
    ("single_head",            (1, 64, 1, 64, 64, 64, 1)),
    ("shared_group_2h",        (1, 64, 2, 64, 64, 64, 1)),
    ("one_group_per_head_2h",  (1, 64, 2, 64, 64, 64, 2)),
    ("shared_group_4h",        (1, 64, 4, 64, 64, 64, 1)),
    ("two_groups_4h",          (1, 64, 4, 64, 64, 64, 2)),
    ("one_group_per_head_4h",  (1, 64, 4, 64, 64, 64, 4)),
    ("batch2_shared_group",    (2, 64, 2, 64, 64, 64, 1)),
    ("two_chunks",             (1, 128, 2, 64, 64, 64, 2)),
]

GRAD_MODES = [
    "output_only",
    "output_and_final",
    "final_only",
    "scaled_output_and_final",
]
```

### 2.2 用例覆盖统计

| 类别 | Shape 数量 | 边界值数量 | dtype 数量 | 总用例数 |
|---|---:|---:|---:|---:|
| single/multi-head 与 group 组合 | 6 | 4 | 1 | 24 |
| batch/sequence 泛化 | 2 | 4 | 1 | 8 |
| **总计** | **8** | **4** | **1** | **32** |

---

## 3. 使用说明

### 生成测试数据示例

```python
generator = torch.Generator(device="cpu").manual_seed(20260806)
Bsz, L, H, P, N, chunk, G = shape

def randn(*dims):
    return torch.randn(*dims, generator=generator, dtype=torch.float32)

x = randn(Bsz, L, H, P).requires_grad_()
dt = (0.005 + 0.01 * torch.rand(Bsz, L, H, generator=generator)).requires_grad_()
A = (-(0.1 + 0.2 * torch.rand(H, generator=generator))).requires_grad_()
B_tensor = (0.1 * randn(Bsz, L, G, N)).requires_grad_()
C_tensor = (0.1 * randn(Bsz, L, G, N)).requires_grad_()
dout = randn(Bsz, L, H, P)
dfinal_state = randn(Bsz, H, P, N)
```

### 注意事项

- Shape 顺序固定为 `[B,L,H,P,N,chunk,G]`。
- M0 native 报告只统计 `P=N=chunk=64` 的 basic 用例；M1 optional 与
  `N/chunk=128` 将在独立报告中统计，不能混入 M0 pass rate。
- optional 输入不存在时，public autograd 返回的对应梯度必须为 `None`。
- 相同 group 的多个 head 共享 B/C，测试必须检查 head 归约而非只测 `H=G=1`。
- output-only 与 output+final-state 使用相同输入和固定的上游梯度，便于逐项定位。
