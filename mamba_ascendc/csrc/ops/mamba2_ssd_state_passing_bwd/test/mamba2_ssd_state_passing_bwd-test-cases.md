# Mamba2SsdStatePassingBwd 用例设计文档

## 1. 算子标杆

PyTorch FP32 reference：

```python
def state_passing_bwd_ref(states_start, d_states_start, dA_cumsum, dfinal=None):
    g = torch.zeros_like(states_start[:, :, 0]) if dfinal is None else dfinal
    d_chunk_states = torch.empty_like(states_start)
    dA_chunk_last = torch.empty(states_start.shape[:3], device=states_start.device)
    for chunk in range(states_start.shape[2] - 1, -1, -1):
        alpha = torch.exp(dA_cumsum[:, :, chunk, -1])
        d_chunk_states[:, :, chunk] = g
        dA_chunk_last[:, :, chunk] = (
            g * states_start[:, :, chunk]
        ).sum(dim=(-2, -1)) * alpha
        g = d_states_start[:, :, chunk] + alpha[..., None, None] * g
    return d_chunk_states, g, dA_chunk_last
```

NPU 调用：

```python
dU, dinitial, dA_last = torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd(
    states_start, d_states_start, dA_cumsum, dfinal_state
)
```

## 2. 用例说明

### 2.1 测试配置

```python
SUPPORTED_DTYPES = [torch.float32]

# (category, description, [B,H,K,P,N,T])
TEST_SHAPES = [
    ("single", "one stream one chunk T64/N64",       (1, 1, 1, 64, 64, 64)),
    ("single", "one stream four chunks T64/N64",     (1, 1, 4, 64, 64, 64)),
    ("heads",  "two heads two chunks T64/N64",       (1, 2, 2, 64, 64, 64)),
    ("batch",  "two batches two heads T64/N64",      (2, 2, 2, 64, 64, 64)),
    ("n128",   "N128 with T64",                      (1, 2, 2, 64, 128, 64)),
    ("t128",   "N128 with T128",                     (1, 2, 2, 64, 128, 128)),
]

GENERAL_SHAPES = [
    ("Small", "minimum K with four heads",            (1, 4, 1, 64, 64, 64)),
    ("Large", "eight chunks four heads N64",          (1, 4, 8, 64, 64, 64)),
    ("Large", "four chunks four heads N128",          (1, 4, 4, 64, 128, 128)),
    ("Large", "two batches eight heads N128",         (2, 8, 2, 64, 128, 128)),
]

BOUNDARY_VALUES = [
    "dfinal_none",
    "dfinal_zero",
    "dfinal_random",
]
```

每个 shape 遍历三种 final-state gradient 模式，共 30 个 FP32 case。

### 2.2 用例覆盖统计

| 类别 | Shape 数量 | 边界值数量 | dtype 数量 | 总用例数 |
|---|---:|---:|---:|---:|
| 常规形状 | 6 | 3 | 1 | 18 |
| 泛化形状 | 4 | 3 | 1 | 12 |
| **总计** | **10** | **3** | **1** | **30** |

## 3. 使用说明

### 生成测试数据示例

```python
generator = torch.Generator(device="cpu").manual_seed(20260807)
B, H, K, P, N, T = shape
states_start = 0.1 * torch.randn(B, H, K, P, N, generator=generator)
d_states_start = 0.1 * torch.randn(B, H, K, P, N, generator=generator)
# Negative prefix sums model stable Mamba decay and exercise ScalarExp.
dA_cumsum = -0.01 - 0.2 * torch.rand(B, H, K, T, generator=generator)
dfinal_state = 0.1 * torch.randn(B, H, P, N, generator=generator)
```

### 注意事项

1. Shape 顺序固定为 `[B,H,K,P,N,T]`，不是 public SSD 的 `[B,L,H,P,N,C,G]`。
2. `dfinal_none` 验证 optional 缺省路径；`dfinal_zero` 验证显式零 tensor 与缺省等价。
3. 逐项检查 `d_chunk_states`、`dinitial_state`、`dA_chunk_last`。
4. `K=1` 覆盖初末 chunk 重合；`K=8` 覆盖较长 reverse recurrence。
5. NPU 算子只接受 design 中列出的 FP32 contiguous P/N/T 规格；非法输入另做 Host error tests。
