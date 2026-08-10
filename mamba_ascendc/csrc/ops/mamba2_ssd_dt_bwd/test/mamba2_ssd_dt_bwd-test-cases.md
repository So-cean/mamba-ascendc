# Mamba2SsdDtBwd 用例设计文档

## 1. 算子标杆

```python
def dt_bwd_ref(x, d_xdt, g_cs, dt, A, bias, softplus, limits):
    B, H, K, T, P = d_xdt.shape
    u = dt + (0.0 if bias is None else bias)
    q_pre = torch.nn.functional.softplus(u) if softplus else u
    q = q_pre.clamp(*limits)
    q_h = q.permute(0, 2, 1).reshape(B, H, K, T)
    x_h = x.permute(0, 2, 1, 3).reshape(B, H, K, T, P)
    gw = g_cs.flip(-1).cumsum(-1).flip(-1)
    dq = (d_xdt * x_h).sum(-1) + gw * A.view(1, H, 1, 1)
    chain = ((q_pre >= limits[0]) & (q_pre <= limits[1])).float()
    if softplus:
        chain *= torch.sigmoid(u)
    ddt_h = dq * chain.permute(0, 2, 1).reshape(B, H, K, T)
    dx = (d_xdt * q_h[..., None]).reshape(B, H, K*T, P).permute(0, 2, 1, 3)
    return dx, ddt_h.reshape(B, H, K*T).permute(0, 2, 1), \
        (gw*q_h).sum((0, 2, 3)), ddt_h.sum((0, 2, 3))
```

NPU 调用：

```python
torch.ops.mamba_ascend.mamba2_ssd_dt_bwd(
    x, d_xdt_total, g_dA_cs_total, dt, A, dt_bias,
    dt_softplus, dt_limit_min, dt_limit_max
)
```

## 2. 测试配置

```python
SUPPORTED_DTYPES = [torch.float32]

# [B,L,H,P,T]
TEST_SHAPES = [
    ("Single", "one stream one T64 chunk", (1, 64, 1, 64, 64)),
    ("Heads", "two heads two T64 chunks", (1, 128, 2, 64, 64)),
    ("T128", "three heads one T128 chunk", (1, 128, 3, 64, 128)),
    ("Batch", "batch2 four heads T64", (2, 256, 4, 64, 64)),
    ("Scale", "batch2 eight heads T128", (2, 512, 8, 64, 128)),
    ("Scale", "batch4 four heads T128", (4, 256, 4, 64, 128)),
    ("Chunks", "eight T64 chunks", (2, 512, 4, 64, 64)),
    ("Occupancy", "batch4 eight heads", (4, 128, 8, 64, 128)),
]

BOUNDARY_VALUES = [
    "plain_positive_dt",
    "bias_finite_clamp",
    "bias_softplus",
    "bias_softplus_finite_clamp",
]
```

8 shapes × 4 feature modes = 32 常规 FP32 用例。另有 8 个定向边界用例：

1. `q_pre == min`；
2. `q_pre == max`；
3. min/max 两侧各一个 ULP；
4. softplus `u=-30`；
5. softplus `u=+30`；
6. `A=0`；
7. `g_cs` 仅首 token 非零；
8. `g_cs` 仅末 token 非零且 `K=4`，检查反向 scan 不跨 chunk。

总计 40 个精度用例，另加 1 个 FP64 方向导数用例。

## 3. 判定

- dtype：仅 FP32。
- 生态精度阈值：`MERE < 2^-13` 且 `MARE < 10 * 2^-13`。
- 对 `dx/ddt/dA/dt_bias` 分别计算 MERE、MARE、MaxAbsErr、MeanAbsErr、CosSim。
- 算子使用 `[H,B,K]` partial 和第二次原生 kernel 固定顺序归约；同一输入重复
  5 次要求 `dA/dt_bias` bitwise 一致。
