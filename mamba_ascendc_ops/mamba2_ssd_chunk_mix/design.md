# Mamba2SsdChunkMix OPS 设计

## 接口

```text
Mamba2SsdChunkMix(
    x_cube:       FP16 [B,H,K,T,P],
    d_a_cumsum:  FP32 [B,H,K,T],
    b_cube:       FP16 [B,K,G,N,T],
    c_cube:       FP16 [B,K,G,T,N],
) -> (
    y_diag:       FP32 [B,H,K,T,P],
    chunk_states: FP32 [B,H,K,N,P] or [B,H,K,P,N],
)
```

支持 `(T,N,P)=(64,64,64)`、`(64,128,64)`、`(128,128,64)`，且
`H % G == 0`。

## 计算

每个 chunk 计算：

```text
CB = C @ B^T
W  = tril(CB * exp(dA_i - dA_j))
y_diag = W @ xdt
chunk_states = B^T @ (xdt * exp(dA_last - dA))
```

Cube 完成三个矩阵乘，Vector 完成 B transpose、decay、causal mask 和 FP16 cast。
所有 Cube accumulator 和输出为 FP32。

## 并行

- kernel task type：`MIX_AIC_1_2`。
- 两个 AIV 各负责一半 token rows。
- 小任务以 `(B,H,K)` 分发，获得 head 并行度。
- `(B,G,K)` 足以占满 AIC 时切换 group-task，同核复用 B transpose 和 CB。
- Host 从 platform API 查询 AIC/AIV 和 Matmul system workspace。

## TilingKey

| Key | T | N | State layout |
|---:|---:|---:|---|
| 2 | 64 | 64 | `[N,P]` |
| 3 | 64 | 128 | legacy `[P,N]` |
| 7 | 128 | 128 | `[N,P]` |

## UB 和 workspace

T64 每个 AIV 约使用 41.9 KiB UB。T128 每个 AIV 约使用 182.8 KiB，接近
910B 192 KiB 上限，因此 T128 不做第二套输入/计算 buffer。

每个 MIX core 的 GM workspace 包含两个 weighted-X slots、两个 W slots、一个
FP32 CB slot；Key 3 另有 legacy B slot。AIC/AIV 用 event ID 在两个 slots 上交替，
避免覆盖仍在消费的数据。

## 验证

- `test/test_mamba2_ssd_chunk_mix_basic.py`
- `test/test_mamba2_ssd_chunk_mix_128.py`
- `test/test_mamba2_ssd_off_epilogue.py`
- `test/test_mamba2_ssd_state_epilogue.py`
- `test/mamba2_ssd_mssanitizer_test.py`

完整公开 API 精度由
`mamba_ascendc/csrc/ops/mamba2_ssd_fwd/test/run_mamba2_ssd_chunk_mix_precision_report.py`
验证。
