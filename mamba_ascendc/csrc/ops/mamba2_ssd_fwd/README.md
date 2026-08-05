# ascend_kernel.mamba2_ssd_fwd

```python
mamba2_ssd_fwd(
    x, dt, A, B, C, chunk_size,
    D=None, z=None, dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    initial_states=None,
    return_final_state=False,
)
```

在 Ascend NPU 上执行固定长度 Mamba-2 SSD forward。参数语义与
`mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` 对齐。

## Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `x` | Tensor | required | `[B,L,H,P]` 输入。 |
| `dt` | Tensor | required | `[B,L,H]` 时间步。 |
| `A` | Tensor | required | `[H]` head 衰减参数。 |
| `B` | Tensor | required | `[B,L,G,N]` variable B。 |
| `C` | Tensor | required | `[B,L,G,N]` variable C。 |
| `chunk_size` | int | required | 逻辑 chunk，优化路径支持 64/128。 |
| `D` | Tensor? | `None` | `[H]` 或 `[H,P]` skip 参数。 |
| `z` | Tensor? | `None` | 与 `x` 同形状的 SiLU gate。 |
| `dt_bias` | Tensor? | `None` | `[H]` dt bias。 |
| `dt_softplus` | bool | `False` | 对 `dt + dt_bias` 应用 softplus。 |
| `dt_limit` | tuple | `(0, inf)` | dt clamp 范围。 |
| `initial_states` | Tensor? | `None` | `[B,H,P,N]` 初始状态。 |
| `return_final_state` | bool | `False` | 是否同时返回最终状态。 |

## Supported dtypes

- 公开输入和输出：`float32`
- Cube 输入：内部转换为 FP16
- decay、state、Cube accumulate 和输出：FP32

## Shape

```text
x              [B,L,H,P]
dt             [B,L,H]
A              [H]
B, C           [B,L,G,N], H % G == 0
initial_states [B,H,P,N]
out            [B,L,H,P]
final_state    [B,H,P,N]
```

## Dispatch

公开 Python API 根据 shape 选择三条路径：

1. Cube/MIX 主路径：`P=64`、`N in {64,128}`、`chunk_size in {64,128}`、
   `L % 64 == 0`，并启用 `MAMBA_ASCENDC_CHUNK_MIX=1`。
2. aligned fallback：维度 16 对齐、`P*N <= 8192`。
3. direct recurrence fallback：处理其余合法 FP32 shape。

主路径执行：

```text
preprocess -> chunk_mix -> state_epilogue
                         -> state_passing + off_epilogue（部分小任务）
```

ChunkMix 使用 1 AIC + 2 AIV 的 MIX block。Cube 完成 `C@B^T`、`W@xdt`、
`B^T@weighted_x` 和 state projection；Vector 完成 decay、mask、transpose、
state recurrence 和 D/z epilogue。

## Constraints

- tensor 必须位于 NPU、contiguous、FP32。
- `B/C` shape 必须一致，且 `H % G == 0`。
- `chunk_size > 0`，`dt_limit[0] <= dt_limit[1]`。
- 当前不支持 `seq_idx`、`cu_seqlens`、varlen 和 backward。
- `MAMBA_ASCENDC_CHUNK128=1` 启用保留的 chunk-128 实现。
- `MAMBA_ASCENDC_NATIVE_PREPROCESS=0` 使用最终 AscendC preprocess。

## Returns

- `return_final_state=False`：返回 `out [B,L,H,P]`。
- `return_final_state=True`：返回 `(out, final_state)`。

底层 raw op
`torch.ops.mamba_ascend.mamba2_ssd_fwd(...) -> (Tensor, Tensor)`始终返回两个
Tensor，它是 direct fallback，不代表 Cube/MIX 公开路径。

## Example

```python
import torch
from ascend_kernel import mamba2_ssd_fwd

device = "npu:0"
x = torch.randn(1, 128, 2, 64, device=device, dtype=torch.float32)
dt = torch.rand(1, 128, 2, device=device, dtype=torch.float32)
A = -torch.rand(2, device=device, dtype=torch.float32)
B = torch.randn(1, 128, 1, 64, device=device, dtype=torch.float32)
C = torch.randn(1, 128, 1, 64, device=device, dtype=torch.float32)

out, final_state = mamba2_ssd_fwd(
    x, dt, A, B, C,
    chunk_size=64,
    dt_softplus=True,
    return_final_state=True,
)
```

## Validation

```bash
python mamba_ascendc/csrc/ops/mamba2_ssd_fwd/test/run_mamba2_ssd_fwd_aligned_precision.py
python mamba_ascendc/csrc/ops/mamba2_ssd_fwd/test/run_mamba2_ssd_chunk_mix_precision_report.py
python benchmarks/mamba2_npu_final_bench.py --cases medium extreme --skip-precision
```

## Pip installation

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
bash scripts/build_mamba_ascendc_wheel.sh
python -m pip install dist/mamba_ascendc-0.1.0-*.whl --no-deps
```

wheel 已携带 custom OPP 与 ACLNN bridge。`import ascend_kernel` 会自动配置
`ASCEND_CUSTOM_OPP_PATH`、预加载 `libcust_opapi.so` 并启用 Cube/MIX dispatch。
目标环境仍需预装兼容的 CANN、PyTorch 和 torch_npu。

最近一次全量回归：主扩展与子算子 31/31 UT 通过，公开 FWD 31/31 case
通过，Cube/MIX 主路径 35/35 case 通过。
