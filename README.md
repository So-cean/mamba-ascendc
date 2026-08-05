# Mamba-2 on Ascend

Mamba-2 SSD forward 的三个可对照实现：纯 PyTorch reference、
Triton-Ascend 和 AscendC。仓库同时提供官方 `mamba_ssm` 的 A100 benchmark，
用于采用相同 shape、dtype、warmup 和 repeat 做跨平台比较。

> 当前范围仅为 forward。Backward 尚未实现；AscendC 是 NPU 正式优化路径，
> Triton-Ascend 保留为实现与数值对照路径。

## 目录

| 路径 | 内容 |
|---|---|
| `mamba_torch/` | 纯 PyTorch SSD reference 与纯 Vision-Mamba2 网络 |
| `mamba_triton_ascend/mamba2/` | Mamba-2 Triton-Ascend forward |
| `mamba_ascendc/` | AscendC PyTorch 扩展、Python API 和通用 fallback |
| `mamba_ascendc_ops/` | AscendC Cube/Vector OPS 工程 |
| `benchmarks/` | A100、Triton-Ascend、AscendC 和 Vision-Mamba2 benchmark |
| `examples/` | 三种实现的最小调用示例 |
| `tests/mamba2/` | PyTorch、A100 和 Triton-Ascend 精度测试 |

旧 Selective Scan 的 `ops/`、`ops2/`、调试脚本、集群作业和内部实验记录不属于
本仓库的公开交付范围。

## 输入接口

三个 SSD API 采用相同的主要参数：

- `x`: `[B, L, H, P]`
- `dt`: `[B, L, H]`
- `A`: `[H]`
- `B`, `C`: `[B, L, G, N]`
- `chunk_size`: `int`
- 可选：`D`、`z`、`dt_bias`、`initial_states`、`dt_softplus`

### 1. PyTorch reference

```python
from mamba_torch import ssd_chunk_scan_ref

out, final_state = ssd_chunk_scan_ref(
    x, dt, A, B, C, chunk_size=128,
    return_final_state=True,
)
```

完整示例：`python examples/mamba2_reference_demo.py`。

### 2. Triton-Ascend

需要已安装兼容的 CANN、PyTorch/torch_npu 和 Triton-Ascend：

```python
from mamba_triton_ascend.mamba2 import mamba_chunk_scan_combined

out, final_state = mamba_chunk_scan_combined(
    x, dt, A, B, C, chunk_size=128,
    return_final_states=True,
    backend="triton",
)
```

输入必须位于 NPU。完整示例：
`python examples/mamba2_triton_ascend_demo.py`。

### 3. AscendC

设置 CANN 环境后构建并安装包含 custom OPP 的 wheel：

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
bash scripts/build_mamba_ascendc_wheel.sh
python -m pip install dist/mamba_ascendc-0.1.0-*.whl --no-deps
```

```python
from ascend_kernel import mamba2_ssd_fwd

out, final_state = mamba2_ssd_fwd(
    x, dt, A, B, C, chunk_size=128,
    return_final_state=True,
)
```

完整示例：`python examples/mamba2_ascendc_demo.py`。详细支持范围见
[`docs/fwd/README.md`](docs/fwd/README.md)。

## 精度测试

```bash
# CPU reference
pytest -q tests/mamba2/test_reference.py

# A100 official mamba_ssm vs PyTorch reference
pytest -q tests/mamba2/test_gpu_reference.py

# Triton-Ascend vs PyTorch reference
pytest -q tests/mamba2/test_npu_compile.py tests/mamba2/test_npu_triton_fwd.py
```

AscendC 的公开精度入口位于：

```bash
python mamba_ascendc/csrc/ops/mamba2_ssd_fwd/test/run_mamba2_ssd_fwd_aligned_precision.py
python mamba_ascendc/csrc/ops/mamba2_ssd_fwd/test/run_mamba2_ssd_chunk_mix_precision_report.py
```

## 性能比较

三个算子 benchmark 使用相同 case 表和 FP32 输入：

```bash
# A100：官方 mamba_ssm
python benchmarks/mamba2_gpu_bench.py --cases medium extreme

# Ascend NPU：Triton-Ascend
python benchmarks/mamba2_triton_ascend_bench.py --cases medium extreme

# Ascend NPU：AscendC
python benchmarks/mamba2_npu_final_bench.py --cases medium extreme --skip-precision
```

默认输出为一行一个 JSON。对比时应使用同一 case、warmup、repeat、dtype，且只比较
SSD forward，不把数据生成、模型其他层或进程启动时间计入。纯 Vision-Mamba2 网络
可通过 `benchmarks/vmamba2_network_bench.py` 单独测试。

## 已验证环境

- Ascend 910B / CANN 8.2.RC1 / torch_npu 2.6.0
- Triton-Ascend 3.2.0
- NVIDIA A100 80GB / 官方 `mamba_ssm`

环境安装位置、集群节点、分区和调度策略由使用者自行配置，不属于仓库内容。
