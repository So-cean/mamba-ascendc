# AscendC Forward 说明

## 实现组成

| 层次 | 路径 | 作用 |
|---|---|---|
| Python API | `mamba_ascendc/python/ascend_kernel/ascend_kernel/mamba2.py` | 参数检查、shape dispatch、fallback |
| PyTorch 注册 | `mamba_ascendc/csrc/register.cpp` | 注册 `torch.ops.mamba_ascend.*` |
| Preprocess | `mamba_ascendc/csrc/ops/mamba2_ssd_preprocess/` | dt、dA cumsum、xdt、B/C layout |
| Chunk MIX bridge | `mamba_ascendc/csrc/aclnn/mamba2_ssd_chunk_mix.cpp` | 调用 OPS 工程 ACLNN 接口 |
| Chunk MIX | `mamba_ascendc_ops/mamba2_ssd_chunk_mix/op_kernel/` | chunk-64/128 Cube/Vector 主计算 |
| State epilogue | `mamba2_ssd_state_epilogue.cpp` | state passing、Cube projection、D/z |
| Generic fallback | `mamba_ascendc/csrc/ops/mamba2_ssd_fwd/` | 通用 FP32 recurrence |
| Reference | `mamba_torch/ssd_reference.py` | 数值基准 |

`mamba_ascendc` 和 `mamba_ascendc_ops` 都是构建 wheel 的必要组成部分。

## 构建与安装

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
bash scripts/build_mamba_ascendc_wheel.sh
python -m pip install dist/mamba_ascendc-0.1.0-*.whl --no-deps
```

wheel 包含 Python API、PyTorch 扩展、ACLNN bridge 和 custom OPP。目标环境仍需
预装版本兼容的 CANN、PyTorch 和 torch_npu。

## 调用

```python
from ascend_kernel import mamba2_ssd_fwd

out, final_state = mamba2_ssd_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size=128,
    D=D,
    z=z,
    dt_bias=dt_bias,
    dt_softplus=True,
    return_final_state=True,
)
```

## 验证

```bash
python mamba_ascendc/csrc/ops/mamba2_ssd_fwd/test/run_mamba2_ssd_fwd_aligned_precision.py
python mamba_ascendc/csrc/ops/mamba2_ssd_fwd/test/run_mamba2_ssd_chunk_mix_precision_report.py
python benchmarks/mamba2_npu_final_bench.py --cases medium extreme --skip-precision
```

当前公开精度矩阵包含 31 个 API case 和 35 个 Cube/MIX 主路径 case。Forward
输出和 final state 均与 `mamba_torch.ssd_reference` 比较。Backward 尚未实现。

## 支持边界

- 当前优化路径面向 Ascend 910B、FP32 API 和 variable B/C。
- 支持 `D`、`z`、`dt_bias`、`dt_softplus`、`initial_states` 和 final state。
- 主要优化规格使用 chunk size 64 或 128；其他规格进入 aligned/direct fallback。
- 构建产物、profiler trace、集群作业和内部实验记录不纳入源码仓库。
