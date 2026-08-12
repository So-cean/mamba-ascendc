# Validation map

本仓库的测试分布在三个位置，因为它们验证的边界不同。完整源码和构建关系见
[`../STRUCTURE.md`](../STRUCTURE.md)。

## Test suites

| Suite | Path | Device | Purpose |
|---|---|---|---|
| Mathematical oracle | `tests/mamba2/test_reference.py` | CPU | SSD 数学语义 |
| Vision-Mamba2 | `tests/mamba2/test_vmamba2_network.py` | CPU/GPU/NPU | 纯网络结构与 backend 接入 |
| A100 equivalence | `tests/mamba2/test_gpu_reference.py` | CUDA | upstream `mamba_ssm` 对拍 |
| Triton-Ascend | `tests/mamba2/test_npu_compile.py`、`tests/mamba2/test_npu_triton_fwd.py` | NPU | DSL 编译与 FWD 精度 |
| AscendC public API | `mamba_ascendc/tests/` | NPU | FWD、public autograd BWD、dispatch |
| AIV/fallback component | `mamba_ascendc/csrc/ops/*/test/` | NPU | 单算子 precision/profiler/sanitizer |
| Cube/MIX component | `mamba_ascendc_ops/mamba2_ssd_chunk_mix/test/` | NPU | OPP kernel 与架构专项 |
| Package | `mamba_ascendc/tests/test_pip_package.py` | NPU | bundled OPP 与 extension 加载 |

## CPU CI

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
pytest -q \
  tests/mamba2/test_reference.py \
  tests/mamba2/test_vmamba2_network.py
```

## Source-candidate NPU test

开发时先构建两个工程，然后显式加载候选产物：

```bash
MAMBA2_TEST_EXTENSION_LIB=$PWD/mamba_ascendc/python/ascend_kernel/ascend_kernel/lib/libascend_kernel.so \
MAMBA2_TEST_OPP_ROOT=$PWD/mamba_ascendc_ops/mamba2_ssd_chunk_mix/build_out/packages \
python mamba_ascendc/tests/run_source_candidate.py <test-file.py>
```

不要在候选验证阶段依赖已安装 wheel；否则旧 bundled OPP 可能覆盖刚编译的 kernel。

## Required gates

| Change | Minimum gate |
|---|---|
| Reference/API | CPU oracle + public backend tests |
| Triton-Ascend | NPU compile + precision + same-NPU benchmark |
| Vector/fallback kernel | component precision + public precision + target SoC |
| Cube/MIX kernel | component precision + public precision + 910B3/950PR dual gate |
| Backward | 41-case public M1 precision + directional/final-state gradients |
| Tiling/buffer/offset | 上述 precision + mssanitizer |
| Release wheel | source candidate green + package test + clean-process import |

性能变化还必须使用相同 shape 和 feature 重新执行 device-event benchmark；
profiler 数据用于解释瓶颈，不能替代端到端 latency。

## Result reporting

每次设备验证至少记录：

- Git commit 和候选产物 hash；
- device、CANN、PyTorch、torch_npu 版本；
- shape 顺序 `[B,L,H,P,N,chunk,G]`、dtype 和 optional features；
- pass/total、worst NRMSE、lowest cosine、finite/determinism；
- source-direct 或 packaged-wheel 加载方式。

公开仓库只保留最终精度报告和 README benchmark 快照；完整运行日志、原始 profiler
和失败候选保留在本地实验目录。
