# mamba_ascendc

Mamba-2 SSD forward 的 PyTorch 扩展和 Python API。该工程负责注册
`torch.ops.mamba_ascend.*`，并通过 aclnn bridge 调用独立 OPS 工程中的
Cube/MIX kernels。

## 工程关系

```text
ascend_kernel.mamba2_ssd_fwd
  ├─ mamba2_ssd_preprocess             本工程，Vector
  ├─ mamba2_ssd_chunk_mix              mamba_ascendc_ops，Cube + Vector
  ├─ mamba2_ssd_state_epilogue         mamba_ascendc_ops，Cube + Vector
  ├─ mamba2_ssd_off_epilogue           mamba_ascendc_ops，小任务 fallback
  ├─ mamba2_ssd_state_passing          本工程，aligned fallback
  ├─ mamba2_ssd_prepare                本工程，aligned fallback
  └─ mamba2_ssd_fwd                    本工程，通用 direct fallback
```

正式发布会将两个工程的产物合并进同一个 `mamba-ascendc` wheel；用户不需要
单独安装 OPS 工程生成的 `.run` 包。

## Pip wheel

在已配置 CANN、PyTorch 和 torch_npu 的 NPU 环境中执行：

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
scripts/build_mamba_ascendc_wheel.sh
python -m pip install dist/mamba_ascendc-0.1.0-*.whl --no-deps
```

wheel 包含 OPS kernel binary、custom OPP、`libcust_opapi.so` 和
`libascend_kernel.so`。导入 `ascend_kernel` 时会自动完成运行时配置；不再需要
手工设置 `ASCEND_CUSTOM_OPP_PATH` 或 `MAMBA_CHUNK_MIX_OP_API_LIB`。

`mamba_ascendc/build.sh` 仍是统一打包脚本调用的内部扩展构建步骤。

## Python API

```python
from ascend_kernel import mamba2_ssd_fwd, mamba_chunk_scan_combined
```

完整接口、支持 shape 和 dispatch 规则见
`csrc/ops/mamba2_ssd_fwd/README.md`。
