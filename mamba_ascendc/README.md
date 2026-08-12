# mamba_ascendc

Mamba-2 SSD forward/backward 的 PyTorch 扩展和 Python API。该工程负责注册
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
  ├─ mamba2_ssd_fwd                    本工程，通用 direct fallback
  ├─ mamba2_ssd_bwd                    本工程，M0 native correctness fallback
  ├─ mamba2_ssd_chunk_scan_bwd_off     OPS 工程，M1 Cube + Vector
  ├─ mamba2_ssd_chunk_scan_bwd_diag_state
  │                                      OPS 工程，M1 7-GEMM Cube + Vector
  ├─ mamba2_ssd_state_passing_bwd      本工程，M1 reverse state passing
  ├─ mamba2_ssd_dt_bwd                 本工程，M1 dt/A/bias chain
  └─ public backward glue              recompute/transpose/add/off-dC reduction
```

正式发布会将两个工程的产物合并进同一个 `mamba-ascendc` wheel；用户不需要
单独安装 OPS 工程生成的 `.run` 包。

完整 source-of-truth 与修改边界见
[`../docs/source-layout.md`](../docs/source-layout.md)。

## Pip wheel

在已配置 CANN、PyTorch 和 torch_npu 的 NPU 环境中执行：

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
scripts/build_mamba_ascendc_wheel.sh
python -m pip install dist/mamba_ascendc-*.whl --no-deps
```

wheel 包含 OPS kernel binary、custom OPP、`libcust_opapi.so` 和
`libascend_kernel.so`。导入 `ascend_kernel` 时会自动完成运行时配置；不再需要
手工设置 `ASCEND_CUSTOM_OPP_PATH` 或 `MAMBA_CHUNK_MIX_OP_API_LIB`。

`mamba_ascendc/build.sh` 仍是统一打包脚本调用的内部扩展构建步骤。

## 源码开发与候选验证

日常 kernel 迭代不构建或安装 wheel。扩展使用增量开发构建；OPS 工程当前会
重新生成 `build_out`，并将候选 kernel stage 到其中的 `packages`；测试进程在导入
`torch_npu` 前显式指定两项候选产物：

```bash
cd mamba_ascendc_ops/mamba2_ssd_chunk_mix
bash build.sh                       # 默认 target=install，不生成 .run
cd ../../

cd mamba_ascendc
MAMBA_ASCENDC_DEV_BUILD=1 ./build.sh <soc-version>
cd ..

MAMBA2_TEST_EXTENSION_LIB=/path/to/libascend_kernel.so \
MAMBA2_TEST_OPP_ROOT=$PWD/mamba_ascendc_ops/mamba2_ssd_chunk_mix/build_out/packages \
python mamba_ascendc/tests/run_source_candidate.py <test-file.py>
```

`run_source_candidate.py` 会直接注册候选 `.so` 和 `libcust_opapi.so`，并阻止
当前 Python 环境中已安装的 `ascend_kernel` wheel 抢先加载旧 OPP。只有精度、
性能和双架构回归完成后，才执行上面的 Pip wheel 发布流程；需要独立 OPS
发布包时再显式执行 `bash build.sh package`。

## Python API

```python
from ascend_kernel import mamba2_ssd_fwd, mamba_chunk_scan_combined
```

完整接口、支持 shape 和 dispatch 规则见
`csrc/ops/mamba2_ssd_fwd/README.md`。

Backward 同时保留 M0 correctness fallback 和 M1 native-core public autograd。
M1 的 off-diagonal、reverse state passing、diagonal/chunk-state 和 dt/A/bias
核心分支均已接入 native 子算子，支持 full optional 的九项梯度；public 层仍保留
少量 gating/add/shared-group reduction tensor glue。端到端 41-case 最终报告位于
`tests/mamba2_ssd_bwd_m1_precision_report_910b3-70pct-final-20260809.md`，分解设计见
`csrc/ops/mamba2_ssd_bwd/design.md`。
