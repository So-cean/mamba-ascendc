# Repository structure and handoff guide

本文档说明 `mamba-ascendc` 的正式源码边界、执行路径、构建关系、测试层级和
benchmark 证据链。它面向继续开发 Forward、Backward、Ascend 910B3 或
Ascend 950PR 的接手者。

根目录 [`README.md`](README.md) 面向使用者和项目展示；本文档面向开发与交接。

## 1. 一分钟定位

| 要修改的内容 | Source of truth | 首要验证 |
|---|---|---|
| Mamba-2 数学语义 | `mamba_torch/ssd_reference.py` | `tests/mamba2/test_reference.py` |
| Public API、dispatch、autograd | `mamba_ascendc/python/ascend_kernel/ascend_kernel/mamba2.py` | `mamba_ascendc/tests/test_mamba2_ssd_{fwd,bwd_m1}.py` |
| PyTorch custom-op schema | `mamba_ascendc/csrc/register.cpp`、`mamba_ascendc/csrc/ops.h` | Python import 与 public API 测试 |
| AIV/Vector 或通用 fallback | `mamba_ascendc/csrc/ops/<op>/` | 对应目录测试 + public 回归 |
| AIC Cube/MIX | `mamba_ascendc_ops/mamba2_ssd_chunk_mix/op_{host,kernel}/` | OPS component 测试 + public 回归 |
| ACLNN bridge | `mamba_ascendc/csrc/aclnn/` | source-candidate public 回归 |
| Triton-Ascend 对照 | `mamba_triton_ascend/mamba2/` | `tests/mamba2/test_npu_triton_fwd.py` |
| A100 对照 | `benchmarks/mamba2_gpu_bench.py`、`mamba2_backward_bench.py` | 同 shape、同 feature 的 device-event benchmark |
| README 图和公开数字 | `benchmarks/results/readme_benchmarks.json` | `python benchmarks/plot_readme_figures.py` 后 `git diff --exit-code -- assets/` |
| Wheel 交付 | `scripts/build_mamba_ascendc_wheel.sh` | `mamba_ascendc/tests/test_pip_package.py` |

`mamba_ascendc_ops/mamba2_ssd_chunk_mix/` 是历史工程名。它现在包含多个 Forward
和 Backward Cube/MIX 子算子，不只包含 `chunk_mix`；不要仅因为目录名而将其他
OPS kernel 移出该工程。

## 2. 正式目录树

```text
mamba-ascendc/
├── README.md                       # 使用、最终 benchmark 与项目说明
├── STRUCTURE.md                    # 本交接文档
├── mamba_torch/                    # PyTorch 数学 oracle 与 Vision-Mamba2
├── mamba_triton_ascend/            # Triton-Ascend forward 对照
├── mamba_ascendc/                  # Public API、autograd、AIV/fallback、扩展
│   ├── csrc/
│   │   ├── register.cpp            # torch.ops.mamba_ascend schema/registration
│   │   ├── ops.h                   # C++ 声明
│   │   ├── aclnn/                  # 独立 OPS 的 ACLNN bridge
│   │   └── ops/<op>/               # op_host、op_kernel、design、局部测试
│   ├── python/ascend_kernel/        # pip package 与 runtime OPP discovery
│   └── tests/                       # Public AscendC FWD/BWD/打包/架构回归
├── mamba_ascendc_ops/
│   └── mamba2_ssd_chunk_mix/        # Cube/MIX custom OPP 工程
│       ├── op_host/                 # shape 检查、tiling、SoC dispatch
│       ├── op_kernel/               # AscendC AIC/AIV kernels
│       └── test/                    # OPS component 精度、profile、sanitizer
├── tests/mamba2/                    # CPU、GPU、Triton、模型集成入口
├── benchmarks/                      # 可复用的端到端 benchmark/profiler
├── examples/                        # Reference、Triton、AscendC 最小用法
├── assets/                          # 由公开 JSON 快照生成的 README SVG
├── docs/profiling/                  # 精选 profiling 报告和结构化分析
├── requirements/                    # common/dev/gpu/npu 分离依赖
└── scripts/                         # release build 与仓库一致性检查
```

以下目录不是正式源码：`build/`、`build_out/`、`output/`、`kernel_meta/`、
`opp_*`、`dist_candidate*`、raw profiler trace、外部 `mamba/` clone 和本地
`docs/plan|report`。它们均由 `.gitignore` 隔离。

## 3. Forward 数据流

Public 入口是：

```python
from ascend_kernel import mamba2_ssd_fwd
```

主要调用链：

```text
mamba2_ssd_fwd
└─ _mamba2_ssd_fwd_impl
   ├─ grouped/Cube-MIX path
   │  ├─ Preprocess
   │  ├─ ChunkMix or ChunkMixGrouped
   │  ├─ StatePassingGrouped
   │  ├─ StateProjectionGrouped
   │  └─ StateVectorEpilogueGrouped[Train]
   ├─ aligned path
   │  ├─ Preprocess / Prepare
   │  └─ StateEpilogue
   └─ generic fallback
      └─ mamba2_ssd_fwd direct kernel
```

Dispatch 与 feature 约束集中在
`mamba_ascendc/python/ascend_kernel/ascend_kernel/mamba2.py`。新增 shape 时，不能
只让底层 kernel 能运行；必须同时更新 dispatch 条件、public 精度测试和 README
支持范围。

## 4. Backward 数据流

Public autograd 有两个层级：

- M0：较小范围的 correctness fallback。
- M1：`P=N=chunk_size=64` 的 native-core 主路径。

M1 主要阶段：

```text
recompute/cache
├─ Prepare / PrepareDCB
├─ off-diagonal ChunkScan BMM
├─ reverse StatePassing
├─ diagonal + chunk-state finalize
├─ DtBwd / A / bias gradients
├─ Gate
└─ public-layout reductions
```

修改 backward 时至少同时检查：九项梯度、`final_state` 梯度语义、group-shared
`B/C` reduction、saved tensor unique storage 和 peak memory。只验证某个子算子
不能替代 public autograd 回归。

## 5. 两个 Ascend 工程的关系

`mamba_ascendc` 与 `mamba_ascendc_ops` 不是两个实现版本。

```text
mamba_ascendc_ops
  └─ custom OPP + libcust_opapi.so
                         \
                          ├─ scripts/build_mamba_ascendc_wheel.sh
                         /
mamba_ascendc
  └─ libascend_kernel.so + Python API
                          ↓
                one mamba-ascendc wheel
```

- `mamba_ascendc_ops` 使用 msOpGen/CANN OPS 构建流程，负责 Cube/MIX。
- `mamba_ascendc` 注册 `torch.ops.mamba_ascend.*`，并通过 ACLNN bridge 调用 OPP。
- 发布脚本将两者合并成一个 wheel；用户不应分别安装两个工程。
- `mamba_ascendc_ops/.../build.sh` 当前会重建 `build_out`，不是 kernel 级增量构建。
- 日常验证应直接加载源码 `.so + OPP`；wheel 只用于冻结后的安装验收。

## 6. 测试层级

详细命令见 [`tests/README.md`](tests/README.md)。测试不能只按文件数量汇总，
应按下面的 Gate 汇报：

| Gate | 目录 | 回答的问题 |
|---|---|---|
| G0 CPU oracle | `tests/mamba2/` | 数学语义和模型结构是否正确 |
| G1 backend equivalence | `tests/mamba2/` | A100/Triton 是否与 oracle 对齐 |
| G2 public AscendC | `mamba_ascendc/tests/` | 用户 API、dispatch、autograd 是否正确 |
| G3 component | `mamba_ascendc/csrc/ops/*/test/`、OPS `test/` | 单个 kernel/tiling 是否正确 |
| G4 architecture | 文件名含 `_950` 或 dual gate | 910B3/950PR 专用调度是否都通过 |
| G5 safety | `*mssanitizer*` | UB/GM 越界、非法访问和泄漏 |
| G6 package | `test_pip_package.py` | wheel 是否真正包含并加载候选 OPP |

对公共 kernel 的修改，验收顺序固定为：component precision → public precision →
双架构专项 → performance → mssanitizer（涉及 buffer/offset/tiling 时）→ wheel。

## 7. Benchmark 与 profiling 证据链

| 结论 | 入口 | 正式计时/数据 |
|---|---|---|
| A100 forward | `mamba2_gpu_bench.py` | CUDA event、p50 |
| AscendC forward | `mamba2_forward_training_bench.py` | NPU event、p50 |
| Forward/Backward training | `mamba2_backward_bench.py` | event、saved tensor、peak memory |
| PyTorch/Triton/AscendC | `mamba2_npu_decomposition_bench.py` | 同 NPU、同 shape |
| Forward profile | `mamba2_forward_npu_profile.py` | `torch_npu.profiler` |
| Backward profile | `mamba2_backward_npu_profile.py` | `torch_npu.profiler` |
| Vision-Mamba2 | `vmamba2_network_bench.py` | network/mixer/SSD 分范围 event |

规则：

1. 跨平台 latency 使用 device event；profiler 只解释阶段和 engine 行为。
2. Shape、dtype、optional feature、随机种子、final-state gradient 必须一致。
3. Public 结论只来自 public API，不用单核或单个子算子代替。
4. `benchmarks/results/readme_benchmarks.json` 是 README 数字和图的唯一公开快照。
5. Raw trace、候选 JSONL 和调度日志留在本地，不提交到公开仓库。

## 8. 当前平台交接状态

| 平台 | 当前可交接状态 | 注意事项 |
|---|---|---|
| Ascend 910B3 | README 快照对应当前公开主结果 | 公共源码变更仍需重新跑 precision、event 和 profiler |
| Ascend 950PR | 已存在并重新验证过可用的历史 checkpoint | 当前 HEAD 的 source-direct grouped fast path 在 2026-08-10 复测出现 output 回归；修复并完成新一轮门禁前，不应用历史 checkpoint 结果替代当前源码结果 |
| A100 80GB | `mamba_ssm` forward/backward 基线可复现 | 必须记录 `mamba_ssm` 版本和 final-state gradient 口径 |

950PR 回归的已知边界：final state 正确，grouped token output 错误；公共 generic
fallback 可以正确，但不能作为 grouped fast-path 性能结论。下一位开发者应以已通过
checkpoint 为对照，逐项检查共享 epilogue helper、D broadcast、队列/UB 复用和
64-token slab 变更，不要继续在未通过候选上叠加优化。

## 9. 交接工作流

### 修改数学或接口

1. 先更新 `mamba_torch` 和 CPU test。
2. 更新 public schema/API/dispatch。
3. 更新 AscendC/Triton backend。
4. 运行 public 精度，而不只运行 component test。
5. 更新 API README、support matrix 和 example。

### 修改 AscendC kernel

1. 确认 source of truth 位于 `csrc/ops` 还是 `mamba_ascendc_ops/.../op_kernel`。
2. 不编辑生成到 package `opp/`、`build_out/` 或 `kernel_meta/` 的副本。
3. 直接构建并加载 source candidate。
4. 每次只验证一个候选；失败候选不进入 wheel。
5. Shared kernel 变更必须分别在 910B3 和 950PR 验证。

### 更新公开 benchmark

1. 保存包含环境和 protocol 的原始本地结果。
2. 将最终、同口径数字人工整理进 `readme_benchmarks.json`。
3. 重新生成 `assets/*.svg`。
4. 检查 README 表格、JSON 和 SVG 数字一致。
5. 只提交精选 report，不提交完整 profiler archive。

## 10. 本地生成物与安全清理

本地常见生成物包括：

- `mamba_ascendc/build/`、`mamba_ascendc/output/`
- `mamba_ascendc_ops/**/build_out/`
- `kernel_meta/`、`mindstudio_sanitizer_log/`
- `opp_*`、`dist_candidate*`
- `benchmarks/profiles/`、除公开快照外的 `benchmarks/results/`
- 外部参考 clone：`mamba/`、`nd-Mamba2-torch/`

清理前先预览：

```bash
git status --short --ignored
git clean -ndX
```

不要直接对仓库根目录执行删除命令。确认某次实验已归档后，只清理对应的显式目录。
`git status` 干净只表示正式 tracked source 没有改动，不表示本地没有旧 OPP 或 wheel。

## 11. 还需要补充的公开证据

按优先级建议继续补充：

1. **当前源码的 950PR 重新验收**：同 shape 的 precision、FWD/BWD event、stage
   profile；在回归修复前不加入新的 950PR 对比图。
2. **Backward heavy scaling 图**：A100 与 910B3/950PR 从 H32 到 H256 的
   backward-only latency 和相对吞吐，确认两端均进入 sustained scaling。
3. **训练内存图**：saved-tensor unique storage、peak allocated 和 workspace 随
   head/batch 的变化，解释算子速度与训练可用性的共同约束。
4. **Support/dispatch matrix**：FP32/BF16、P/N/chunk、optional feature、FWD/BWD、
   910B3/950PR 的支持状态；这比继续增加优化版本对比更有价值。
5. **安全与稳定性摘要**：determinism、mssanitizer、重复运行和 wheel source-vs-
   packaged equivalence 的最终 Gate。

不建议在 README 展示 V1/V2/V21 等开发版本历史，也不建议用理论 TFLOPS 归一化
跨架构结果。README 保留最终实现、最终数字和能够解释结果的 profiling 即可。
