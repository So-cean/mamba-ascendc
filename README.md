# Mamba-2 SSD for Ascend NPU

[![CPU CI](https://github.com/So-cean/mamba-ascendc/actions/workflows/ci.yml/badge.svg)](https://github.com/So-cean/mamba-ascendc/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/So-cean/mamba-ascendc)](https://github.com/So-cean/mamba-ascendc/releases)

Native **AscendC forward and backward** for Mamba-2 selective scan / Structured
State Space Duality (SSD), exposed to PyTorch through `torch_npu`. The project
includes a PyTorch reference, Triton-Ascend comparisons, A100 benchmarks and
Vision-Mamba2 model integration, with validation paths for Ascend 910B3 and 950PR.

面向昇腾 NPU 的 Mamba-2 SSD 原生算子：从 AscendC 前向、反向与 PyTorch 接入，
到同平台性能对照、A100 跨设备验证和 Vision-Mamba2 网络集成。

[Install / 安装](#installation--安装) · [Run an example / 使用](#usage--使用) ·
[Performance analysis / 性能分析](docs/performance.md) ·
[Source guide / 代码导航](STRUCTURE.md) ·
[Release v0.1.1](https://github.com/So-cean/mamba-ascendc/releases/tag/v0.1.1)

## Benchmark

![Same-device PyTorch, Triton-Ascend and AscendC comparison](assets/framework-decomposition.svg)

| Workload / 场景 | Published result / 已有结果 | Details / 详情 |
|---|---|---|
| Same-device SSD forward · 910B3 | **0.632 ms** AscendC; **9.94×** over PyTorch and **6.81×** over Triton-Ascend | [Implementation comparison](docs/performance.md#pytorch--triton-ascend--ascendc) |
| H256 SSD training forward · 910B3 vs A100 80GB PCIe | **31.888 ms** vs **22.583 ms**; **70.82%** of A100 throughput | [Matched-shape comparison](docs/performance.md#a100-80gb-与-ascend-910b3) |
| Vision-Mamba2 tiny · full-network forward | **1856.6 ms** on 910B3 vs **1430.8 ms** on A100; **0.771×** relative throughput | [Model integration](docs/performance.md#vision-mamba2-integration) |

These are three different workloads. Shapes use `[B,L,H,P,N,chunk,G]`: the
same-device comparison is `[4,2048,16,64,128,128,4]`, the H256 comparison is
`[8,4096,256,64,64,64,64]` with final state returned, and the 12-block vision model
uses batch 32 at `1024×1024`. Operator timings use device events and FP32 public
tensors; Cube kernels use FP16 operands with FP32 accumulation. The ratios apply
to these measured configurations.

三组结果分别对应同平台算子对照、大规格训练前向和完整网络前向；具体输入、
计时口径与精度记录见[性能分析](docs/performance.md)。

The separate A100 inference matrix covers **52/52** shapes, including sequence
length 16384 at **15.041 ms**. [Shape coverage and reproduction](benchmarks/README.md#forward).

## Start here / 阅读入口

| I want to… / 目的 | Entry / 入口 |
|---|---|
| Build and run the operator / 安装并运行 | [Installation](#installation--安装), then the [complete forward demo](examples/mamba2_ascendc_demo.py) |
| Use autograd / 接入训练 | [Training example](#ascendc-training) and [supported scope](#limitations-and-roadmap--当前限制与后续工作) |
| Reproduce or inspect a result / 复现与分析 | [Benchmark commands](benchmarks/README.md) · [Full analysis and figures](docs/performance.md) |
| Work on the kernels / 修改算子 | [Source structure](STRUCTURE.md) · [Test guide](tests/README.md) · [Contributing](CONTRIBUTING.md) |

## Installation / 安装

需要已配置的 CANN、PyTorch、torch_npu 和 AscendC 编译环境。已验证组合：

| 组件 | Ascend 910B3 | Ascend 950PR |
|---|---|---|
| CANN | 8.2.RC1 | 9.0.0 |
| Python | 3.11 | 3.11 |
| PyTorch / torch_npu | 2.6.0 / 2.6.0 | 2.7.1 / 2.7.1.post4 |
| Triton-Ascend | 3.2.0 | 3.2.1 |

PyTorch、torch_npu、CANN、CUDA 和 Triton-Ascend 属于平台绑定依赖，不由
本项目跨平台混装。先安装匹配的硬件软件栈，再选择 Python 依赖：

```bash
# CPU reference / repository development
python -m pip install -r requirements/dev.txt

# A100 benchmark, after installing CUDA PyTorch
python -m pip install -r requirements/gpu.txt

# Ascend, after installing CANN + torch + torch_npu + Triton-Ascend
python -m pip install -r requirements/npu.txt
```

依赖边界详见 [`requirements/README.md`](requirements/README.md)。

构建并安装包含 custom OPP 的 wheel：

```bash
git clone https://github.com/So-cean/mamba-ascendc.git
cd mamba-ascendc

export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
bash scripts/build_mamba_ascendc_wheel.sh
python -m pip install dist/mamba_ascendc-*.whl --no-deps

# On an allocated Ascend device
python examples/mamba2_ascendc_demo.py
```

wheel 包含 operator binary、custom OPP、`libcust_opapi.so` 和
`libascend_kernel.so`；安装后导入 `ascend_kernel` 会完成运行时注册。

## Usage / 使用

### AscendC inference

```python
import torch
import torch_npu
from ascend_kernel import mamba2_ssd_fwd

device = "npu"
B, L, H, P, N, G = 1, 128, 2, 64, 128, 1

x = torch.randn(B, L, H, P, device=device, dtype=torch.float32)
dt = torch.rand(B, L, H, device=device, dtype=torch.float32)
A = -torch.rand(H, device=device, dtype=torch.float32)
Bt = torch.randn(B, L, G, N, device=device, dtype=torch.float32)
Ct = torch.randn(B, L, G, N, device=device, dtype=torch.float32)

out, final_state = mamba2_ssd_fwd(
    x, dt, A, Bt, Ct,
    chunk_size=128,
    dt_softplus=True,
    return_final_state=True,
)
```

完整示例：[`examples/mamba2_ascendc_demo.py`](examples/mamba2_ascendc_demo.py)。

### AscendC training

This standalone example uses the M1 backward shape: contiguous FP32 with
`P=N=chunk_size=64`.

```python
import torch
import torch_npu
from ascend_kernel import mamba2_ssd_fwd

device = "npu"
B, L, H, P, N, G = 1, 128, 2, 64, 64, 1
x = torch.randn(B, L, H, P, device=device, dtype=torch.float32)
dt = torch.rand(B, L, H, device=device, dtype=torch.float32)
A = -torch.rand(H, device=device, dtype=torch.float32)
Bt = torch.randn(B, L, G, N, device=device, dtype=torch.float32)
Ct = torch.randn(B, L, G, N, device=device, dtype=torch.float32)
for tensor in (x, dt, A, Bt, Ct):
    tensor.requires_grad_(True)

out, final_state = mamba2_ssd_fwd(
    x, dt, A, Bt, Ct,
    chunk_size=64,
    dt_softplus=True,
    return_final_state=True,
)
loss = out.square().mean() + final_state.square().mean()
loss.backward()
```

当前 M1 native-core backward 需要 contiguous FP32、`P=N=chunk_size=64`。
不满足 M1 规格时，仅支持域内 basic case 会走 M0 correctness fallback；其他训练
规格显式抛出 `NotImplementedError`。带 `D/z/dt_bias/initial_states` 的完整特性
验证见[精度报告](mamba_ascendc/tests/mamba2_ssd_bwd_m1_precision_report_910b3-70pct-final-20260809.md)。

<details>
<summary>Reference and comparison backends / 参考与对照实现</summary>

### PyTorch reference

```python
from mamba_torch import ssd_chunk_scan_ref

out, final_state = ssd_chunk_scan_ref(
    x, dt, A, Bt, Ct,
    chunk_size=128,
    return_final_state=True,
)
```

### Triton-Ascend

```python
from mamba_triton_ascend.mamba2 import mamba_chunk_scan_combined

out, final_state = mamba_chunk_scan_combined(
    x, dt, A, Bt, Ct,
    chunk_size=128,
    return_final_states=True,
    backend="triton",
)
```

</details>

## Limitations and roadmap / 当前限制与后续工作

- Public dtype 当前为 FP32；Cube 内部使用 FP16 operand。
- M1 backward 当前只覆盖 contiguous `P=N=chunk=64`，varlen/packed sequence
  尚未实现。
- Forward 下一阶段目标是将 H256 提升到 A100 吞吐的 75%；优先处理
  StateEpilogue consumer layout、`y_diag/chunk_state` 物化和低密度 Cube 映射。
- Backward 优先共设计 Cube BMM 输出与 Vector consumer layout，再优化 Gate/Dt
  的 head-block contiguous tiling 和 AIC/AIV pipeline。
- 需要继续扩展 BF16、更多 `N/chunk_size`、`seq_idx/cu_seqlens` 和模型训练回归。

## Accuracy / 精度

AscendC 以 PyTorch reference 作为 oracle。Cube 内部使用 mixed precision，因此按
NRMSE、cosine、finite 和定向梯度检查共同验收，不宣称纯 FP32 GEMM 精度。

| 范围 | 结果 | 最差 NRMSE | 最低 cosine |
|---|---:|---:|---:|
| Grouped forward + determinism | `4/4 PASS` | 见原始报告 | 见原始报告 |
| M1 full-feature backward | `41/41 PASS` | `1.078e-3` | `0.999992847` |
| Vision-Mamba2 NPU/A100 logits | PASS | `3.418e-4` | `1.00000048` |

Backward 报告：
[`mamba2_ssd_bwd_m1_precision_report_910b3-70pct-final-20260809.md`](mamba_ascendc/tests/mamba2_ssd_bwd_m1_precision_report_910b3-70pct-final-20260809.md)。

## Implementation / 实现

| Path / 路径 | Forward | Backward | Purpose / 用途 |
|---|---|---|---|
| PyTorch reference | Yes | PyTorch autograd | Mathematical oracle / 数学与精度基线 |
| Triton-Ascend | Yes | No | Same-NPU DSL comparison / DSL 对照 |
| AscendC | Yes | M0/M1 public autograd | Primary NPU inference/training path / 主路径 |
| A100 `mamba_ssm` | Upstream | Upstream | Cross-platform baseline / 跨平台基线 |

公开接口对齐
`mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` 的张量语义：

```text
x                [B, L, H, P]
dt               [B, L, H]
A                [H]
B, C             [B, L, G, N]
initial_states   [B, H, P, N]       optional
out              [B, L, H, P]
final_state      [B, H, P, N]       optional
```

支持 `D`、`z`、`dt_bias`、`dt_softplus`、`dt_limit`、`initial_states` 和
`return_final_state`。Public tensor 为 FP32；Cube 内部使用 FP16 operand 与
FP32 accumulate。当前高性能 backward M1 路径要求 contiguous FP32 且
`P=N=chunk_size=64`。

### AscendC 执行路径

```text
ascend_kernel.mamba2_ssd_fwd
├─ Preprocess             AIV: softplus / clamp / decay / producer layout
├─ ChunkMix               AIC+AIV: chunk-local GEMMs and causal work
├─ Transpose              producer → state-consumer layout
└─ StateEpilogueTrain     AIC+AIV: state recurrence / projection / D/z

public autograd backward
├─ Prepare / PrepareDCB
├─ off-diagonal Cube BMM
├─ reverse StatePassing
├─ diagonal + chunk-state finalize
├─ DtBwd / Gate
└─ reductions and public-layout gradients
```

核心源码：

| 目录 | 内容 |
|---|---|
| [`mamba_torch/`](mamba_torch/) | PyTorch SSD reference 与 Vision-Mamba2 网络 |
| [`mamba_triton_ascend/mamba2/`](mamba_triton_ascend/mamba2/) | Triton-Ascend forward |
| [`mamba_ascendc/`](mamba_ascendc/) | PyTorch 扩展、public API、Vector 与 fallback kernels |
| [`mamba_ascendc_ops/`](mamba_ascendc_ops/) | AscendC Cube/MIX OPS 工程 |
| [`benchmarks/`](benchmarks/) | GPU/NPU benchmark 与 profiler 入口 |
| [`tests/mamba2/`](tests/mamba2/) | reference、GPU、Triton-Ascend 测试 |

`mamba_ascendc` 负责 public API、autograd、Vector/fallback kernel 和 PyTorch
扩展；`mamba_ascendc_ops` 负责 Cube/MIX custom OPP。二者由发布脚本合并成一个
wheel，不是两个竞争版本。完整修改边界和交接流程见
[`STRUCTURE.md`](STRUCTURE.md)，精简 source-of-truth 表见
[`docs/source-layout.md`](docs/source-layout.md)。

## Validation and reproduction / 验证与复现

开发阶段直接加载源码候选，不通过 wheel，避免环境中旧 OPP 抢先加载：

```bash
cd mamba_ascendc_ops/mamba2_ssd_chunk_mix
bash build.sh
cd ../../mamba_ascendc
MAMBA_ASCENDC_DEV_BUILD=1 ./build.sh <soc-version>
cd ..

MAMBA2_TEST_EXTENSION_LIB=$PWD/mamba_ascendc/python/ascend_kernel/ascend_kernel/lib/libascend_kernel.so \
MAMBA2_TEST_OPP_ROOT=$PWD/mamba_ascendc_ops/mamba2_ssd_chunk_mix/build_out/packages \
python mamba_ascendc/tests/run_source_candidate.py <test-file.py>
```

常用验证：

```bash
# CPU reference
pytest -q tests/mamba2/test_reference.py

# A100 mamba_ssm vs reference
pytest -q tests/mamba2/test_gpu_reference.py

# Triton-Ascend vs reference
pytest -q tests/mamba2/test_npu_compile.py tests/mamba2/test_npu_triton_fwd.py

# AscendC forward demo and M1 backward precision
python examples/mamba2_ascendc_demo.py
pytest -q mamba_ascendc/tests/test_mamba2_ssd_bwd_m1.py \
  mamba_ascendc/tests/test_mamba2_ssd_bwd_m1_precision.py
python mamba_ascendc/tests/run_mamba2_ssd_bwd_m1_precision_report.py
```

性能：

```bash
# A100 official mamba_ssm
python benchmarks/mamba2_gpu_bench.py \
  --suite standard --warmup 30 --repeat 200

# Triton-Ascend
python benchmarks/mamba2_triton_ascend_bench.py \
  --cases medium extreme --warmup 30 --repeat 200

# AscendC
python benchmarks/mamba2_npu_final_bench.py \
  --suite standard --warmup 30 --repeat 200

# Forward/backward unified benchmark
python benchmarks/mamba2_backward_bench.py \
  --backend ascendc --device npu --cases m1_extreme \
  --feature full --warmup 10 --repeat 30

# Regenerate README figures
python benchmarks/plot_readme_figures.py
```

benchmark 口径、字段和更多命令见 [`benchmarks/README.md`](benchmarks/README.md)。

公开 GitHub Actions 验证 CPU reference、纯 Vision-Mamba2 forward、Python
源码语法和 README 图表可复现性。AscendC 编译、NPU 精度与 msprof profiling
需要 910B3/950PR 专用环境，按上面的源码候选流程执行。

## Frequently asked questions / 常见问题

### Does Mamba2 run on Huawei Ascend NPU?

Yes. This repository implements the Mamba2 SSD/selective-scan core as a native
AscendC custom operator and provides test paths for Ascend 910B3 and 950PR.

### Is this primarily a Triton implementation?

No. AscendC is the main implementation. Triton-Ascend is retained as a same-NPU
comparison and migration reference.

### Are forward and backward implemented?

Forward and public-autograd backward are available. Forward covers generic,
aligned, and Cube/MIX paths; the fastest native backward path currently requires
contiguous FP32 with `P=N=chunk_size=64`.

### Are the A100 and Ascend benchmark shapes identical?

Yes. The two forward runners import the same canonical shape matrix and use
device-event timing. Published cross-platform ratios do not use theoretical
TFLOPS normalization.

## References / 参考项目

- [state-spaces/mamba](https://github.com/state-spaces/mamba) — Mamba/Mamba-2 官方实现
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- [triton-lang/triton](https://github.com/triton-lang/triton)
- [Ascend/triton-ascend](https://github.com/Ascend/triton-ascend)
- [cann/ops-transformer experimental Mamba](https://gitcode.com/cann/ops-transformer/tree/master/experimental/mamba) — fixed-shape AscendC Mamba-2 forward suboperators
- [Ascend/samples](https://github.com/Ascend/samples) — AscendC 自定义算子样例
- [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba)

提交问题或改动前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## Machine-readable metadata / 机器可读元数据

- [`codemeta.json`](codemeta.json) — CodeMeta JSON-LD software metadata
- [`CITATION.cff`](CITATION.cff) — citation metadata recognized by GitHub
- [`llms.txt`](llms.txt) — concise project map for generative-engine crawlers
- [`benchmarks/results/readme_benchmarks.json`](benchmarks/results/readme_benchmarks.json)
  — structured benchmark, precision, and profiler snapshot used by this README

README、CodeMeta 和 CFF 是项目事实与引用信息的主来源。`llms.txt`
是补充的生成式搜索导航文件，不替代 README、API 文档或原始测试数据。

## License

项目自研代码采用 [Apache License 2.0](LICENSE)。来自 Mamba、Ascend
agent-skills、Ascend custom-operator scaffold 和 Makeself 的文件保留各自原始
许可与版权声明；其中 ACLNN helper 使用
[Mulan PSL 2.0](LICENSES/MulanPSL-2.0.txt)。完整映射见 [`NOTICE`](NOTICE) 和
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
安全问题请按 [`SECURITY.md`](SECURITY.md) 私下报告。
