# Benchmarks

本目录提供 Mamba-2 SSD 的 PyTorch reference、A100 `mamba_ssm`、
Triton-Ascend 和 AscendC benchmark。历史七元组
`[B,L,H,P,N,chunk,G]` 是 workload 配置，不是纯 tensor shape；其中
`chunk` 是独立算法参数。新增结果将 tensor shape `[B,L,H,P,N,G]` 与
`chunk_size` 分列，并使用 FP32 public tensors。

## 入口总览

| 目标 | 脚本 | 公开结论 |
|---|---|---|
| Shape matrix | `mamba2_shape_matrix.py` | GPU/NPU 共用的 52-case 参数表与 suite |
| A100 forward | `mamba2_gpu_bench.py` | official `mamba_ssm` device-event latency |
| AscendC forward scaling | `mamba2_forward_training_bench.py` | public FWD 与 head scaling |
| Training FWD/BWD/memory | `mamba2_backward_bench.py` | backward-only、saved tensor、peak memory |
| Same-NPU implementations | `mamba2_npu_decomposition_bench.py` | PyTorch/Triton/AscendC |
| Forward/Backward profiler | `mamba2_{forward,backward}_npu_profile.py` | stage 与 engine 分解 |
| Vision-Mamba2 | `vmamba2_network_bench.py` | network/mixer/SSD 范围 |

正式源码、测试 Gate 和生成物边界见 [`../STRUCTURE.md`](../STRUCTURE.md)。

## 计时规则

跨平台比较必须满足：

1. shape、dtype、optional inputs 和随机种子一致；
2. kernel 编译、首次 autotune、输入构造和 H2D copy 不进入计时；
3. GPU/NPU 使用设备 Event，默认报告 p50；
4. backward-only 不包含 forward，并明确是否注入 `dfinal_state`；
5. 精度测试与性能循环分离；
6. 记录设备、框架、backend 版本、warmup、repeat 和 dispatch 路径。

不同 shape 的小任务固定开销不同，不能用单 kernel、单核或未达到 sustained
scaling 的结果替代 public API 端到端性能。

## Forward

Forward 不再只使用 tiny/medium/extreme 三个历史点。统一矩阵包含 52 个 case，
其中默认 `standard` suite 为 31 个代表点，分别覆盖：

- `generic`、`aligned`、`cube_mix` 三条 910B3 dispatch 路径；
- 非整除 sequence tail，以及 logical chunk `16/32/64/128/256/512`；
- sequence、batch、head、group、headdim 和 dstate 独立 scaling；
- 既有 heavy H128/H256 sustained-throughput gate。

先查看矩阵，不分配设备：

```bash
python benchmarks/mamba2_shape_matrix.py --suite standard --format markdown
```

相同 suite 名称由 A100 和 AscendC runner 共用，避免两端 shape 漂移：

```bash
# NVIDIA GPU：官方 mamba_ssm
python benchmarks/mamba2_gpu_bench.py \
  --suite standard --warmup 30 --repeat 200 \
  --output benchmarks/results/mamba2_forward_gpu.jsonl

# Ascend NPU：Triton-Ascend
python benchmarks/mamba2_triton_ascend_bench.py \
  --cases medium extreme --warmup 30 --repeat 200 \
  --output benchmarks/results/mamba2_forward_triton_ascend.jsonl

# Ascend NPU：AscendC
python benchmarks/mamba2_npu_final_bench.py \
  --suite standard --warmup 30 --repeat 200 \
  --output benchmarks/results/mamba2_forward_ascendc.jsonl
```

AscendC runner 默认在计时前用 NPU PyTorch reference 检查 output 与 final state。
只有已通过独立精度回归、且 reference 会显著抬高 stress case 峰值内存时，才允许
显式加 `--skip-precision`。完整 sweep 可分别使用 `--suite dispatch|sequence|batch|heads|groups|inner|stress|all`；`--cases` 仍可选择单点。

### A100 80GB shape matrix result

2026-08-12 在 NVIDIA A100 80GB PCIe、`mamba_ssm 2.2.6.post3` 上运行完整
`all` suite，52/52 case 成功。Public inputs 为 FP32，启用
`D/z/dt_bias/softplus/initial_state/final_state`，CUDA Event，`warmup=5`、
`repeat=30`、报告 median。以下只展示单变量 scaling；完整结构化快照在
`readme_benchmarks.json`。

| Sequence length `L` | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---:|---:|---:|---:|---:|---:|---:|
| A100 median | 0.827 ms | 0.949 ms | 1.879 ms | 3.758 ms | 7.485 ms | 15.041 ms |

固定 `[B,H,P,N,chunk,G]=[8,32,64,128,128,8]`。从 `L=2048` 开始，latency
随 sequence 基本线性增长；`L=512/1024` 仍有明显的约 0.8 ms 固定开销平台。

| Batch `B` | 1 | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|---:|
| A100 median | 0.835 ms | 0.832 ms | 0.838 ms | 0.966 ms | 1.910 ms |

固定 `[L,H,P,N,chunk,G]=[2048,16,64,128,128,4]`。`B<=4` 尚未充分 stress
GPU；到 `B=8/16` 才进入吞吐 scaling。

| Heads `H` | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|
| A100 median | 0.823 ms | 0.823 ms | 0.826 ms | 0.904 ms | 1.713 ms |

固定 `[B,L,P,N,chunk,G]=[4,2048,64,128,128,4]`。H128/H256 heavy stress
分别为 `12.376 ms` 和 `24.796 ms`，证明在足够大的 workload 上 A100 latency
也近似随任务量线性增长，不能用小 shape 的固定开销平台推断 sustained throughput。

另外抽取 generic、tail、aligned、N64/N128 Cube-like 和 logical chunk 256 共
6 个 FP32 case 与 CPU PyTorch reference 对拍：6/6 通过，worst output NRMSE
`7.91e-4`，worst final-state NRMSE `7.81e-4`，全部 finite。大规格官方 Triton
kernel 的少量近零元素会超过旧的逐元素 `rtol=1e-2, atol=3e-3`，因此新增矩阵
门禁同时限制 NRMSE 和 max-absolute error；原有小 shape 严格 assert-close 测试不变。

训练 forward heavy sweep 固定 `B=8,L=4096,P=N=chunk=64,H/G=4`：

```bash
python benchmarks/mamba2_forward_training_bench.py \
  --backend ascendc --device npu \
  --cases m1_extreme m1_stress_h64 m1_stress_h128 m1_stress_h256 \
  --feature full --warmup 10 --repeat 30 \
  --output benchmarks/results/mamba2_fwd_head_scaling.jsonl
```

## Backward

`mamba2_backward_bench.py` 输出 forward、backward-only、forward+backward、
saved-tensor bytes、额外 peak memory 和逐梯度 checksum。

```bash
# CPU reference：只运行小 shape
python benchmarks/mamba2_backward_bench.py \
  --backend reference --device cpu \
  --cases cpu_tiny cpu_small --feature full \
  --warmup 1 --repeat 3

# A100 official mamba_ssm：output gradient only
python benchmarks/mamba2_backward_bench.py \
  --backend mamba_ssm --device cuda \
  --cases m1_extreme m1_stress_h64 m1_stress_h128 m1_stress_h256 \
  --feature full --warmup 10 --repeat 30 \
  --output benchmarks/results/mamba2_bwd_a100_output_only.jsonl

# 相同 shape 的 AscendC public autograd
python benchmarks/mamba2_backward_bench.py \
  --backend ascendc --device npu \
  --cases m1_extreme m1_stress_h64 m1_stress_h128 m1_stress_h256 \
  --feature full --warmup 10 --repeat 30 \
  --output benchmarks/results/mamba2_bwd_ascendc_output_only.jsonl
```

需要测试 output + final-state gradient 时，两端都追加 `--final-state-grad`，不得
把该结果与 output-only 数据混合。

## Same-NPU implementation comparison

```bash
python benchmarks/mamba2_npu_decomposition_bench.py \
  --cases medium --backends pytorch triton-ascend ascendc \
  --warmup 30 --repeat 100 \
  --output benchmarks/results/framework_decomposition.json
```

该实验解释 PyTorch composition、Triton-Ascend 与 AscendC 在同一 NPU 上的
kernel decomposition 和端到端差异，不把编程语言本身解释为固定加速倍数。

## External `ops-transformer` comparison

[`cann/ops-transformer/experimental/mamba`](https://gitcode.com/cann/ops-transformer/tree/master/experimental/mamba)
提供面向 Nemotron-H 的 AscendC Mamba-2 forward 子算子。其公开 910B3 tensor
规格为 `[B,L,H,P,N,G]=[1,1024,128,64,128,8]`，算法参数
`chunk_size=256`，核心 SSD 被拆成四段：

| Stage | PyTorch | AscendC | Self speedup |
|---|---:|---:|---:|
| `chunk_cumsum` | `336 us` | `74 us` | `4.54x` |
| `chunk_state` | `352 us` | `97 us` | `3.63x` |
| `chunk_state_passing` | `762 us` | `104 us` | `7.33x` |
| `chunk_scan` | `1961 us` | `297 us` | `6.60x` |
| Sum of independently timed stages | `3.411 ms` | `0.572 ms` | `5.96x` |

两套实现的工程边界不同：

| Dimension | This project | `ops-transformer/experimental/mamba` |
|---|---|---|
| Public execution | one Mamba-2 SSD public call | four public suboperator calls |
| Published 910B3 shapes | scaling suite over `B/L/H` plus smaller correctness cases | one fixed inner shape, `[1,1024,128,64,128,8]` |
| Forward features | `D/z/dt_bias/softplus/initial/final-state` | four-stage chain covers `D/dt_bias/softplus/initial/final-state`; no published `z` chain |
| Backward | public autograd, M1 native-core path | not present in the experimental Mamba directory |
| Main optimization scope | integrated public operator, broader dispatch and training path | aggressively specialized four-stage FP16/MIX forward |

这些数字是上游 README 的独立子算子报告值，不是本仓库复跑的端到端 chain，
也不与本项目不同 shape 的公开结果并排计算比值。`chunk_size` 不是 tensor shape；
它是 SSD 的 logical partition 参数。本项目现将大 logical chunk 映射为多个
64/128-token Cube/MIX micro-tile，例如 public `chunk_size=256` 执行为 `2x128`，
而不是构造一个超出 UB 预算的 T256 tile。

直接比较固定以下控制变量：910B3、同一 CANN/PyTorch 环境、同一输入值和 public
dtype、`P=64`、`N=128`、`chunk=256`、`G=8`、`H=128`，同时启用双方共同具备的
`D/dt_bias/softplus/initial_state/final_state`，关闭 `z`。Scaling 只改变 `L` 或
`B`：

| Case | Tensor shape `[B,L,H,P,N,G]` | Logical `chunk_size` | Scaling axis |
|---|---|---:|---|
| S1 | `[1,256,128,64,128,8]` | `256` | 1 logical chunk |
| S2 | `[1,512,128,64,128,8]` | `256` | sequence |
| S3 | `[1,1024,128,64,128,8]` | `256` | upstream published shape |
| S4 | `[1,2048,128,64,128,8]` | `256` | sequence stress |
| S5 | `[2,2048,128,64,128,8]` | `256` | batch stress |

每个 case 必须同时报告完整 chain device latency、四阶段 breakdown、峰值显存和
精度；不允许把官方四段独立计时之和与本项目 public-call latency 直接相除。

本项目对应的可复现入口为：

```bash
python benchmarks/mamba2_ops_transformer_shape_bench.py \
  --cases s1 s2 s3 s4 s5 --warmup 5 --repeat 20 \
  --time-reference --output benchmarks/results/ops_transformer_matched.json
```

本项目在 Ascend 910B3 上的 source-direct device-event 结果如下，`warmup=5`、
`repeat=20`。PyTorch reference 与 AscendC 使用同一组 NPU FP32 输入；官方列仅在
上游实际公开的 S3 shape 填值，不对其它尺寸做外推：

| Case | Tensor shape `[B,L,H,P,N,G]` | Logical chunk | Micro-tile | This project | NPU PyTorch | Self speedup | Official published |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 | `[1,256,128,64,128,8]` | `256` | `128` | `0.620 ms` | `12.046 ms` | `19.43x` | — |
| S2 | `[1,512,128,64,128,8]` | `256` | `128` | `0.630 ms` | `13.206 ms` | `20.97x` | — |
| S3 | `[1,1024,128,64,128,8]` | `256` | `128` | `0.821 ms` | `13.649 ms` | `16.62x` | `0.572 ms` |
| S4 | `[1,2048,128,64,128,8]` | `256` | `128` | `1.690 ms` | `12.841 ms` | `7.60x` | — |
| S5 | `[2,2048,128,64,128,8]` | `256` | `128` | `3.745 ms` | `18.962 ms` | `5.06x` | — |

五个 case 均 finite；worst output NRMSE 为 `2.707e-4`，worst final-state NRMSE
为 `3.825e-4`。S3 现在具有相同 tensor shape 和相同 logical chunk：原始数字中
本项目 latency 比官方公开值高 `43.6%`。这说明官方固定规格实现更快，但该差值
仍不是最终严格 speedup，因为官方 `0.572 ms` 是 FP16/mixed-input 四阶段独立计时
之和，本项目 `0.821 ms` 是 FP32 public call 的完整 device chain。其余 scaling
点必须等上游在同环境成功构建后实测。

2026-08-11 在 910B3/CANN 8.2.RC1 上对上游 commit
`4c8dbb11343357bed5f2df5cbb0b3216551dd691` 做了源码复跑。上游使用 CANN 9
参数 `--npu-arch=dav-2201 -xasc`，默认构建白名单也未启用 Mamba。兼容实验只修改
构建、fat-object packaging 和 Torch host binding，未修改四个 device kernel 或
tiling：纯 Vector `chunk_cumsum` 可以执行；但用 CANN 8.2
`-cce-enable-mix` 生成的 `chunk_state` 在上游公布的 S3 shape 上仍报
`507015 AICore exception`（D-cache/UB bus response），另一轮表现为 `507014
AICore timeout`。保留上游 `RunOpApiV2` launch context 后问题仍可复现。

因此 CANN 8.2 产物只是“编译/链接成功”，不能视为与 CANN 9 MIX codegen 等价；
没有产生可通过精度门禁的完整 chain latency。上表仍只把 `0.572 ms` 标记为上游
公开的四阶段独立计时之和，不把失败运行或不完整的 `chunk_cumsum` 计时用于排名。

完成直接对比需要：在 CANN 9 的 910B3 环境源码构建上游四个算子；用一个 wrapper
串起完整 forward；统一输入、输出、optional feature、warmup/repeat 与设备事件；
最后分别报告 chain latency 和四段 profiler breakdown。本项目 backward 暂无对应
上游实现，不能纳入该 forward 对比。

完整 chain harness 已保留为 `mamba2_ops_transformer_direct_bench.py`。它要求外部
目录提供四个带 raw-pointer C ABI 的 adapter library，并在任何 timing 之前以
PyTorch reference 执行 finite、NRMSE 和 cosine 精度门禁；当前 CANN 8.2 失败产物
不会输出 latency JSON。

## Profiling

Forward/Backward profiler 使用 `torch_npu.profiler`，推荐
`wait=0,warmup=5,active=5,repeat=1`：

```bash
python benchmarks/mamba2_forward_npu_profile.py \
  --cases m1_stress_h256 --warmup 5 --active 5 \
  --profile-root benchmarks/profiles/fwd_h256

python benchmarks/mamba2_backward_npu_profile.py \
  --cases m1_stress_h256 --warmup 5 --active 5 \
  --profile-root benchmarks/profiles/bwd_h256
```

Profiler 数据用于阶段分解、timeline 和 engine ratio；跨平台 latency 使用独立
device-event benchmark，避免 profiler overhead 污染结论。

## Source-direct AscendC development

日常 kernel 迭代直接加载当前源码 `.so + OPP`，不依赖已安装 wheel：

```bash
MAMBA2_TEST_EXTENSION_LIB=$PWD/mamba_ascendc/python/ascend_kernel/ascend_kernel/lib/libascend_kernel.so \
MAMBA2_TEST_OPP_ROOT=$PWD/mamba_ascendc_ops/mamba2_ssd_chunk_mix/build_out/packages \
python mamba_ascendc/tests/run_source_candidate.py \
  benchmarks/mamba2_backward_bench.py \
  --backend ascendc --device npu --cases m1_extreme \
  --feature full --warmup 10 --repeat 30
```

wheel 仅用于代码冻结后的安装和 bundled OPP 验收。

## Vision-Mamba2

```bash
python benchmarks/vmamba2_network_bench.py \
  --device npu --backend npu --config tiny \
  --image-size 1024 --batch 32 --merge-mode stream \
  --warmup 10 --repeat 30 \
  --output benchmarks/results/vmamba2_npu.json
```

GPU 端将 `--device/--backend` 改为 `cuda/cuda`。网络时间、Mamba mixer 时间和
SSD core 时间来自不同 instrumentation 范围，不能直接相加。

## README figures

公开 README 图表由单个去标识快照生成：

```bash
python benchmarks/plot_readme_figures.py
```

- 输入：`benchmarks/results/readme_benchmarks.json`
- 输出：`assets/*.svg`
- 当前公开图：跨平台 forward、msprof pipeline、head scaling、hardware
  utilization、三实现对比、backward breakdown、data movement 和
  Vision-Mamba2 integration。

`readme_benchmarks.json` 是唯一提交到公开仓库的汇总快照。原始 JSONL、候选
结果和 profiler trace 默认由 `.gitignore` 排除；只有经过同口径复核的最终数字
才能进入该快照。当前仍应补充的核心证据是 backward heavy scaling、训练
saved-tensor/peak-memory scaling，以及回归修复后的当前源码 950PR 结果。
