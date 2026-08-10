# Benchmarks

本目录提供 Mamba-2 SSD 的 PyTorch reference、A100 `mamba_ssm`、
Triton-Ascend 和 AscendC benchmark。公开结果统一使用
`[B,L,H,P,N,chunk,G]` shape 顺序和 FP32 public tensors。

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

```bash
# NVIDIA GPU：官方 mamba_ssm
python benchmarks/mamba2_gpu_bench.py \
  --cases medium extreme --warmup 30 --repeat 200 \
  --output benchmarks/results/mamba2_forward_gpu.jsonl

# Ascend NPU：Triton-Ascend
python benchmarks/mamba2_triton_ascend_bench.py \
  --cases medium extreme --warmup 30 --repeat 200 \
  --output benchmarks/results/mamba2_forward_triton_ascend.jsonl

# Ascend NPU：AscendC
python benchmarks/mamba2_npu_final_bench.py \
  --cases medium extreme --warmup 30 --repeat 200 --skip-precision \
  --output benchmarks/results/mamba2_forward_ascendc.jsonl
```

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
