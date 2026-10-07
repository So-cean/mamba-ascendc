# Performance analysis / 性能分析

[Project overview](../README.md) · [Installation](../README.md#installation--安装) · [Benchmark commands](../benchmarks/README.md)

This page collects the existing benchmark results, precision context and profiler
analysis. Commands assume the repository root as the working directory. Results
come from the published benchmark snapshot; this documentation update does not
introduce new measurements.

## Measurement protocol / 测量口径

Unless noted otherwise, every cross-platform result uses identical shapes,
FP32 public tensors, optional inputs, device events, `warmup=10`, `repeat=30`,
and p50 latency. The A100 runs `mamba_ssm 2.2.6.post3`; Ascend results load the
source-built extension and custom OPP. Theoretical TFLOPS are intentionally not
used to normalize different architectures.

除非单独说明，性能结果使用相同 shape、FP32 public tensors、完整 optional
inputs、设备 Event、`warmup=10`、`repeat=30` 和 p50。A100 运行
`mamba_ssm 2.2.6.post3`；910B3 直接加载当前源码 `.so + OPP`，不经过已安装
wheel。理论 TFLOPS 未用于归一化，因为不同厂商、数据类型和统计口径不能直接
等价比较。

Forward 的 GPU/NPU runner 共享同一份 52-case shape matrix；默认 `standard`
suite 取 31 个代表点，覆盖 generic/aligned/Cube-MIX 路径、非整除 tail、
`L/B/H/G/P/N/chunk_size` scaling 和 heavy H128/H256。矩阵可直接查看：

```bash
python benchmarks/mamba2_shape_matrix.py --suite standard --format markdown
```

完整 case、计时规范与 suite 命令见
[`benchmarks/README.md`](../benchmarks/README.md#forward)。

最新 A100 80GB inference matrix 已完成 52/52 case。固定
`[B,H,P,N,chunk,G]=[8,32,64,128,128,8]` 时，sequence scaling 为：

| `L` | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---:|---:|---:|---:|---:|---:|---:|
| A100 80GB | 0.827 ms | 0.949 ms | 1.879 ms | 3.758 ms | 7.485 ms | 15.041 ms |

这说明 A100 在小 shape 上主要处于约 0.8 ms 的固定开销平台；从 `L=2048`
开始才呈现清晰的 sustained linear scaling。六类新增 shape 的 FP32 reference
精度 smoke 为 6/6 通过，worst output/final-state NRMSE 分别为 `7.91e-4` 和
`7.81e-4`。完整 batch/head/heavy 数据与测试口径见 benchmark 文档。

### A100 80GB 与 Ascend 910B3

![H256 A100 and Ascend 910B3 training gate](../assets/h256-training-gate.svg)

主门禁固定 heavy training forward shape
`[B,L,H,P,N,chunk,G]=[8,4096,256,64,64,64,64]`，并返回 final state。

| 平台 | 实现 | Forward | 相对 A100 吞吐 |
|---|---|---:|---:|
| A100 80GB PCIe | `mamba_ssm` | `22.583 ms` | `100%` |
| Ascend 910B3 | AscendC | `31.888 ms` | `70.82%` |

图中第一部分给出绝对 latency，第二部分直接给出相同 workload 下的相对吞吐，
不使用 A100/910B3 的理论算力数字解释结果。当前 forward 已达到 70% 门禁；
75% stretch 目标为 `≤30.111 ms`。

For the sustained H256 training workload, Ascend 910B3 reaches `70.82%` of the
A100 throughput without theoretical-compute normalization.

### Forward pipeline 与 msprof timeline

![AscendC pipeline and measured msprof timeline](../assets/ascendc-pipeline.svg)

上半部分来自 H256 forward 的真实 `kernel_details.csv`，下半部分对应实现中的
数据流。五个 active step 的统计如下：

| 指标 | 结果 |
|---|---:|
| profiler service time | `33.9664 ms/step` |
| device busy union | `33.9633 ms/step` |
| kernel 间空洞 | `0.00313 ms/step` |
| underfeed ratio | `0.00923%` |
| 最大 kernel 间隙 | `0.0015 ms` |
| 正式 device-event p50 | `31.8882 ms` |

因此这个 shape 不是 launch/host underfeed 主导：四个设备阶段几乎无缝衔接。
profiler 自身使 wall time 比 device-event p50 高约 6.5%，所以 profiler 只用于
阶段分解，跨平台表使用 device Event。

The measured device timeline is continuous: host/kernel underfeed is only
`0.00923%`, so the remaining forward gap is inside device work rather than
launch bubbles.

完整证据见
[`H256 profiling architecture report`](../docs/profiling/model_architecture_report_profile_fwd_h256_910b3_final_20260810.md)
和
[`anomaly JSON`](../docs/profiling/mamba2_h256_fwd_anomaly_20260810.json)。

### Heavy-shape scaling

![Forward head scaling](../assets/head-scaling.svg)

固定 `B=8,L=4096,P=N=chunk=64,H/G=4`，将 head 从 32 扩展到 256。图中
左图是绝对 latency，右图为 `A100 latency / 910B3 latency`，即 910B3 相对
A100 的持续吞吐。

| H / G | A100 FWD | 910B3 FWD | 910B3/A100 吞吐 |
|---:|---:|---:|---:|
| 32 / 8 | `3.080 ms` | `4.619 ms` | `66.68%` |
| 64 / 16 | `5.805 ms` | `8.631 ms` | `67.26%` |
| 128 / 32 | `11.379 ms` | `16.793 ms` | `67.76%` |
| 256 / 64 | `22.583 ms` | `31.888 ms` | `70.82%` |

两端进入 sustained scaling 后 latency 均近似随 head 线性增加。H256 的
Forward 使用单 case 隔离复测值，避免多 case 串行 sweep 的显存/cache 状态影响。

Both devices enter near-linear sustained scaling from H32 to H256; the Ascend
throughput ratio improves from `66.68%` to `70.82%` as fixed overhead is
amortized.

### Hardware utilization

![AscendC hardware utilization](../assets/hardware-utilization.svg)

| Stage | Mean latency | Cube active | AIC MAC | AIV MTE2 | 结论 |
|---|---:|---:|---:|---:|---|
| Preprocess | `6.667 ms` | - | - | `26.5%` | Vector/control 与输入布局 |
| ChunkMix | `14.850 ms` | `99.1%` | `16.1%` | `27.8%` | AIC 持续工作，但有效 MAC 仍低 |
| StateEpilogue | `12.303 ms` | `98.3%` | `2.9%` | `76.5%` | 数据搬运与低密度 projection 主导 |

`Cube active` 表示 Cube pipeline 被持续调度，不等于每个周期都在执行有效 MMAD。
不同 engine 的 activity 可以重叠，不能相加为 100%。当前最明确的 forward
优化方向是 StateEpilogue 的 consumer layout、GM→UB 搬运和小矩阵映射，而不是
继续减少已经只有微秒级的 kernel gap。

High Cube activity does not imply high useful MAC density. StateEpilogue is the
main architectural bottleneck, with `98.3%` Cube active but only `2.9%` AIC MAC
and `76.5%` AIV MTE2 activity.

### PyTorch / Triton-Ascend / AscendC

![PyTorch, Triton-Ascend and AscendC comparison](../assets/framework-decomposition.svg)

三种实现运行在同一张 910B3 上，shape 为
`[4,2048,16,64,128,128,4]`，并启用 `D/z/dt_bias/softplus`。

| NPU 实现 | p50 latency | Kernels / forward | 相对 PyTorch | 相对 Triton |
|---|---:|---:|---:|---:|
| PyTorch composition | `6.287 ms` | `335` | `1.00×` | `0.68×` |
| Triton-Ascend | `4.305 ms` | `20` | `1.46×` | `1.00×` |
| AscendC | `0.632 ms` | `3` | `9.94×` | `6.81×` |

这是同平台实现层级对比，不表示“AscendC 语言固定比 Triton 快 6.81×”。差异来自
算子融合、kernel 数量、中间 tensor 物化、Cube/Vector 映射和数据布局的共同变化。

On the same 910B3 workload, the final AscendC path reduces the forward from 335
PyTorch kernels or 20 Triton-Ascend kernels to 3 device kernels, reaching
`9.94×` over eager PyTorch and `6.81×` over the current Triton-Ascend path.

### Backward breakdown

![AscendC backward breakdown](../assets/backward-breakdown.svg)

H256 backward 的 profiler kernel mean 为 `82.454 ms`，与 device-event p50
`82.596 ms` 一致。主要阶段为：

| Stage | Mean latency | BWD 占比 |
|---|---:|---:|
| BatchMatMul | `23.053 ms` | `28.0%` |
| DtBwd | `14.723 ms` | `17.9%` |
| DiagFinalize | `10.841 ms` | `13.1%` |
| Gate | `9.096 ms` | `11.0%` |
| Off path | `8.111 ms` | `9.8%` |
| StatePassing | `7.229 ms` | `8.8%` |
| Prepare | `7.242 ms` | `8.8%` |

最大单项是 Cube BMM，但优化不能只看 Cube utilization：BMM 输出仍需落 GM 并由
Vector consumer 读回。下一轮应优先共设计 BMM 输出与 Vector consumer layout，
其次继续做 Gate/Dt 的 head-block contiguous tiling 和受控 AIC/AIV pipeline。

Backward is currently dominated by Cube BMM (`28.0%`), DtBwd (`17.9%`), and
DiagFinalize (`13.1%`). The next optimization must co-design Cube output layout
with Vector consumers instead of optimizing either engine in isolation.

### Cross-kernel data movement

![AscendC logical intermediate movement](../assets/data-movement.svg)

H256 forward 的 shape-derived logical lower bound 为 `12.06 GiB` 中间结果写回+
重读，其中 `y_diag` 与 `chunk_state` 合计 `8 GiB`（`66.3%`）。这是根据 tensor
shape 计算的跨 kernel 物化量，不是 msprof HBM bandwidth counter。它解释了为什么
head 增大后 latency 线性增长，也说明简单把更多计算交给 Cube 并不能消除
producer/consumer layout 不匹配。

The H256 forward materializes a shape-derived lower bound of `12.06 GiB` across
kernel boundaries; `y_diag` and `chunk_state` account for `66.3%` of it.

### Vision-Mamba2 integration

![Pure Vision-Mamba2 end-to-end](../assets/vmamba2-end-to-end.svg)

纯 Vision-Mamba2 tiny 网络使用 12 个 Mamba block、4 个扫描方向，每次 forward
调用 48 次 SSD。batch 32、`1024×1024` 图像的结果：

| 范围 | A100 80GB | Ascend 910B3 | 910B3/A100 吞吐 |
|---|---:|---:|---:|
| Full network | `1430.8 ms` | `1856.6 ms` | `0.771×` |
| Mamba mixers | `1081.4 ms` | `1611.6 ms` | `0.671×` |
| SSD cores | `455.9 ms` | `699.7 ms` | `0.651×` |

跨设备 logits 对拍为 MaxAbs `3.76e-4`、NRMSE `3.42e-4`、cosine
`1.00000048`。Full network 单独计时；Mixer/SSD 来自 instrumented forward，
组件时间不可直接相加。

The pure 12-block Vision-Mamba2 network validates real model integration at
batch 32 and `1024×1024`: full-network Ascend throughput is `0.771×` A100, with
cross-device logits NRMSE `3.42e-4`.

机器可读的 README 数据快照位于
[`benchmarks/results/readme_benchmarks.json`](../benchmarks/results/readme_benchmarks.json)，
图表由 [`benchmarks/plot_readme_figures.py`](../benchmarks/plot_readme_figures.py)
生成。
