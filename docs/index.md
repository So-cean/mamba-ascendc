---
title: "Mamba2 Selective Scan AscendC Operator"
description: "Mamba2 (Mamba-2) selective scan and SSD forward/backward for Huawei Ascend 910B3 and 950PR, implemented as a native AscendC custom operator."
permalink: /
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "mamba-ascendc",
  "alternateName": ["Mamba2 Selective Scan AscendC", "Mamba-2 SSD for Ascend NPU"],
  "description": "Native AscendC custom operator for Mamba2 selective scan and Structured State Space Duality forward/backward on Huawei Ascend 910B3 and 950PR.",
  "codeRepository": "https://github.com/So-cean/mamba-ascendc",
  "programmingLanguage": ["AscendC", "C++", "Python", "Triton"],
  "runtimePlatform": ["Huawei Ascend 910B3", "Huawei Ascend 950PR", "CANN", "torch_npu"],
  "license": "https://www.apache.org/licenses/LICENSE-2.0",
  "version": "0.1.1",
  "author": {"@type": "Person", "name": "So-cean", "url": "https://github.com/So-cean"}
}
</script>

# Mamba2 / Mamba-2 Selective Scan for Huawei Ascend NPU

`mamba-ascendc` implements the Mamba2 Structured State Space Duality (SSD)
selective-scan core as a native AscendC custom operator. It provides forward
and public-autograd backward for PyTorch on Huawei Ascend 910B3 and Ascend
950PR, with CANN and `torch_npu` integration.

The repository includes a PyTorch mathematical reference, a Triton-Ascend
comparison, same-shape NVIDIA A100 80GB `mamba_ssm` baselines, accuracy tests,
device-event benchmarks, msprof profiling, and Vision-Mamba2 integration.

## Implementation

| Path | Role |
|---|---|
| AscendC | Main NPU forward/backward custom operator |
| PyTorch | Numerical oracle and fallback |
| Triton-Ascend | Same-NPU DSL comparison |
| A100 `mamba_ssm` | Cross-platform reference |

Public tensor semantics follow
`mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`. The current API
supports `D`, `z`, `dt_bias`, `dt_softplus`, `dt_limit`, `initial_states`, and
`return_final_state`.

## Benchmark coverage

GPU and NPU forward runners share one 52-case shape matrix. The default
31-case suite covers generic, aligned, and Cube/MIX execution; non-divisible
sequence tails; logical chunk sizes from 16 to 512; and sequence, batch, head,
group, head-dimension, and state-dimension scaling. Timing uses device events,
and AscendC accuracy is checked against the PyTorch reference before normal
performance timing.

- [Read the full README](https://github.com/So-cean/mamba-ascendc#readme)
- [Install and use the operator](https://github.com/So-cean/mamba-ascendc#installation--安装)
- [Benchmark methodology](https://github.com/So-cean/mamba-ascendc/blob/main/benchmarks/README.md)
- [Accuracy and validation map](https://github.com/So-cean/mamba-ascendc/blob/main/tests/README.md)
- [Profiling report](https://github.com/So-cean/mamba-ascendc/blob/main/docs/profiling/model_architecture_report_profile_fwd_h256_910b3_final_20260810.md)
- [Source code](https://github.com/So-cean/mamba-ascendc)

## Frequently asked questions

### Does Mamba2 run on Huawei Ascend?

Yes. This project implements the Mamba2 SSD/selective-scan core as an AscendC
operator and has test paths for Ascend 910B3 and 950PR.

### Is this a Triton port?

No. AscendC is the main implementation. Triton-Ascend remains in the repository
as a comparison path.

### Does it include backward?

Yes. The public PyTorch API exposes autograd backward. The fastest native
backward path currently has narrower shape constraints than forward.

### How is it compared with A100?

The repository uses identical Mamba2 workload tuples on both platforms and
device-event latency. It does not normalize results using incomparable vendor
TFLOPS figures.
