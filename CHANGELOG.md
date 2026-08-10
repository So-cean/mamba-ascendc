# Changelog

All notable public changes are recorded here. Versions follow Semantic
Versioning for the Python API; kernel coverage remains subject to the support
matrix documented in the README.

## [0.1.0] - 2026-08-10

Initial public release.

### Added

- Mamba-2 SSD AscendC forward with `D`, `z`, `dt_bias`, `dt_softplus`,
  `dt_limit`, initial state, and final-state output.
- M0 correctness backward and optimized M1 public-autograd backward for
  contiguous FP32 `P=N=chunk_size=64`.
- PyTorch reference, Triton-Ascend forward comparison, and A100 `mamba_ssm`
  benchmark adapters.
- Pure Vision-Mamba2 integration and cross-device logits validation.
- Reproducible benchmark snapshot, README figures, and msprof architecture
  reports.
- Source-built wheel containing the Python API, PyTorch extension, custom OPP,
  `libcust_opapi.so`, and runtime license notices.
- CPU CI, issue templates, contribution guide, incremental Ruff/clang-format
  checks, and protected `main` branch.

### Validated environments

- Ascend 910B3: CANN 8.2.RC1, Python 3.11, PyTorch/torch_npu 2.6.0,
  Triton-Ascend 3.2.0.
- Ascend 950PR: CANN 9.0.0, Python 3.11, PyTorch 2.7.1,
  torch_npu 2.7.1.post4, Triton-Ascend 3.2.1.
- NVIDIA A100 80GB: official `mamba_ssm 2.2.6.post3` comparison baseline.

### Known limitations

- Public tensors are FP32; Cube operands use FP16 with FP32 accumulation.
- Optimized M1 backward does not yet support varlen/packed sequences or generic
  `P`, `N`, and chunk sizes.
- Prebuilt binary wheels are not published because OPP binaries are tied to the
  target SoC and CANN toolchain. Build the wheel in the target Ascend
  environment with `scripts/build_mamba_ascendc_wheel.sh`.

[0.1.0]: https://github.com/So-cean/mamba-ascendc/releases/tag/v0.1.0
