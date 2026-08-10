# Contributing

Contributions should keep the public Mamba-2 tensor semantics aligned across the
PyTorch reference, Triton-Ascend, and AscendC implementations.

## Public checks

The GitHub Actions workflow covers checks that can run without accelerator
hardware:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
pytest -q tests/mamba2/test_reference.py tests/mamba2/test_vmamba2_network.py
git ls-files '*.py' -z | xargs -0 python -m py_compile
git ls-files '*.py' -z | xargs -0 ruff check --force-exclude
python benchmarks/plot_readme_figures.py
git diff --exit-code -- assets/
```

## Ascend changes

AscendC and Triton-Ascend changes require dedicated NPU validation before review:

1. Build and load the source candidate rather than an older installed wheel.
2. Compare forward or backward outputs with `mamba_torch` using the same public
   FP32 inputs and optional features.
3. Cover at least one small correctness case and one heavy target shape.
4. For performance changes, report device events with warmup, repeat count, p50,
   the exact baseline, and an msprof or `torch_npu.profiler` breakdown.
5. Run mssanitizer when changing tiling, buffer sizes, offsets, or data movement.

The source-candidate build and test commands are documented in the root README.

## Benchmark reporting

Every comparison must record:

- device and software versions;
- `[B, L, H, P, N, chunk_size, G]` and dtype;
- enabled `D`, `z`, `dt_bias`, softplus, initial/final-state, and backward options;
- synchronization and timing method;
- warmup, repeat count, and statistic;
- implementation and version used as the baseline.

Do not compare results that use different semantics or report theoretical TFLOPS
as a direct cross-architecture normalization.

## Public repository hygiene

Do not commit credentials, private filesystem paths, cluster node names, scheduler
job IDs, complete raw profiler archives, generated OPP packages, wheels, or build
trees. Keep final reproducible benchmark data in
`benchmarks/results/readme_benchmarks.json` and regenerate figures with
`benchmarks/plot_readme_figures.py`.

Ruff and clang-format are enforced incrementally for project-owned files changed
by a pull request. Upstream Mamba sources, generated CANN/msOpGen scaffold, and
generated benchmark artifacts are excluded from formatting enforcement.
