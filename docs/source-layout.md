# Source layout and ownership

This is the short source-ownership reference. The comprehensive development,
test, benchmark, generated-artifact, and handoff guide is
[`../STRUCTURE.md`](../STRUCTURE.md).

The repository builds one public `ascend_kernel` Python package from two
Ascend projects. They are separate build domains rather than competing
implementations.

| Path | Source of truth | Output |
|---|---|---|
| `mamba_torch/` | Mathematical semantics and CPU oracle | PyTorch reference |
| `mamba_triton_ascend/` | Same-NPU DSL comparison only | Triton-Ascend forward |
| `mamba_ascendc/` | Public API, autograd, dispatch, AIV/fallback kernels, ACLNN bridges | `libascend_kernel.so` and Python package |
| `mamba_ascendc_ops/` | AIC Cube/MIX kernels and custom-operator registration | custom OPP and `libcust_opapi.so` |

`scripts/build_mamba_ascendc_wheel.sh` is the release boundary: it builds the
custom OPP first, builds the PyTorch extension second, and packages both into
one platform wheel. Users do not install the two projects independently.

## Editable implementation files

- Public function signatures, validation, dispatch, and autograd:
  `mamba_ascendc/python/ascend_kernel/ascend_kernel/mamba2.py`
- PyTorch custom-op registration and ACLNN bridges:
  `mamba_ascendc/csrc/register.cpp` and `mamba_ascendc/csrc/aclnn/`
- AIV/fallback kernels:
  `mamba_ascendc/csrc/ops/<op>/op_host/` and `op_kernel/`
- AIC Cube/MIX kernels:
  `mamba_ascendc_ops/mamba2_ssd_chunk_mix/op_host/` and `op_kernel/`
- Mathematical reference:
  `mamba_torch/ssd_reference.py`

The files under `mamba_ascendc_ops/mamba2_ssd_chunk_mix/cmake/` and
`scripts/` are CANN/msOpGen build scaffold. Modify them only for build-system
compatibility, not for operator algorithms.

## Change routing

1. Change mathematical behavior in `mamba_torch` and add a CPU test first.
2. Change the public Ascend API or backward contract in `mamba_ascendc`.
3. Put Vector/fallback work in `mamba_ascendc/csrc/ops`.
4. Put Cube/MIX work in `mamba_ascendc_ops` and expose it through an ACLNN
   bridge in `mamba_ascendc/csrc/aclnn`.
5. Keep Triton-Ascend changes isolated to the comparison implementation.

## 中文说明

`mamba_ascendc` 与 `mamba_ascendc_ops` 不是两个可替换版本。前者负责 public
API、autograd、Vector/fallback kernel 和 PyTorch 扩展；后者负责 Cube/MIX
custom OPP。发布脚本将二者合并为一个 `mamba-ascendc` wheel。算法修改应进入
上述 editable implementation files，不应写入 msOpGen 生成的 CMake scaffold。
