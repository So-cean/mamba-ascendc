"""Mamba-2 SSD training benchmark for CPU, GPU mamba_ssm and AscendC.

The script reports training forward, backward-only, forward+backward, saved
tensors and the extra CUDA memory allocated beyond inputs/upstream gradients.
All backends reuse the same case table and JSON schema.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import torch


_ASCEND_API = None
_ASCEND_OP_API_HANDLE = None


def load_ascend_backend():
    """Load the packaged backend or a source-tree candidate extension.

    ``MAMBA2_TEST_EXTENSION_LIB`` is intentionally a development-only path:
    it keeps the installed package's (or an explicitly selected) custom OPP,
    loads the candidate shared library directly, and executes the Python API
    from this checkout.  No candidate wheel is built or installed.
    """
    global _ASCEND_API, _ASCEND_OP_API_HANDLE
    if _ASCEND_API is not None:
        return _ASCEND_API

    candidate_library = os.environ.get("MAMBA2_TEST_EXTENSION_LIB")
    if not candidate_library:
        import ascend_kernel

        _ASCEND_API = ascend_kernel
        return _ASCEND_API

    explicit_opp_root = os.environ.get("MAMBA2_TEST_OPP_ROOT")
    if explicit_opp_root:
        vendor_root = Path(explicit_opp_root) / "vendors" / "customize"
    else:
        package_spec = importlib.util.find_spec("ascend_kernel")
        if package_spec is None or package_spec.origin is None:
            raise RuntimeError(
                "an installed ascend_kernel package is required to locate the "
                "custom OPP; set MAMBA2_TEST_OPP_ROOT to override it"
            )
        package_root = Path(package_spec.origin).parent
        vendor_root = package_root / "opp" / "vendors" / "customize"
    op_api_library = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
    missing = [
        str(path)
        for path in (Path(candidate_library), vendor_root, op_api_library)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError("missing candidate runtime artifact(s): " + ", ".join(missing))

    current_opp = [
        entry for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
        if entry
    ]
    if str(vendor_root) not in current_opp:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
            [str(vendor_root), *current_opp]
        )
    os.environ["MAMBA_CHUNK_MIX_OP_API_LIB"] = str(op_api_library)
    os.environ.setdefault("MAMBA_ASCENDC_CHUNK_MIX", "1")
    os.environ.setdefault("MAMBA_ASCENDC_CHUNK128", "1")
    os.environ.setdefault("MAMBA_ASCENDC_NATIVE_PREPROCESS", "0")
    _ASCEND_OP_API_HANDLE = ctypes.CDLL(
        str(op_api_library), mode=ctypes.RTLD_GLOBAL
    )
    torch.ops.load_library(str(Path(candidate_library).resolve()))

    source = (
        Path(__file__).resolve().parents[1]
        / "mamba_ascendc"
        / "python"
        / "ascend_kernel"
        / "ascend_kernel"
        / "mamba2.py"
    )
    source_spec = importlib.util.spec_from_file_location(
        "mamba2_ascendc_source_candidate", source
    )
    if source_spec is None or source_spec.loader is None:
        raise RuntimeError(f"cannot load source backend from {source}")
    module = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(module)
    _ASCEND_API = module
    return _ASCEND_API


# Internal tuple order is B, L, H, P, N, G, C.  JSON also emits the canonical
# public order B, L, H, P, N, C, G explicitly.
CASES = {
    "cpu_tiny": (1, 16, 2, 8, 8, 1, 8),
    "cpu_small": (1, 32, 2, 8, 8, 1, 16),
    "tiny": (1, 128, 2, 64, 64, 1, 64),
    "small": (2, 512, 8, 64, 64, 1, 64),
    "medium": (4, 2048, 16, 64, 128, 4, 128),
    "extreme": (8, 4096, 32, 64, 128, 8, 128),
    "super_extreme": (8, 8192, 32, 64, 128, 8, 128),
    "ultra_extreme": (8, 16384, 32, 64, 128, 8, 128),
    "seq_1024": (8, 1024, 32, 64, 128, 8, 128),
    "seq_2048": (8, 2048, 32, 64, 128, 8, 128),
    "seq_4096": (8, 4096, 32, 64, 128, 8, 128),
    "seq_8192": (8, 8192, 32, 64, 128, 8, 128),
    "seq_16384": (8, 16384, 32, 64, 128, 8, 128),
    # AscendC M1 comparison cases.  Keep P=N=chunk=64 so these shapes are
    # executable by both the current native NPU path and official mamba_ssm.
    "m1_tiny": (1, 64, 1, 64, 64, 1, 64),
    "m1_small": (1, 128, 4, 64, 64, 2, 64),
    "m1_medium": (2, 1024, 8, 64, 64, 2, 64),
    "m1_large": (4, 2048, 16, 64, 64, 4, 64),
    "m1_extreme": (8, 4096, 32, 64, 64, 8, 64),
    # A100 saturation sweep.  Keep every algorithmic dimension fixed and
    # scale only B so latency/task reveals the transition from fixed launch
    # overhead to the sustained-throughput region.
    "m1_stress_b16": (16, 4096, 32, 64, 64, 8, 64),
    "m1_stress_b32": (32, 4096, 32, 64, 64, 8, 64),
    "m1_stress_b64": (64, 4096, 32, 64, 64, 8, 64),
    "m1_stress_b128": (128, 4096, 32, 64, 64, 8, 64),
    # Cross-platform stress sweep for the current AscendC implementation.
    # B=8 and K=L/C=64 keep B*K at the gate reduction limit (512), while
    # H/G scale together so heads_per_group remains 4 on every point.
    "m1_stress_h64": (8, 4096, 64, 64, 64, 16, 64),
    "m1_stress_h128": (8, 4096, 128, 64, 64, 32, 64),
    "m1_stress_h256": (8, 4096, 256, 64, 64, 64, 64),
    "m1_stress_h512": (8, 4096, 512, 64, 64, 128, 64),
}


@dataclass(frozen=True)
class FeatureConfig:
    use_d: bool
    use_z: bool
    use_dt_bias: bool
    use_initial_state: bool
    dt_softplus: bool
    dt_limit: tuple[float, float]


FEATURES = {
    "basic": FeatureConfig(False, False, False, False, False, (0.0, float("inf"))),
    # Keep softplus(dt + bias) inside the clamp interval for the main training
    # benchmark so ddt and ddt_bias exercise real gradient work.  Saturated
    # clamp values belong in precision boundary cases, not the headline perf case.
    "full": FeatureConfig(True, True, True, True, True, (1e-3, 1.0)),
}


class SavedTensorStats:
    """Collect logical tensor bytes and deduplicated backing-storage bytes."""

    def __init__(self):
        self.logical_bytes = 0
        self.storages: dict[tuple[str, int | None, int], int] = {}
        self.signatures: dict[
            tuple[tuple[int, ...], str], dict[str, int]
        ] = defaultdict(lambda: {"count": 0, "logical_bytes": 0})

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        self.logical_bytes += tensor.numel() * tensor.element_size()
        signature = (tuple(tensor.shape), str(tensor.dtype))
        self.signatures[signature]["count"] += 1
        self.signatures[signature]["logical_bytes"] += (
            tensor.numel() * tensor.element_size()
        )
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        self.storages[key] = max(self.storages.get(key, 0), storage.nbytes())
        return tensor

    @staticmethod
    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @property
    def unique_bytes(self) -> int:
        return sum(self.storages.values())

    @property
    def breakdown(self) -> list[dict]:
        rows = [
            {
                "shape": list(shape),
                "dtype": dtype,
                "count": values["count"],
                "logical_mib": values["logical_bytes"] / 2**20,
            }
            for (shape, dtype), values in self.signatures.items()
        ]
        return sorted(rows, key=lambda row: -row["logical_mib"])


def _leaf(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device).detach().requires_grad_(True)


def make_inputs(case_name: str, feature_name: str, device: torch.device):
    batch, seqlen, nheads, headdim, dstate, ngroups, _ = CASES[case_name]
    feature = FEATURES[feature_name]
    generator = torch.Generator(device="cpu").manual_seed(20260806)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    values = {
        "x": randn(batch, seqlen, nheads, headdim),
        "dt": 0.01
        + 0.1
        * torch.rand(
            batch, seqlen, nheads, generator=generator, dtype=torch.float32
        ),
        "A": -0.1 - 0.4 * torch.rand(nheads, generator=generator),
        "B": randn(batch, seqlen, ngroups, dstate) / 5,
        "C": randn(batch, seqlen, ngroups, dstate) / 5,
        "D": randn(nheads, headdim) if feature.use_d else None,
        "z": randn(batch, seqlen, nheads, headdim) if feature.use_z else None,
        "dt_bias": randn(nheads) * 0.1 if feature.use_dt_bias else None,
        "initial_states": (
            randn(batch, nheads, headdim, dstate) / 5
            if feature.use_initial_state
            else None
        ),
    }
    leaves = {
        name: _leaf(value, device) if isinstance(value, torch.Tensor) else None
        for name, value in values.items()
    }
    # NPU H2D copies are asynchronous even for these ordinary CPU tensors.
    # Keep every source in ``values`` alive until the transfers complete;
    # otherwise large benchmark inputs can be released/reused before the
    # device has consumed them, corrupting correctness checks while leaving
    # steady-state timing deceptively stable.
    if device.type == "npu":
        torch.npu.synchronize(device)
    return leaves


def trainable_inputs(inputs) -> tuple[list[str], list[torch.Tensor]]:
    names = []
    tensors = []
    for name in ("x", "dt", "A", "B", "C", "D", "z", "dt_bias", "initial_states"):
        value = inputs[name]
        if value is not None:
            names.append(name)
            tensors.append(value)
    return names, tensors


def make_forward(
    backend: str,
    case_name: str,
    feature_name: str,
    inputs,
    return_final_state: bool,
):
    _, _, _, _, _, _, chunk_size = CASES[case_name]
    feature = FEATURES[feature_name]

    if backend == "reference":
        from mamba_torch.ssd_reference import ssd_chunk_scan_ref

        def run():
            return ssd_chunk_scan_ref(
                inputs["x"],
                inputs["dt"],
                inputs["A"],
                inputs["B"],
                inputs["C"],
                chunk_size,
                D=inputs["D"],
                z=inputs["z"],
                dt_bias=inputs["dt_bias"],
                dt_softplus=feature.dt_softplus,
                dt_limit=feature.dt_limit,
                initial_states=inputs["initial_states"],
                return_final_state=return_final_state,
            )

        return run

    if backend == "mamba_ssm":
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        def run():
            return mamba_chunk_scan_combined(
                inputs["x"],
                inputs["dt"],
                inputs["A"],
                inputs["B"],
                inputs["C"],
                chunk_size,
                D=inputs["D"],
                z=inputs["z"],
                dt_bias=inputs["dt_bias"],
                initial_states=inputs["initial_states"],
                dt_softplus=feature.dt_softplus,
                dt_limit=feature.dt_limit,
                return_final_states=return_final_state,
            )

        return run

    if backend == "ascendc":
        ascend_kernel = load_ascend_backend()

        def run():
            return ascend_kernel.mamba2_ssd_fwd(
                inputs["x"],
                inputs["dt"],
                inputs["A"],
                inputs["B"],
                inputs["C"],
                chunk_size,
                D=inputs["D"],
                z=inputs["z"],
                dt_bias=inputs["dt_bias"],
                initial_states=inputs["initial_states"],
                dt_softplus=feature.dt_softplus,
                dt_limit=feature.dt_limit,
                return_final_state=return_final_state,
            )

        return run

    raise ValueError(f"Unsupported backend: {backend}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def timed_samples(
    call: Callable[[], object], device: torch.device, warmup: int, repeat: int
) -> list[float]:
    for _ in range(warmup):
        call()
        synchronize(device)

    samples = []
    for _ in range(repeat):
        if device.type in ("cuda", "npu"):
            runtime = torch.cuda if device.type == "cuda" else torch.npu
            start = runtime.Event(enable_timing=True)
            end = runtime.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start_ns = time.perf_counter_ns()
            call()
            synchronize(device)
            samples.append((time.perf_counter_ns() - start_ns) / 1e6)
    return samples


def summarize(samples: Iterable[float]) -> dict[str, float]:
    values = list(samples)
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p90_ms": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def select_outputs(
    result, dout: torch.Tensor, dfinal: torch.Tensor, final_state_grad: bool
):
    if final_state_grad:
        out, final_state = result
        return (out, final_state), (dout, dfinal)
    return (result,), (dout,)


def autograd_grad(
    result,
    grad_inputs: list[torch.Tensor],
    dout: torch.Tensor,
    dfinal: torch.Tensor,
    final_state_grad: bool,
    retain_graph: bool,
):
    outputs, grad_outputs = select_outputs(result, dout, dfinal, final_state_grad)
    return torch.autograd.grad(
        outputs,
        grad_inputs,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )


def memory_metrics(
    forward,
    grad_inputs,
    dout,
    dfinal,
    final_state_grad,
    device: torch.device,
):
    stats = SavedTensorStats()
    accelerator_memory = (
        torch.cuda
        if device.type == "cuda"
        else torch.npu if device.type == "npu" else None
    )
    if accelerator_memory is not None:
        synchronize(device)
        accelerator_memory.empty_cache()
        accelerator_memory.reset_peak_memory_stats(device)
        base_allocated = accelerator_memory.memory_allocated(device)
        base_reserved = accelerator_memory.memory_reserved(device)
    else:
        base_allocated = None
        base_reserved = None
    with torch.autograd.graph.saved_tensors_hooks(stats.pack, stats.unpack):
        result = forward()
    synchronize(device)
    after_forward = (
        accelerator_memory.memory_allocated(device)
        if accelerator_memory is not None
        else None
    )
    grads = autograd_grad(
        result,
        grad_inputs,
        dout,
        dfinal,
        final_state_grad,
        retain_graph=False,
    )
    synchronize(device)
    peak_allocated = (
        accelerator_memory.max_memory_allocated(device)
        if accelerator_memory is not None
        else None
    )
    peak_reserved = (
        accelerator_memory.max_memory_reserved(device)
        if accelerator_memory is not None
        else None
    )
    del result, grads
    return {
        "saved_tensor_logical_mib": stats.logical_bytes / 2**20,
        "saved_tensor_unique_storage_mib": stats.unique_bytes / 2**20,
        "forward_allocated_extra_mib": (
            (after_forward - base_allocated) / 2**20
            if accelerator_memory is not None
            else None
        ),
        "training_peak_extra_mib": (
            (peak_allocated - base_allocated) / 2**20
            if accelerator_memory is not None
            else None
        ),
        "training_peak_reserved_extra_mib": (
            (peak_reserved - base_reserved) / 2**20
            if accelerator_memory is not None
            else None
        ),
    }


def benchmark_case(
    backend: str,
    device: torch.device,
    case_name: str,
    feature_name: str,
    warmup: int,
    repeat: int,
    final_state_grad: bool,
):
    inputs = make_inputs(case_name, feature_name, device)
    grad_names, grad_inputs = trainable_inputs(inputs)
    forward = make_forward(
        backend,
        case_name,
        feature_name,
        inputs,
        return_final_state=final_state_grad,
    )
    batch, _, nheads, headdim, dstate, _, _ = CASES[case_name]
    torch.manual_seed(20260806)
    dout = torch.randn_like(inputs["x"])
    dfinal = torch.randn(
        batch,
        nheads,
        headdim,
        dstate,
        dtype=torch.float32,
        device=device,
    )

    forward_times = timed_samples(forward, device, warmup, repeat)

    fixed_result = forward()
    synchronize(device)

    def backward_only():
        return autograd_grad(
            fixed_result,
            grad_inputs,
            dout,
            dfinal,
            final_state_grad,
            retain_graph=True,
        )

    backward_times = timed_samples(backward_only, device, warmup, repeat)
    del fixed_result

    def forward_backward():
        result = forward()
        return autograd_grad(
            result,
            grad_inputs,
            dout,
            dfinal,
            final_state_grad,
            retain_graph=False,
        )

    forward_backward_times = timed_samples(
        forward_backward, device, warmup, repeat
    )
    one_result = forward()
    one_grads = autograd_grad(
        one_result,
        grad_inputs,
        dout,
        dfinal,
        final_state_grad,
        retain_graph=False,
    )
    synchronize(device)
    output_tensors = one_result if isinstance(one_result, tuple) else (one_result,)
    outputs_finite = all(
        bool(torch.isfinite(value).all().item()) for value in output_tensors
    )
    finite = all(torch.isfinite(grad).all().item() for grad in one_grads)
    raw_gradient_checksums = {
        name: grad.detach().float().sum().item()
        for name, grad in zip(grad_names, one_grads)
    }
    gradient_checksums = {
        name: value if math.isfinite(value) else None
        for name, value in raw_gradient_checksums.items()
    }
    grad_checksum = (
        sum(raw_gradient_checksums.values())
        if all(math.isfinite(value) for value in raw_gradient_checksums.values())
        else None
    )
    del one_result, one_grads

    memory = memory_metrics(
        forward,
        grad_inputs,
        dout,
        dfinal,
        final_state_grad,
        device,
    )
    batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size = CASES[case_name]
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "npu":
        device_name = torch.npu.get_device_name(device)
    else:
        device_name = platform.processor()
    device_properties = (
        torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    )
    try:
        backend_version = (
            importlib.metadata.version("mamba-ssm")
            if backend == "mamba_ssm"
            else (
                importlib.metadata.version("mamba-ascendc")
                if backend == "ascendc"
                else None
            )
        )
    except importlib.metadata.PackageNotFoundError:
        backend_version = "unknown"
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "device_type": device.type,
        "device": device_name or platform.machine(),
        "device_total_memory_mib": (
            device_properties.total_memory / 2**20 if device_properties else None
        ),
        "compute_capability": (
            [device_properties.major, device_properties.minor]
            if device_properties
            else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "backend_version": backend_version,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
        ),
        "case": case_name,
        "shape_b_l_h_p_n_c_g": [
            batch,
            seqlen,
            nheads,
            headdim,
            dstate,
            chunk_size,
            ngroups,
        ],
        "logical_head_chunk_tasks": batch * nheads * (seqlen // chunk_size),
        "dtype": "float32",
        "feature": feature_name,
        "final_state_grad": final_state_grad,
        "gradient_names": grad_names,
        "warmup": warmup,
        "repeat": repeat,
        "timing_method": (
            "device_event_per_iteration"
            if device.type in ("cuda", "npu")
            else "perf_counter_ns_with_device_sync"
        ),
        "forward": summarize(forward_times),
        "backward_only": summarize(backward_times),
        "forward_backward": summarize(forward_backward_times),
        **memory,
        "outputs_finite": outputs_finite,
        "gradients_finite": finite,
        "gradient_checksums": gradient_checksums,
        "gradient_checksum": grad_checksum,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("reference", "mamba_ssm", "ascendc"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), required=True)
    parser.add_argument("--feature", choices=tuple(FEATURES), default="full")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--final-state-grad", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSONL output path; overwritten for this benchmark run",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be >= 0 and repeat must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.backend == "mamba_ssm" and args.device != "cuda":
        parser.error("mamba_ssm benchmark requires --device cuda")
    if args.backend == "ascendc" and args.device != "npu":
        parser.error("ascendc benchmark requires --device npu")
    if args.device == "npu":
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            parser.error("NPU is not available")
        if args.backend == "ascendc":
            load_ascend_backend()

    device = torch.device(args.device)
    output_handle = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")
    try:
        for case_name in args.cases:
            try:
                result = benchmark_case(
                    args.backend,
                    device,
                    case_name,
                    args.feature,
                    args.warmup,
                    args.repeat,
                    args.final_state_grad,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                if "out of memory" not in str(error).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = {
                    "status": "oom",
                    "backend": args.backend,
                    "device_type": args.device,
                    "case": case_name,
                    "feature": args.feature,
                    "final_state_grad": args.final_state_grad,
                    "error": str(error),
                }
            line = json.dumps(result, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            if output_handle is not None:
                output_handle.write(line + "\n")
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()


if __name__ == "__main__":
    main()
