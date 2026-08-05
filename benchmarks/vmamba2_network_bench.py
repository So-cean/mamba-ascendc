"""Forward and component benchmark for a pure Vision-Mamba2 network."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import json
import os
import statistics
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mamba_torch.vmamba2_network import (
    PureVisionMamba2,
    SsdCore,
    VisionMamba2Config,
    VisionMamba2Mixer,
    count_mamba_modules,
)


CONFIGS = {
    "smoke": VisionMamba2Config(
        image_size=64,
        dims=(64, 128),
        depths=(1, 1),
        ngroups=(1, 1),
        chunk_sizes=(64, 64),
        directions=2,
    ),
    "micro": VisionMamba2Config(
        image_size=256,
        dims=(64, 128, 256, 512),
        depths=(1, 1, 2, 1),
        ngroups=(1, 1, 2, 4),
        chunk_sizes=(128, 128, 128, 64),
        directions=4,
    ),
    "tiny": VisionMamba2Config(),
}


def load_npu_runtime() -> None:
    library = os.environ.get("MAMBA_CHUNK_MIX_OP_API_LIB")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def select_device(requested: str) -> torch.device:
    if requested == "npu":
        import torch_npu  # noqa: F401

        load_npu_runtime()
        return torch.device("npu:0")
    if requested == "cuda":
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    if hasattr(torch, "npu") and torch.npu.is_available():
        load_npu_runtime()
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def memory_backend(device: torch.device):
    if device.type == "cuda":
        return torch.cuda
    if device.type == "npu":
        return torch.npu
    return None


def reset_peak_memory(device: torch.device) -> None:
    backend = memory_backend(device)
    if backend is not None:
        backend.reset_peak_memory_stats(device)


def memory_metrics(device: torch.device) -> dict[str, float | int] | None:
    backend = memory_backend(device)
    if backend is None:
        return None
    allocated = backend.memory_allocated(device)
    reserved = backend.memory_reserved(device)
    peak_allocated = backend.max_memory_allocated(device)
    peak_reserved = backend.max_memory_reserved(device)
    result: dict[str, float | int] = {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "allocated_gib": allocated / 2**30,
        "reserved_gib": reserved / 2**30,
        "peak_allocated_gib": peak_allocated / 2**30,
        "peak_reserved_gib": peak_reserved / 2**30,
    }
    try:
        free, total = backend.mem_get_info(device)
    except (AttributeError, RuntimeError, TypeError):
        return result
    result.update(
        {
            "device_free_bytes": free,
            "device_total_bytes": total,
            "device_free_gib": free / 2**30,
            "device_total_gib": total / 2**30,
            "peak_reserved_fraction_of_total": peak_reserved / total,
        }
    )
    return result


def new_event(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.Event(enable_timing=True)
    if device.type == "npu":
        return torch.npu.Event(enable_timing=True)
    raise RuntimeError("device events require CUDA or NPU")


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p90_ms": percentile(values, 0.9),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def benchmark_full(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[torch.Tensor, list[float]]:
    with torch.no_grad():
        for _ in range(warmup):
            output = model(image)
        synchronize(device)
        starts = [new_event(device) for _ in range(repeat)]
        ends = [new_event(device) for _ in range(repeat)]
        for start, end in zip(starts, ends):
            start.record()
            output = model(image)
            end.record()
        synchronize(device)
    return output, [start.elapsed_time(end) for start, end in zip(starts, ends)]


class ComponentEvents:
    def __init__(self, device: torch.device, module_type: type) -> None:
        self.device = device
        self.module_type = module_type
        self.pending: dict[int, list] = {}
        self.pairs: list[tuple] = []
        self.handles = []

    def install(self, model: torch.nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, self.module_type):
                self.handles.append(module.register_forward_pre_hook(self._before))
                self.handles.append(module.register_forward_hook(self._after))

    def _before(self, module, _inputs) -> None:
        event = new_event(self.device)
        event.record()
        self.pending.setdefault(id(module), []).append(event)

    def _after(self, module, _inputs, _output) -> None:
        end = new_event(self.device)
        end.record()
        start = self.pending[id(module)].pop()
        self.pairs.append((start, end))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def benchmark_components(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[list[float], list[float], list[float]]:
    with torch.no_grad():
        for _ in range(warmup):
            model(image)
        synchronize(device)

        mixer_events = ComponentEvents(device, VisionMamba2Mixer)
        ssd_events = ComponentEvents(device, SsdCore)
        mixer_events.install(model)
        ssd_events.install(model)
        mixer_ranges = []
        ssd_ranges = []
        full_pairs = []
        for _ in range(repeat):
            mixer_begin = len(mixer_events.pairs)
            ssd_begin = len(ssd_events.pairs)
            full_start = new_event(device)
            full_end = new_event(device)
            full_start.record()
            model(image)
            full_end.record()
            full_pairs.append((full_start, full_end))
            mixer_ranges.append((mixer_begin, len(mixer_events.pairs)))
            ssd_ranges.append((ssd_begin, len(ssd_events.pairs)))
        synchronize(device)
        mixer_events.remove()
        ssd_events.remove()

    mixer_times = [
        sum(mixer_events.pairs[index][0].elapsed_time(
            mixer_events.pairs[index][1]
        ) for index in range(begin, end))
        for begin, end in mixer_ranges
    ]
    ssd_times = [
        sum(ssd_events.pairs[index][0].elapsed_time(
            ssd_events.pairs[index][1]
        ) for index in range(begin, end))
        for begin, end in ssd_ranges
    ]
    instrumented_full_times = [
        start.elapsed_time(end) for start, end in full_pairs
    ]
    return mixer_times, ssd_times, instrumented_full_times


def accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    error = actual - expected
    return {
        "max_abs": error.abs().max().item(),
        "mean_abs": error.abs().mean().item(),
        "nrmse": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected).clamp_min(1e-12)
        ).item(),
        "cosine": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "npu":
        return torch.npu.get_device_name(device)
    return "CPU"


def stage_ssd_shapes(
    config: VisionMamba2Config, batch: int
) -> list[dict[str, int | list[int]]]:
    height = config.image_size // config.patch_size
    width = height
    shapes = []
    for stage, (dim, depth, groups, chunk) in enumerate(
        zip(config.dims, config.depths, config.ngroups, config.chunk_sizes)
    ):
        length = height * width
        padded = ((length + chunk - 1) // chunk) * chunk
        heads = dim * config.expand // config.headdim
        shapes.append(
            {
                "stage": stage,
                "resolution": [height, width],
                "depth": depth,
                "ssd_shape_b_l_h_p_n_c_g": [
                    batch,
                    padded,
                    heads,
                    config.headdim,
                    config.dstate,
                    chunk,
                    groups,
                ],
                "real_length": length,
                "padded_length": padded,
                "invocations": depth * config.directions,
            }
        )
        height //= 2
        width //= 2
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cuda", "npu", "cpu"), default="auto")
    parser.add_argument("--backend", choices=("auto", "cuda", "npu", "reference"), default="auto")
    parser.add_argument("--config", choices=tuple(CONFIGS), default="tiny")
    parser.add_argument("--image-size", type=int)
    parser.add_argument(
        "--merge-mode", choices=("stream", "stack"), default="stream"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="measure only the uninstrumented full network (for memory calibration)",
    )
    parser.add_argument("--save-output")
    parser.add_argument("--reference-output")
    parser.add_argument("--output")
    args = parser.parse_args()

    device = select_device(args.device)
    if device.type == "cpu" and args.repeat > 2:
        raise ValueError("CPU reference is intended only for smoke runs")
    config = CONFIGS[args.config]
    if args.image_size is not None:
        if args.image_size <= 0 or args.image_size % config.patch_size:
            raise ValueError("image size must be positive and divisible by patch size")
        config = replace(config, image_size=args.image_size)
    config = replace(config, streaming_merge=args.merge_mode == "stream")
    torch.manual_seed(20260803)
    model = PureVisionMamba2(config, backend=args.backend).eval().to(device)
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    image = torch.randn(
        args.batch,
        config.in_channels,
        config.image_size,
        config.image_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    reset_peak_memory(device)
    output, full_times = benchmark_full(
        model, image, device, args.warmup, args.repeat
    )
    if args.skip_components:
        mixer_times = ssd_times = instrumented_full_times = None
    else:
        mixer_times, ssd_times, instrumented_full_times = benchmark_components(
            model, image, device, args.warmup, args.repeat
        )
    with torch.no_grad():
        repeat_output = model(image)
        synchronize(device)

    result = {
        "device": device_name(device),
        "device_type": device.type,
        "backend": args.backend,
        "config": args.config,
        "batch": args.batch,
        "image_shape": list(image.shape),
        "dims": list(config.dims),
        "depths": list(config.depths),
        "directions": config.directions,
        "merge_mode": args.merge_mode,
        "chunk_sizes": list(config.chunk_sizes),
        "stage_ssd_shapes": stage_ssd_shapes(config, args.batch),
        "modules": count_mamba_modules(model),
        "ssd_invocations_per_forward": sum(config.depths) * config.directions,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "full_network": summarize(full_times),
        "memory": memory_metrics(device),
        "component_timing_note": (
            "Component shares use the same event-instrumented forward; "
            "full_network is measured separately without hooks."
        ),
        "repeat_determinism": accuracy(repeat_output, output),
        "output_shape": list(output.shape),
        "output_checksum": output.float().sum().item(),
    }
    if instrumented_full_times is not None:
        result.update(
            {
                "instrumented_full_network": summarize(instrumented_full_times),
                "mamba_mixers_only": summarize(mixer_times),
                "ssd_cores_only": summarize(ssd_times),
                "mamba_share_of_instrumented_full": statistics.median(mixer_times)
                / statistics.median(instrumented_full_times),
                "ssd_share_of_instrumented_full": statistics.median(ssd_times)
                / statistics.median(instrumented_full_times),
            }
        )
    if args.reference_output:
        expected = torch.load(args.reference_output, map_location="cpu")
        result["cross_device_accuracy"] = accuracy(output, expected)
    if args.save_output:
        destination = Path(args.save_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output.float().cpu(), destination)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
