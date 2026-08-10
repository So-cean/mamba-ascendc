"""Matched GPU/NPU benchmark for Mamba-2 forward execution.

Training mode keeps autograd enabled and reports the tensors saved by the
backend for backward.  Inference mode executes the same public operator under
``torch.no_grad()``.  Both modes reuse the case/input definitions from the
backward benchmark so cross-platform shapes cannot silently diverge.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from mamba2_backward_bench import (
    CASES,
    FEATURES,
    SavedTensorStats,
    load_ascend_backend,
    make_forward,
    make_inputs,
    summarize,
    synchronize,
    timed_samples,
)


def benchmark_case(
    backend: str,
    device: torch.device,
    case_name: str,
    feature_name: str,
    mode: str,
    warmup: int,
    repeat: int,
    return_final_state: bool,
) -> dict:
    inputs = make_inputs(case_name, feature_name, device)
    forward = make_forward(
        backend, case_name, feature_name, inputs, return_final_state
    )

    if mode == "inference":
        def invoke():
            with torch.no_grad():
                return forward()
    else:
        invoke = forward

    samples = timed_samples(invoke, device, warmup, repeat)

    saved = SavedTensorStats()
    if mode == "training":
        with torch.autograd.graph.saved_tensors_hooks(saved.pack, saved.unpack):
            result = forward()
    else:
        with torch.no_grad():
            result = forward()
    synchronize(device)
    output = result[0] if isinstance(result, tuple) else result
    output_finite = bool(torch.isfinite(output).all().item())
    final_state_finite = (
        bool(torch.isfinite(result[1]).all().item())
        if isinstance(result, tuple)
        else None
    )
    raw_checksum = output.detach().float().sum().item()
    checksum = raw_checksum if math.isfinite(raw_checksum) else None
    del result, output

    batch, seqlen, nheads, headdim, dstate, ngroups, chunk_size = (
        CASES[case_name]
    )
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        properties = torch.cuda.get_device_properties(device)
    elif device.type == "npu":
        device_name = torch.npu.get_device_name(device)
        properties = None
    else:
        device_name = platform.processor()
        properties = None
    try:
        backend_version = importlib.metadata.version(
            "mamba-ssm" if backend == "mamba_ssm" else "mamba-ascendc"
        )
    except importlib.metadata.PackageNotFoundError:
        backend_version = "unknown"

    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "backend_version": backend_version,
        "device_type": device.type,
        "device": device_name,
        "device_total_memory_mib": (
            properties.total_memory / 2**20 if properties else None
        ),
        "case": case_name,
        "shape_b_l_h_p_n_c_g": [
            batch, seqlen, nheads, headdim, dstate, chunk_size, ngroups
        ],
        "logical_head_chunk_tasks": batch * nheads * (seqlen // chunk_size),
        "dtype": "float32",
        "feature": feature_name,
        "mode": mode,
        "return_final_state": return_final_state,
        "warmup": warmup,
        "repeat": repeat,
        "timing_method": "device_event_per_iteration",
        "forward": summarize(samples),
        "saved_tensor_logical_mib": saved.logical_bytes / 2**20,
        "saved_tensor_unique_storage_mib": saved.unique_bytes / 2**20,
        "saved_tensor_breakdown": saved.breakdown,
        "output_finite": output_finite,
        "final_state_finite": final_state_finite,
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("mamba_ssm", "ascendc"), required=True
    )
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), required=True)
    parser.add_argument("--feature", choices=tuple(FEATURES), default="full")
    parser.add_argument("--mode", choices=("training", "inference"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--return-final-state", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.backend == "mamba_ssm" and args.device != "cuda":
        parser.error("mamba_ssm requires --device cuda")
    if args.backend == "ascendc" and args.device != "npu":
        parser.error("ascendc requires --device npu")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be >= 0 and repeat must be > 0")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.device == "npu":
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            parser.error("NPU is not available")
        load_ascend_backend()

    device = torch.device(args.device)
    rows = [
        benchmark_case(
            args.backend,
            device,
            case,
            args.feature,
            args.mode,
            args.warmup,
            args.repeat,
            args.return_final_state,
        )
        for case in args.cases
    ]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
