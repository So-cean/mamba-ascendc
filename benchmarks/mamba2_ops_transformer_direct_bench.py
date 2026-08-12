"""Direct same-shape benchmark against ops-transformer experimental Mamba-2.

The external implementation is loaded as four separately packaged shared
libraries because the CANN 8.2 fat-object linker cannot combine multiple MIX
translation units.  A raw-pointer C ABI bypasses the upstream Torch host ABI,
which was built for a different libstdc++ version.  Packaging and binding are
changed, but the four upstream device kernels, tiling, and execution chain are
not modified.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ.setdefault("MAMBA_ASCENDC_CHUNK_MIX", "1")
os.environ.setdefault("MAMBA_ASCENDC_CHUNK128", "1")

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


CASES = {
    "s1": (1, 256, 128, 64, 128, 8),
    "s2": (1, 512, 128, 64, 128, 8),
    "s3": (1, 1024, 128, 64, 128, 8),
    "s4": (1, 2048, 128, 64, 128, 8),
    "s5": (2, 2048, 128, 64, 128, 8),
}
CHUNK_SIZE = 256
_OPS_LIBRARIES = None
_OPS_KEEPALIVE = None


def load_ops_transformer(library_dir: Path) -> None:
    global _OPS_LIBRARIES
    if _OPS_LIBRARIES is not None:
        return
    libraries = {}
    for suffix in ("cumsum", "state", "state_passing", "scan"):
        path = library_dir / f"_C_{suffix}.so"
        if not path.is_file():
            raise FileNotFoundError(path)
        libraries[suffix] = ctypes.CDLL(
            str(path.resolve()), mode=ctypes.RTLD_GLOBAL
        )

    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    signatures = {
        "cumsum": (9, 4),
        "state": (7, 7),
        "state_passing": (8, 9),
        "scan": (13, 7),
    }
    for suffix, (pointer_count, integer_count) in signatures.items():
        workspace_size = getattr(
            libraries[suffix],
            f"mamba2_chunk_{suffix}_system_workspace_size",
        )
        workspace_size.argtypes = []
        workspace_size.restype = ctypes.c_uint64
        launch = getattr(libraries[suffix], f"launch_mamba2_chunk_{suffix}")
        launch.argtypes = [pointer] * pointer_count + [integer] * integer_count
        launch.restype = integer
    _OPS_LIBRARIES = libraries


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _stream_ptr():
    return ctypes.c_void_p(torch.npu.current_stream().npu_stream)


def _system_workspace_size(suffix: str) -> int:
    function = getattr(
        _OPS_LIBRARIES[suffix],
        f"mamba2_chunk_{suffix}_system_workspace_size",
    )
    return int(function())


def _launch(suffix: str, tensors, dimensions) -> None:
    function = getattr(_OPS_LIBRARIES[suffix], f"launch_mamba2_chunk_{suffix}")
    debug = os.environ.get("OPS_TRANSFORMER_DEBUG_SYNC") == "1"
    if debug:
        print(f"launch_{suffix}=START", flush=True)
    status = function(
        *(_ptr(tensor) for tensor in tensors),
        _stream_ptr(),
        *dimensions,
    )
    if status != 0:
        raise RuntimeError(f"ops-transformer {suffix} launch failed: {status}")
    if debug:
        torch.npu.synchronize()
        print(f"launch_{suffix}=PASS", flush=True)


def make_inputs(case: str):
    batch, seqlen, nheads, headdim, dstate, ngroups = CASES[case]
    generator = torch.Generator().manual_seed(20260811)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    values = {
        "x": randn(batch, seqlen, nheads, headdim),
        "dt": 0.01
        + 0.1 * torch.rand(batch, seqlen, nheads, generator=generator),
        "A": -(0.1 + 0.4 * torch.rand(nheads, generator=generator)),
        "B": randn(batch, seqlen, ngroups, dstate) / 5,
        "C": randn(batch, seqlen, ngroups, dstate) / 5,
        "D": randn(nheads),
        "dt_bias": randn(nheads) * 0.1,
        "initial_states": randn(batch, nheads, headdim, dstate) / 5,
    }
    return {name: value.npu() for name, value in values.items()}


def invoke_project(values):
    return ascend_kernel.mamba2_ssd_fwd(
        values["x"],
        values["dt"],
        values["A"],
        values["B"],
        values["C"],
        CHUNK_SIZE,
        D=values["D"],
        dt_bias=values["dt_bias"],
        dt_softplus=True,
        initial_states=values["initial_states"],
        return_final_state=True,
    )


def invoke_reference(values):
    return ssd_chunk_scan_ref(
        values["x"],
        values["dt"],
        values["A"],
        values["B"],
        values["C"],
        CHUNK_SIZE,
        D=values["D"],
        dt_bias=values["dt_bias"],
        dt_softplus=True,
        initial_states=values["initial_states"],
        return_final_state=True,
    )


def invoke_ops_transformer(values):
    global _OPS_KEEPALIVE
    batch, seqlen, nheads, headdim = values["x"].shape
    ngroups, dstate = values["B"].shape[-2:]
    nchunks = seqlen // CHUNK_SIZE
    x = values["x"].reshape(
        batch, nchunks, CHUNK_SIZE, nheads, headdim
    )
    b = values["B"].reshape(
        batch, nchunks, CHUNK_SIZE, ngroups, dstate
    )
    c = values["C"].reshape(
        batch, nchunks, CHUNK_SIZE, ngroups, dstate
    )
    dt = values["dt"].reshape(batch, nchunks, CHUNK_SIZE, nheads)
    mask = torch.ones_like(dt)

    a_fp32 = values["A"].to(torch.float32)
    dt_fp16 = dt.to(torch.float16)
    bias_fp16 = values["dt_bias"].to(torch.float16)
    mask_fp16 = mask.to(torch.float16)
    dt_out = torch.empty_like(dt, dtype=torch.float32)
    d_a_cs = torch.empty_like(dt, dtype=torch.float32)
    d_a_last = torch.empty(
        (batch, nchunks, 1, nheads), device=dt.device, dtype=torch.float32
    )
    cumsum_workspace = torch.empty(
        1024 + _system_workspace_size("cumsum"),
        device=dt.device,
        dtype=torch.uint8,
    )
    _launch(
        "cumsum",
        (
            a_fp32,
            dt_fp16,
            bias_fp16,
            mask_fp16,
            dt_out,
            d_a_cs,
            d_a_last,
            cumsum_workspace,
        ),
        (batch, nchunks, nheads, CHUNK_SIZE),
    )

    b_fp16 = b.to(torch.float16)
    x_fp16 = x.to(torch.float16)
    chunk_states = torch.empty(
        (batch, nchunks, nheads, dstate, headdim),
        device=x.device,
        dtype=torch.float32,
    )
    state_workspace_elements = 20 * (
        CHUNK_SIZE * 8 + 8 * CHUNK_SIZE * 64 * 3
    ) + _system_workspace_size("state")
    state_workspace = torch.empty(
        state_workspace_elements, device=x.device, dtype=torch.float32
    )
    _launch(
        "state",
        (dt_out, d_a_cs, b_fp16, x_fp16, chunk_states, state_workspace),
        (
            batch,
            nchunks,
            nheads,
            ngroups,
            CHUNK_SIZE,
            dstate,
            headdim,
        ),
    )

    initial_np = values["initial_states"].transpose(-1, -2).contiguous()
    d_a_chunk = d_a_last.squeeze(2)
    initial_flat = initial_np.reshape(batch, nheads, dstate * headdim)
    chunk_states_flat = chunk_states.reshape(
        batch, nchunks, nheads, dstate * headdim
    )
    c_fp16 = c.to(torch.float16)
    final_np = torch.empty(
        (batch, nheads, dstate, headdim),
        device=x.device,
        dtype=torch.float32,
    )
    state_projection = torch.empty(
        (batch, nchunks, nheads, CHUNK_SIZE, headdim),
        device=x.device,
        dtype=torch.float32,
    )
    state_passing_workspace_elements = (
        batch * nheads * dstate * headdim
        + 20 * dstate * headdim * 3
        + _system_workspace_size("state_passing")
    )
    state_passing_workspace = torch.empty(
        state_passing_workspace_elements,
        device=x.device,
        dtype=torch.float32,
    )
    _launch(
        "state_passing",
        (
            d_a_chunk,
            initial_flat,
            chunk_states_flat,
            c_fp16,
            final_np,
            state_projection,
            state_passing_workspace,
        ),
        (
            batch,
            nheads,
            seqlen,
            nchunks,
            CHUNK_SIZE,
            ngroups,
            dstate,
            headdim,
            dstate * headdim,
        ),
    )

    d_fp16 = values["D"].to(torch.float16)
    d_a_scan = d_a_cs.permute(0, 1, 3, 2).contiguous()
    dt_scan = dt_out.permute(0, 1, 3, 2).contiguous()
    out_cb = torch.empty(
        (batch, nchunks, ngroups, CHUNK_SIZE, CHUNK_SIZE),
        device=x.device,
        dtype=torch.float32,
    )
    out_mm = torch.empty(
        (batch, nchunks, nheads, CHUNK_SIZE, CHUNK_SIZE),
        device=x.device,
        dtype=torch.float16,
    )
    out_mtx = torch.empty(
        (batch, nchunks, nheads, CHUNK_SIZE, headdim),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty(
        (batch, seqlen, nheads * headdim),
        device=x.device,
        dtype=torch.float32,
    )
    scan_workspace = torch.empty(
        20 * CHUNK_SIZE * CHUNK_SIZE + _system_workspace_size("scan"),
        device=x.device,
        dtype=torch.float32,
    )
    _launch(
        "scan",
        (
            c_fp16,
            b_fp16,
            x_fp16,
            d_fp16,
            state_projection,
            d_a_scan,
            dt_scan,
            out_cb,
            out_mm,
            out_mtx,
            out,
            scan_workspace,
        ),
        (
            batch,
            nchunks,
            nheads,
            ngroups,
            CHUNK_SIZE,
            dstate,
            headdim,
        ),
    )

    _OPS_KEEPALIVE = (
        a_fp32,
        dt_fp16,
        bias_fp16,
        mask_fp16,
        dt_out,
        d_a_cs,
        d_a_last,
        cumsum_workspace,
        b_fp16,
        x_fp16,
        chunk_states,
        state_workspace,
        initial_np,
        c_fp16,
        final_np,
        state_projection,
        state_passing_workspace,
        d_fp16,
        d_a_scan,
        dt_scan,
        out_cb,
        out_mm,
        out_mtx,
        out,
        scan_workspace,
    )
    return (
        out.reshape(batch, seqlen, nheads, headdim),
        final_np.transpose(-1, -2).contiguous(),
    )


def error_metrics(actual, expected):
    error = actual.float() - expected.float()
    return {
        "max_abs": error.abs().max().item(),
        "nrmse": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item(),
        "cosine": F.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def time_device(call, warmup: int, repeat: int):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for start, end in zip(starts, ends):
        start.record()
        call()
        end.record()
    torch.npu.synchronize()
    times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
        "p90_ms": sorted(times)[int(0.9 * (repeat - 1))],
    }


def run_case(case: str, warmup: int, repeat: int):
    values = make_inputs(case)
    with torch.no_grad():
        expected_out, expected_state = invoke_reference(values)
        project_out, project_state = invoke_project(values)
        external_out, external_state = invoke_ops_transformer(values)
        torch.npu.synchronize()
        precision = {
            "project_output": error_metrics(project_out, expected_out),
            "project_final_state": error_metrics(
                project_state, expected_state
            ),
            "ops_transformer_output": error_metrics(
                external_out, expected_out
            ),
            "ops_transformer_final_state": error_metrics(
                external_state, expected_state
            ),
        }
        failed = {
            name: metrics
            for name, metrics in precision.items()
            if not metrics["finite"]
            or metrics["nrmse"] > 5e-3
            or metrics["cosine"] < 0.999
        }
        if failed:
            raise RuntimeError(
                "precision gate failed before timing: "
                + json.dumps(failed, ensure_ascii=False)
            )
        result = {
            "case": case,
            "shape_b_l_h_p_n_g": list(CASES[case]),
            "chunk_size": CHUNK_SIZE,
            "input_dtype": "float32",
            "project": time_device(
                lambda: invoke_project(values), warmup, repeat
            ),
            "ops_transformer": time_device(
                lambda: invoke_ops_transformer(values), warmup, repeat
            ),
            **precision,
        }
    result["project_over_external"] = (
        result["ops_transformer"]["median_ms"]
        / result["project"]["median_ms"]
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-library-dir", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=list(CASES)
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_ops_transformer(args.external_library_dir)
    results = []
    for case in args.cases:
        result = run_case(case, args.warmup, args.repeat)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    main()
