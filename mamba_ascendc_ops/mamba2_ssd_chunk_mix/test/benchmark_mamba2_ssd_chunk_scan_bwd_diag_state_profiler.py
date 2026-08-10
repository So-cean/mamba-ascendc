#!/usr/bin/env python3
"""Profiler comparison for the fused diagonal/state Mamba-2 backward."""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch_npu

import ascend_kernel  # noqa: F401 -- registers torch.ops.mamba_ascend


WARMUP = 5
ACTIVE = 5
REPEAT = 1
STEPS = WARMUP + ACTIVE
TEST_DIR = Path(__file__).resolve().parent
CASE_FILE = TEST_DIR / "mamba2_ssd_chunk_scan_bwd_diag_state_perf_cases.jsonl"
TRACE_ROOT = (
    TEST_DIR / "profiler_trace" / "mamba2_ssd_chunk_scan_bwd_diag_state"
)
REPORT_FILE = (
    TEST_DIR
    / "mamba2_ssd_chunk_scan_bwd_diag_state_torch_npu_profiler_report.md"
)


def load_cases():
    if CASE_FILE.suffix != ".jsonl":
        raise ValueError("performance cases must use JSONL")
    return [
        json.loads(line) for line in CASE_FILE.read_text().splitlines() if line.strip()
    ]


def _specs(case):
    return {item["name"]: item for item in case["inputs"]}


def build_inputs(case, device):
    specs = _specs(case)
    generator = torch.Generator(device="cpu").manual_seed(20260821)

    def positive(name, scale=0.1):
        spec = specs[name]
        dtype = {"float32": torch.float32, "float16": torch.float16}[
            spec["dtype"]
        ]
        value = scale + scale * torch.rand(
            tuple(spec["shape"]), generator=generator, dtype=torch.float32
        )
        return value.to(dtype=dtype, device=device)

    gy = positive("gy")
    x = positive("x_cube")
    steps = positive("d_a_cumsum", scale=0.002)
    d_a = -steps.cumsum(dim=-1)
    b_cube = positive("b_cube")
    c_cube = positive("c_cube")
    d_chunk_states = positive("d_chunk_states")
    torch.npu.synchronize()
    return gy, x, d_a, b_cube, c_cube, d_chunk_states


def custom(inputs):
    return torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_diag_state(
        *inputs
    )


def _expand_group(tensor, heads):
    groups = tensor.shape[2]
    return tensor.permute(0, 2, 1, 3, 4).repeat_interleave(
        heads // groups, dim=1
    )


def baseline(inputs):
    """Same-NPU tensor composition replaced by the fused MIX operator."""
    gy, x, d_a, b_cube, c_cube, d_chunk_states = inputs
    heads = gy.shape[1]
    b_nt = _expand_group(b_cube, heads).float()
    b_tn = b_nt.transpose(-1, -2).contiguous()
    c_tn = _expand_group(c_cube, heads).float()
    x_float = x.float()
    gy_half = gy.half().float()
    u_half = d_chunk_states.half().float()

    values = d_a.unsqueeze(-1) - d_a.unsqueeze(-2)
    causal = torch.tril(
        torch.ones((64, 64), dtype=torch.bool, device=gy.device)
    )
    decay = torch.where(causal, torch.exp(values), torch.zeros_like(values))
    cb = torch.matmul(c_tn, b_nt)
    w = (cb * decay).half().float()
    d_x_diag = torch.matmul(w.transpose(-1, -2), gy_half)
    d_w = torch.matmul(gy_half, x_float.transpose(-1, -2))
    d_cb = (d_w * decay).half().float()
    d_c = torch.matmul(d_cb, b_tn)
    d_b_diag = torch.matmul(d_cb.transpose(-1, -2), c_tn)
    product = d_w * w
    g_diag = product.sum(-1) - product.sum(-2)

    decay_to_end = torch.exp(d_a[..., -1:] - d_a)
    r = (x_float * decay_to_end.half().float().unsqueeze(-1)).half().float()
    d_r = torch.matmul(b_tn, u_half.transpose(-1, -2))
    d_b_state = torch.matmul(r, u_half)
    d_x = d_x_diag + d_r * decay_to_end.unsqueeze(-1)
    d_b = d_b_diag + d_b_state
    state_rows = (d_r * r).sum(-1)
    g_state = -state_rows
    tail = g_state[..., -1:] + state_rows.sum(-1, keepdim=True)
    g_state = torch.cat((g_state[..., :-1], tail), dim=-1)
    return d_x, d_b, d_c, g_diag + g_state


def newest_csv(root: Path) -> Path:
    deadline = time.time() + 120.0
    while time.time() < deadline:
        paths = list(root.glob("**/ASCEND_PROFILER_OUTPUT/op_statistic.csv"))
        if paths:
            return max(paths, key=lambda path: path.stat().st_mtime)
        time.sleep(0.2)
    raise FileNotFoundError(f"op_statistic.csv not found under {root}")


def sum_total_time_us(path: Path) -> float:
    total = 0.0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        key = next(
            (
                field
                for field in (reader.fieldnames or [])
                if "Total" in field and "us" in field.lower()
            ),
            None,
        )
        if key is None:
            raise RuntimeError(f"Total Time(us) column missing in {path}")
        for row in reader:
            value = str(row.get(key, "")).strip()
            if value:
                total += float(value)
    return total


def profile_one(fn, inputs, output_dir: Path) -> tuple[float, Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )
    with torch_npu.profiler.profile(
        activities=(
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ),
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=WARMUP,
            active=ACTIVE,
            repeat=REPEAT,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(output_dir)
        ),
        experimental_config=experimental,
    ) as prof:
        for _ in range(STEPS):
            outputs = fn(inputs)
            prof.step()
            del outputs
    torch.npu.synchronize()
    csv_path = newest_csv(output_dir)
    return sum_total_time_us(csv_path) / (ACTIVE * REPEAT), csv_path


def case_shape(case):
    specs = _specs(case)
    gy = specs["gy"]["shape"]
    b_cube = specs["b_cube"]["shape"]
    return [gy[0], gy[1], gy[2], b_cube[2]]


def render_report(rows):
    ratios = [row["ratio"] for row in rows]
    custom_better = sum(ratio > 1.0 for ratio in ratios)
    baseline_better = sum(ratio < 1.0 for ratio in ratios)
    avg_ratio = sum(ratios) / len(ratios)
    lines = [
        "# Mamba2SsdChunkScanBwdDiagState 性能评估结果",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 用例：`mamba2_ssd_chunk_scan_bwd_diag_state_perf_cases.jsonl`。",
        "- 固定 schedule：warmup=5、active=5、repeat=1。",
        "- 指标：每条路径独立采集 `op_statistic.csv`，所有行 `Total Time(us)` 求和后除以 5。",
        "- 标杆：同一 NPU 上被替换的 PyTorch tensor composition（7 matmul + decay/cast/reduce）。",
        "- 加速比：`标杆 / 自定义算子`，大于 1 表示 native AscendC 更快。",
        "",
        "## 性能对比",
        "",
        "| Case | Shape `[B,H,K,G]` | DType | 自定义算子(us) | 标杆(us) | 加速比 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | `{row['shape']}` | FP32/FP16 mixed | "
            f"{row['custom_us']:.3f} | {row['baseline_us']:.3f} | "
            f"{row['ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 全量汇总",
            "",
            "| 指标 | 值 |",
            "|---|---:|",
            f"| 用例数 | {len(rows)} |",
            f"| 平均加速比 | {avg_ratio:.3f} |",
            f"| 自定义算子更优 | {custom_better} |",
            f"| 标杆更优 | {baseline_better} |",
            "",
            "### 按数据类型汇总",
            "",
            "| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |",
            "|---|---:|---:|---:|---:|",
            f"| FP32/FP16 mixed | {len(rows)} | {avg_ratio:.3f} | "
            f"{custom_better} | {baseline_better} |",
            "",
            "## 简短分析",
            "",
            f"- 全部 shape 的平均标杆/自定义比值为 {avg_ratio:.3f}。",
            "- 单任务 shape 受一次 MIX launch、七个依赖 GEMM 和六次 AIC/AIV 交接影响，并非吞吐主场景。",
            "- 任务数增加后，`B×H×K` 的 head/chunk 并行能够填充更多 AIC；标杆仍需物化多个 FP32/FP16 中间张量。",
            "- native 路径减少了 public tensor launch 与 HBM 往返，但每个逻辑核仍使用 144 KiB GM workspace，后续可评估双缓冲和跨任务流水。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    cases = load_cases()
    if len(cases) < 8:
        raise RuntimeError("at least eight profiler cases are required")
    device = torch.device("npu:0")
    rows = []
    for index, case in enumerate(cases):
        inputs = build_inputs(case, device)
        custom_us, custom_csv = profile_one(
            custom, inputs, TRACE_ROOT / "custom" / f"case_{index:03d}"
        )
        baseline_us, baseline_csv = profile_one(
            baseline, inputs, TRACE_ROOT / "baseline" / f"case_{index:03d}"
        )
        ratio = baseline_us / custom_us
        rows.append(
            {
                "case": index,
                "shape": case_shape(case),
                "custom_us": custom_us,
                "baseline_us": baseline_us,
                "ratio": ratio,
            }
        )
        print(
            f"case={index} shape={case_shape(case)} custom={custom_us:.3f}us "
            f"baseline={baseline_us:.3f}us ratio={ratio:.3f} "
            f"csv=({custom_csv.name},{baseline_csv.name})",
            flush=True,
        )
    REPORT_FILE.write_text(render_report(rows), encoding="utf-8")
    print(f"report={REPORT_FILE}")


if __name__ == "__main__":
    main()
