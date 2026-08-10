#!/usr/bin/env python3
"""torch_npu.profiler comparison for Mamba2SsdChunkScanBwdOff."""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import torch_npu

import ascend_kernel  # noqa: F401 -- registers the custom operator


WARMUP = 5
ACTIVE = 5
REPEAT = 1
STEPS = WARMUP + ACTIVE
TEST_DIR = Path(__file__).resolve().parent
CASE_FILE = TEST_DIR / "mamba2_ssd_chunk_scan_bwd_off_perf_cases.jsonl"
TRACE_ROOT = TEST_DIR / "profiler_trace" / "mamba2_ssd_chunk_scan_bwd_off"
REPORT_FILE = TEST_DIR / "mamba2_ssd_chunk_scan_bwd_off_torch_npu_profiler_report.md"


def load_cases():
    if CASE_FILE.suffix != ".jsonl":
        raise ValueError("performance cases must use JSONL")
    return [json.loads(line) for line in CASE_FILE.read_text().splitlines() if line.strip()]


def _specs(case):
    return {item["name"]: item for item in case["inputs"]}


def build_inputs(case, device):
    specs = _specs(case)
    generator = torch.Generator(device="cpu").manual_seed(20260808)

    def positive(name, scale=0.1):
        spec = specs[name]
        dtype = {"float32": torch.float32, "float16": torch.float16}[spec["dtype"]]
        value = scale + scale * torch.rand(
            tuple(spec["shape"]), generator=generator, dtype=torch.float32
        )
        return value.to(dtype=dtype, device=device)

    gy = positive("gy")
    states = positive("states_start")
    d_a = -positive("d_a_cumsum", scale=0.05)
    c_cube = positive("c_cube")
    torch.npu.synchronize()
    return gy, states, d_a, c_cube


def custom(inputs):
    return torch.ops.mamba_ascend.mamba2_ssd_chunk_scan_bwd_off(*inputs)


def baseline(inputs):
    gy, states, d_a, c_cube = inputs
    heads = gy.shape[1]
    groups = c_cube.shape[2]
    c_head = c_cube.permute(0, 2, 1, 3, 4).repeat_interleave(
        heads // groups, dim=1
    )
    q = (gy * torch.exp(d_a)[..., None]).half().float()
    state_half = states.half().float()
    c_float = c_head.float()
    d_states = torch.matmul(q.transpose(-1, -2), c_float)
    d_c_head = torch.matmul(q, state_half)
    g_d_a = (d_c_head * c_float).sum(dim=-1)
    return d_states, d_c_head, g_d_a


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
            wait=0, warmup=WARMUP, active=ACTIVE, repeat=REPEAT, skip_first=0
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
    c_cube = specs["c_cube"]["shape"]
    return [gy[0], gy[1], gy[2], c_cube[2]]


def render_report(rows):
    ratios = [row["ratio"] for row in rows]
    custom_better = sum(ratio > 1.0 for ratio in ratios)
    baseline_better = sum(ratio < 1.0 for ratio in ratios)
    avg_ratio = sum(ratios) / len(ratios)
    lines = [
        "# Mamba2SsdChunkScanBwdOff 性能评估结果",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "- 用例：`mamba2_ssd_chunk_scan_bwd_off_perf_cases.jsonl`",
        "- 固定 schedule：warmup=5、active=5、repeat=1。",
        "- 指标：每个路径独立采集 `op_statistic.csv`，所有行 `Total Time(us)` 求和后除以 5。",
        "- 标杆：同一 NPU 上的 PyTorch 小算子拼接（exp/cast/permute/repeat/matmul/mul/sum）。",
        "- 加速比：`标杆 / 自定义算子`，大于 1 表示自定义算子更快。",
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
            "- 小 shape 同时受 framework launch、workspace 和 AIC/AIV 同步开销影响；大 shape 更能体现 head-task 并行。",
            "- 自定义路径把两个 64×64 GEMM 放在同一个 MIX kernel，并在 Vector 侧就地完成 `gdA_cs`，避免标杆的多 kernel 中间张量往返。",
            "- 当前实现仍会将 Q、Q transpose 和 FP16 state 写入每核 GM workspace，后续可通过任务流水与 workspace 双缓冲继续优化。",
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
            f"csv=({custom_csv.name},{baseline_csv.name})"
        )
    REPORT_FILE.write_text(render_report(rows), encoding="utf-8")
    print(f"report={REPORT_FILE}")


if __name__ == "__main__":
    main()
