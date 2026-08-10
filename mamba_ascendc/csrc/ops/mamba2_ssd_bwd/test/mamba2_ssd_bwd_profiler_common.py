"""torch_npu.profiler helpers for Mamba2 SSD backward M0."""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import time
from collections import defaultdict

import torch
import torch_npu

import ascend_kernel
from mamba_torch.ssd_reference import ssd_chunk_scan_ref


WARMUP = 5
ACTIVE = 5


def load_cases(path):
    if not path.endswith(".jsonl"):
        raise ValueError("performance cases must be JSONL")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_inputs(case, device):
    specs = {item["name"]: item for item in case["inputs"]}
    generator = torch.Generator(device="cpu").manual_seed(20260820)

    def positive(shape, low, span):
        return (low + span * torch.rand(shape, generator=generator)).to(device)

    x = positive(specs["x"]["shape"], 0.1, 0.15)
    dt = positive(specs["dt"]["shape"], 0.002, 0.004)
    A = -positive(specs["A"]["shape"], 0.005, 0.005)
    B = positive(specs["B"]["shape"], 0.05, 0.10)
    C = positive(specs["C"]["shape"], 0.05, 0.10)
    chunk_size = int(specs["chunk_size"]["value"])
    use_final_grad = bool(specs["use_final_grad"]["value"])
    return x, dt, A, B, C, chunk_size, use_final_grad


def prepare_paths(case, device):
    x, dt, A, B, C, chunk_size, use_final_grad = build_inputs(case, device)
    with torch.no_grad():
        out, final_state = ascend_kernel.mamba2_ssd_fwd(
            x, dt, A, B, C, chunk_size, return_final_state=True
        )
        dout = torch.full_like(out, 0.2)
        dfinal = torch.full_like(final_state, 0.1) if use_final_grad else None

    ref_values = tuple(
        value.detach().requires_grad_(True) for value in (x, dt, A, B, C)
    )
    ref_out, ref_final = ssd_chunk_scan_ref(
        *ref_values, chunk_size, return_final_state=True
    )
    ref_outputs = (ref_out, ref_final) if use_final_grad else (ref_out,)
    ref_grad_outputs = (dout, dfinal) if use_final_grad else (dout,)

    def custom_backward():
        result = torch.ops.mamba_ascend.mamba2_ssd_bwd(
            x, dt, A, B, C, dout, final_state,
            None, None, None, None, dfinal,
            False, 0.0, torch.finfo(torch.float32).max,
        )
        return result[0], result[1], result[2].sum(dim=0), result[3], result[4]

    def baseline_backward():
        return torch.autograd.grad(
            ref_outputs,
            ref_values,
            ref_grad_outputs,
            retain_graph=True,
            allow_unused=True,
        )

    return custom_backward, baseline_backward


def _sum_total_time(csv_path):
    total = 0.0
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        key = next(
            (name for name in reader.fieldnames or [] if "Total" in name and "us" in name),
            None,
        )
        if key is None:
            raise RuntimeError(f"Total Time(us) column missing: {csv_path}")
        for row in reader:
            if row.get(key, "").strip():
                total += float(row[key])
    return total


def _newest_csv(root):
    paths = glob.glob(os.path.join(root, "**", "op_statistic.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"op_statistic.csv not found under {root}")
    return max(paths, key=os.path.getmtime)


def profile_callable(fn, handler_dir):
    shutil.rmtree(handler_dir, ignore_errors=True)
    os.makedirs(handler_dir, exist_ok=True)
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(handler_dir),
        experimental_config=config,
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=WARMUP, active=ACTIVE, repeat=1, skip_first=0
        ),
    ) as profiler:
        for _ in range(WARMUP + ACTIVE):
            fn()
            profiler.step()
    torch.npu.synchronize()
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            csv_path = _newest_csv(handler_dir)
            return _sum_total_time(csv_path) / ACTIVE, csv_path
        except FileNotFoundError:
            time.sleep(0.2)
    raise FileNotFoundError(handler_dir)


def shape_label(case):
    specs = {item["name"]: item for item in case["inputs"]}
    x = specs["x"]["shape"]
    n = specs["B"]["shape"][-1]
    c = specs["chunk_size"]["value"]
    g = specs["B"]["shape"][2]
    return str([x[0], x[1], x[2], x[3], n, c, g])


def render_report(results, case_file, trace_root):
    ratios = [item["speedup"] for item in results]
    custom_better = sum(value > 1 for value in ratios)
    baseline_better = sum(value < 1 for value in ratios)
    lines = [
        "# 性能评估结果",
        "",
        "当前结果只评估 M0 correctness kernel 的 backward，不代表计划中的 M1 chunk/Cube 最终性能。",
        "无等价 NPU backward API；标杆是在同一 NPU 上由 `ssd_chunk_scan_ref` 的 PyTorch 基础张量算子及 autograd 组成的小算子路径。",
        "",
        f"- 用例文件：`{case_file}`",
        f"- Trace：`{trace_root}`",
        "- 固定 schedule：warmup=5、active=5、repeat=1。",
        "- 指标：每次 trace 的 `op_statistic.csv` 全部 `Total Time(us)` 求和后除以 5。",
        "",
        "## 性能对比",
        "",
        "| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |",
        "| ---- | ----- | ----- | ------------- | -------- | ------ |",
    ]
    for item in results:
        lines.append(
            f"| {item['case']} | {item['shape']} | {item['dtype']} | "
            f"{item['custom_us']:.3f} | {item['baseline_us']:.3f} | "
            f"{item['speedup']:.3f} |"
        )
    average = sum(ratios) / len(ratios)
    lines += [
        "",
        "## 全量汇总",
        "",
        "| 指标 | 值 |",
        "| ---- | -- |",
        f"| 用例数 | {len(results)} |",
        f"| 平均加速比（>1 表示自定义算子更快） | {average:.3f} |",
        f"| 自定义算子更优（比值>1） | {custom_better} |",
        f"| 标杆更优（比值<1） | {baseline_better} |",
        "",
        "### 按数据类型汇总",
        "",
        "| DType | 用例数 | 平均加速比 | 自定义算子更优 | 标杆更优 |",
        "| ----- | ------ | ---------- | -------------- | -------- |",
        f"| float32 | {len(results)} | {average:.3f} | {custom_better} | {baseline_better} |",
        "",
        "## 简短分析",
        "",
        "- M0 的目标是建立 native autograd correctness；结果只用于量化其性能债务。",
        "- M0 每个 group 内按 head 串行，并将全部 token state 写回 GM，规模增大时计算和搬运均线性增加。",
        "- PyTorch 标杆含多个 NPU kernel 与中间张量；自定义路径的优势主要来自融合，而非当前 Vector 标量循环的利用率。",
        "- M1 需要用 chunk/Cube GEMM、chunk state 重算和 Vector/Cube 流水替换 M0，之后才更新 README 的 backward 性能结论。",
        "",
    ]
    return "\n".join(lines)
