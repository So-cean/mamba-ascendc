"""torch_npu.profiler helpers for reverse chunk-state recurrence."""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import time

import torch
import torch_npu

import ascend_kernel  # noqa: F401


WARMUP = 5
ACTIVE = 5


def load_cases(path):
    if not path.endswith(".jsonl"):
        raise ValueError("performance cases must be JSONL")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_paths(case, device):
    specs = {item["name"]: item for item in case["inputs"]}
    generator = torch.Generator(device="cpu").manual_seed(20260807)

    def positive(shape):
        return (0.02 + 0.08 * torch.rand(shape, generator=generator)).to(device)

    states_start = positive(specs["states_start"]["shape"])
    d_states_start = positive(specs["d_states_start"]["shape"])
    d_a_cumsum = -0.01 - positive(specs["dA_cumsum"]["shape"])
    dfinal_state = (
        positive(states_start.shape[:2] + states_start.shape[3:])
        if specs["use_dfinal"]["value"]
        else None
    )

    def custom():
        return torch.ops.mamba_ascend.mamba2_ssd_state_passing_bwd(
            states_start, d_states_start, d_a_cumsum, dfinal_state
        )

    def baseline():
        gstate = (
            torch.zeros_like(states_start[:, :, 0])
            if dfinal_state is None
            else dfinal_state
        )
        d_u_reverse = []
        d_a_reverse = []
        for chunk in range(states_start.shape[2] - 1, -1, -1):
            alpha = torch.exp(d_a_cumsum[:, :, chunk, -1])
            d_u_reverse.append(gstate)
            d_a_reverse.append(
                (gstate * states_start[:, :, chunk]).sum(dim=(-2, -1))
                * alpha
            )
            gstate = (
                d_states_start[:, :, chunk]
                + alpha[..., None, None] * gstate
            )
        return (
            torch.stack(d_u_reverse[::-1], dim=2),
            gstate,
            torch.stack(d_a_reverse[::-1], dim=2),
        )

    return custom, baseline


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
        raise FileNotFoundError(root)
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
    state = specs["states_start"]["shape"]
    chunk_size = specs["dA_cumsum"]["shape"][-1]
    return str(state + [chunk_size])


def render_report(results, case_file, trace_root):
    ratios = [item["speedup"] for item in results]
    custom_better = sum(value > 1 for value in ratios)
    baseline_better = sum(value < 1 for value in ratios)
    average = sum(ratios) / len(ratios)
    lines = [
        "# 性能评估结果",
        "",
        "评估对象是完整 Mamba2 backward M1 的内部 reverse state-passing 子算子。",
        "无等价 NPU API；标杆路径由同一 NPU 上的 `exp/mul/sum/add/stack` 基础张量算子组成。",
        "",
        f"- 用例文件：`{case_file}`",
        f"- Trace：`{trace_root}`",
        "- 固定 schedule：warmup=5、active=5、repeat=1。",
        "- 指标：`op_statistic.csv` 全部 `Total Time(us)` 求和后除以 5。",
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
        "- Native kernel 将跨 chunk 的 state gradient 保留在 UB，并把 eager 路径的多次 launch 融合成一次。",
        "- `B*H` 决定可并行 stream 数，`K` 决定每条 stream 内不可并行的 reverse recurrence 长度。",
        "- `N=128` 时单 stream 的搬运与 Vector reduction 工作量约为 `N=64` 的两倍。",
        "- 该数据只评价 state-passing 子图；完整 backward 还包含两侧 Cube GEMM 和 dt/A reduction。",
        "",
    ]
    return "\n".join(lines)
