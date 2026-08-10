"""Generate the final, reproducible SVG figures embedded in README.md."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "benchmarks" / "results" / "readme_benchmarks.json"
ASSET_DIR = ROOT / "assets"

BLUE = "#2563eb"
RED = "#e11d48"
AMBER = "#f59e0b"
PURPLE = "#7c3aed"
GREEN = "#059669"
SLATE = "#64748b"
LIGHT = "#e2e8f0"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": SLATE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": LIGHT,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#0f172a",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "svg.hashsalt": "mamba2-readme-final",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = ASSET_DIR / name
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "mamba-ascendc"},
    )
    # Matplotlib writes trailing spaces in multiline SVG path data. Normalize the
    # generated asset so repeated figure generation keeps the Git diff clean.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    if preview_dir := os.environ.get("MAMBA_README_PREVIEW_DIR"):
        preview_path = Path(preview_dir)
        preview_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path / f"{Path(name).stem}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def training_gate(data: dict) -> None:
    gate = data["h256_training_gate"]
    a100 = gate["a100"]["forward_ms"]
    npu = gate["ascend_910b3"]["forward_ms"]
    throughput = a100 / npu * 100.0

    fig, (lat_ax, ratio_ax) = plt.subplots(1, 2, figsize=(10.8, 4.5))
    bars = lat_ax.bar(["A100 80GB", "Ascend 910B3\nAscendC"], [a100, npu], color=[BLUE, RED], width=0.58)
    lat_ax.bar_label(bars, fmt="%.2f ms", padding=3, fontsize=9.0)
    lat_ax.set_ylabel("Median device latency (ms)")
    lat_ax.set_title("Measured forward latency")
    lat_ax.set_ylim(0, npu * 1.22)
    lat_ax.spines[["top", "right"]].set_visible(False)

    bars = ratio_ax.bar(["Ascend 910B3 / A100"], [throughput], color=RED, width=0.55)
    ratio_ax.axhline(70, color=SLATE, linestyle="--", linewidth=1.1)
    ratio_ax.axhline(75, color="#94a3b8", linestyle=":", linewidth=1.1)
    ratio_ax.bar_label(bars, labels=[f"{throughput:.2f}%"], padding=3, fontweight="bold")
    ratio_ax.set_ylim(0, 86)
    ratio_ax.set_ylabel("Throughput relative to A100 (%)")
    ratio_ax.set_title("Same workload · no TFLOPS normalization")
    ratio_ax.text(0.98, 70 / 86 - 0.012, "70% gate", transform=ratio_ax.transAxes, ha="right", va="top", fontsize=8)
    ratio_ax.text(0.98, 75 / 86 + 0.012, "75% stretch", transform=ratio_ax.transAxes, ha="right", va="bottom", fontsize=8)
    ratio_ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Mamba-2 SSD H256 training · A100 vs Ascend 910B3", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Forward · B=8 · L=4096 · H=256 · P=N=chunk=64 · G=64 · FP32 public API · device events · p50",
        ha="center",
        fontsize=8.4,
        color=SLATE,
    )
    fig.subplots_adjust(bottom=0.16, top=0.82, wspace=0.29)
    save(fig, "h256-training-gate.svg")


def pipeline(data: dict) -> None:
    profile = data["forward_profile"]
    colors = {"Preprocess": AMBER, "ChunkMix": RED, "Transpose": "#38bdf8", "StateEpilogue": PURPLE}
    fig, (time_ax, flow_ax) = plt.subplots(
        2, 1, figsize=(10.8, 6.0), gridspec_kw={"height_ratios": [1.05, 1.25]}
    )
    step = profile["representative_step"]
    for item in step:
        time_ax.barh(0, item["duration_ms"], left=item["start_ms"], height=0.55, color=colors[item["name"]])
        center = item["start_ms"] + item["duration_ms"] / 2
        if item["duration_ms"] < 1.0:
            time_ax.annotate(
                f"{item['name']} · {item['duration_ms']:.3f} ms",
                xy=(center, -0.28),
                xytext=(center, -0.46),
                ha="center",
                va="top",
                fontsize=7.8,
                fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": SLATE, "linewidth": 0.8},
            )
        else:
            time_ax.text(
                center,
                0,
                f"{item['name']}\n{item['duration_ms']:.3f} ms",
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="bold",
                color="white",
            )
    time_ax.set_yticks([])
    time_ax.set_xlim(0, profile["representative_service_ms"])
    time_ax.set_xlabel("Relative kernel start time (ms)")
    time_ax.set_title("Real msprof timeline · representative active step")
    time_ax.spines[["top", "right", "left"]].set_visible(False)
    time_ax.grid(axis="x")
    time_ax.set_ylim(-0.58, 0.38)

    flow_ax.set_axis_off()
    flow = [
        ("FP32 inputs", "public layout", LIGHT),
        ("Preprocess", "AIV\nnonlinearity + layout", AMBER),
        ("ChunkMix", "AIC + AIV\nchunk-local GEMMs", RED),
        ("Transpose", "consumer layout", "#38bdf8"),
        ("StateEpilogue", "AIC + AIV\nstate + D/z", PURPLE),
        ("Output", "FP32 + final state", LIGHT),
    ]
    centers = np.linspace(0.075, 0.925, len(flow))
    width, height = 0.135, 0.46
    for index, (center, (title, detail, color)) in enumerate(zip(centers, flow)):
        flow_ax.add_patch(
            FancyBboxPatch(
                (center - width / 2, 0.32),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                transform=flow_ax.transAxes,
                facecolor=color,
                edgecolor=SLATE,
                linewidth=0.8,
            )
        )
        text_color = "white" if color in (RED, PURPLE) else "#0f172a"
        flow_ax.text(center, 0.63, title, transform=flow_ax.transAxes, ha="center", fontweight="bold", color=text_color)
        flow_ax.text(center, 0.44, detail, transform=flow_ax.transAxes, ha="center", fontsize=7.8, color=text_color)
        if index + 1 < len(flow):
            flow_ax.add_patch(
                FancyArrowPatch(
                    (center + width / 2, 0.55),
                    (centers[index + 1] - width / 2, 0.55),
                    transform=flow_ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.0,
                    color=SLATE,
                )
            )
    flow_ax.text(0.5, 0.92, "AscendC staged dataflow", transform=flow_ax.transAxes, ha="center", fontsize=11.5, fontweight="bold")
    flow_ax.text(
        0.5,
        0.10,
        "AIV producers and AIC consumers overlap inside MIX kernels; cross-kernel workspaces remain the main data-movement cost.",
        transform=flow_ax.transAxes,
        ha="center",
        fontsize=8.3,
        color=SLATE,
    )
    fig.suptitle("Mamba-2 SSD H256 forward · pipeline and measured execution", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        f"5 active steps · profiler mean {profile['profile_service_ms']:.3f} ms · device-event p50 {profile['event_ms']:.3f} ms",
        ha="center",
        fontsize=8.5,
        color=SLATE,
    )
    fig.subplots_adjust(bottom=0.07, top=0.88, hspace=0.42)
    save(fig, "ascendc-pipeline.svg")


def head_scaling(data: dict) -> None:
    rows = data["head_scaling"]
    heads = np.array([row["heads"] for row in rows])
    a100 = np.array([row["a100_forward_ms"] for row in rows])
    npu = np.array([row["ascend_forward_ms"] for row in rows])
    ratio = a100 / npu * 100.0
    fig, (lat_ax, ratio_ax) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    lat_ax.plot(heads, a100, marker="o", linewidth=2.4, color=BLUE, label="A100 80GB")
    lat_ax.plot(heads, npu, marker="o", linewidth=2.4, color=RED, label="Ascend 910B3")
    lat_ax.set_xscale("log", base=2)
    lat_ax.set_yscale("log", base=2)
    lat_ax.set_xticks(heads, [str(value) for value in heads])
    lat_ax.set_xlabel("Number of heads H")
    lat_ax.set_ylabel("Median forward latency (ms, log₂)")
    lat_ax.set_title("Measured latency")
    lat_ax.legend(loc="upper left", fontsize=8.3)
    lat_ax.spines[["top", "right"]].set_visible(False)
    ratio_ax.plot(heads, ratio, marker="o", linewidth=2.4, color=RED)
    ratio_ax.axhline(70, color=SLATE, linestyle="--", linewidth=1.0)
    ratio_ax.set_xscale("log", base=2)
    ratio_ax.set_xticks(heads, [str(value) for value in heads])
    ratio_ax.set_ylim(60, 75)
    ratio_ax.set_xlabel("Number of heads H")
    ratio_ax.set_ylabel("910B3/A100 throughput (%)")
    ratio_ax.set_title("Relative sustained throughput")
    ratio_ax.spines[["top", "right"]].set_visible(False)
    for h, value in zip(heads, ratio):
        ratio_ax.annotate(f"{value:.1f}%", (h, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.0)
    fig.suptitle("Mamba-2 SSD forward heavy-shape scaling", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "B=8 · L=4096 · P=N=chunk=64 · H/G=4 · full optional inputs · final state · device events · p50",
        ha="center",
        fontsize=8.4,
        color=SLATE,
    )
    fig.subplots_adjust(bottom=0.16, top=0.82, wspace=0.30)
    save(fig, "head-scaling.svg")


def utilization(data: dict) -> None:
    stages = data["forward_profile"]["stages"]
    metrics = [
        ("Cube active", "cube_active_percent"),
        ("AIC MAC", "aic_mac_percent"),
        ("AIV vector", "aiv_vector_percent"),
        ("AIV scalar", "aiv_scalar_percent"),
        ("AIV MTE2", "aiv_mte2_percent"),
        ("AIV MTE3", "aiv_mte3_percent"),
    ]
    values = np.array([[stage[key] for stage in stages] for _, key in metrics], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(stages)), [stage["name"] for stage in stages])
    ax.set_yticks(np.arange(len(metrics)), [label for label, _ in metrics])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(column, row, f"{value:.1f}%" if value else "—", ha="center", va="center", color="white" if value >= 55 else "#0f172a", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04).set_label("msprof activity ratio (%)")
    ax.set_title("High Cube activity does not imply high effective MAC")
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    fig.text(
        0.5,
        0.015,
        "Ratios may overlap because AIC/AIV/MTE pipelines execute concurrently; they are not occupancy percentages that sum to 100%.",
        ha="center",
        fontsize=8.3,
        color=SLATE,
    )
    fig.subplots_adjust(bottom=0.12, top=0.90)
    save(fig, "hardware-utilization.svg")


def framework_comparison(data: dict) -> None:
    rows = data["framework_comparison"]
    labels = [row["backend"] for row in rows]
    latency = np.array([row["median_ms"] for row in rows])
    launches = np.array([row["kernel_launches"] for row in rows])
    colors = ["#94a3b8", AMBER, RED]
    fig, (lat_ax, launch_ax) = plt.subplots(1, 2, figsize=(10.3, 4.4))
    bars = lat_ax.bar(labels, latency, color=colors)
    lat_ax.set_yscale("log")
    lat_ax.bar_label(bars, fmt="%.3f ms", padding=3, fontsize=8.4)
    lat_ax.set_ylabel("Median device latency (ms, log)")
    lat_ax.set_title("Same 910B3 and shape")
    lat_ax.spines[["top", "right"]].set_visible(False)
    bars = launch_ax.bar(labels, launches, color=colors)
    launch_ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8.4)
    launch_ax.set_ylabel("Device kernels / forward")
    launch_ax.set_title("Execution decomposition")
    launch_ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PyTorch composition → Triton-Ascend → AscendC", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "[4,2048,16,64,128,128,4] · D/z/dt-bias/softplus · device events; launches from 5 profiler steps", ha="center", fontsize=8.3, color=SLATE)
    fig.subplots_adjust(bottom=0.15, top=0.82, wspace=0.30)
    save(fig, "framework-decomposition.svg")


def backward_breakdown(data: dict) -> None:
    profile = data["backward_profile"]
    stages = profile["stages"]
    names = [row["name"] for row in stages][::-1]
    values = np.array([row["duration_ms"] for row in stages][::-1])
    colors = ([BLUE, RED, PURPLE, AMBER, "#0ea5e9"] + ["#94a3b8"] * len(stages))[: len(stages)][::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    bars = ax.barh(names, values, color=colors)
    ax.bar_label(bars, labels=[f"{v:.2f} ms  ({v / profile['profile_ms']:.1%})" for v in values], padding=4, fontsize=8.2)
    ax.set_xlim(0, values.max() * 1.38)
    ax.set_xlabel("Mean device time per backward (ms)")
    ax.set_title("Ascend 910B3 H256 backward · Level1 stage breakdown")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x")
    fig.text(0.5, 0.012, f"Profiler kernel mean {profile['profile_ms']:.2f} ms · device-event p50 {profile['event_ms']:.2f} ms", ha="center", fontsize=8.4, color=SLATE)
    fig.subplots_adjust(left=0.25, right=0.89, bottom=0.13, top=0.88)
    save(fig, "backward-breakdown.svg")


def data_movement(data: dict) -> None:
    traffic = data["forward_profile"]["logical_intermediate_traffic"]
    names = [row["name"] for row in traffic]
    values = np.array([row["write_read_mib"] for row in traffic], dtype=float)
    colors = [AMBER, "#94a3b8", "#38bdf8", BLUE, RED, PURPLE]
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    left = 0.0
    for name, value, color in zip(names, values, colors):
        ax.barh([0], [value], left=left, height=0.48, color=color)
        label = f"{name}\n{value / 1024:.2f} GiB" if value >= 512 else f"{name}\n{value:.0f} MiB"
        if value >= 512:
            ax.text(left + value / 2, 0, label, ha="center", va="center", fontsize=8.2, fontweight="bold", color="white" if color in (BLUE, RED, PURPLE) else "#0f172a")
        left += value
    ax.set_xlim(0, values.sum())
    ax.set_yticks([])
    ax.set_xlabel("Logical cross-kernel traffic per forward (write + reread, MiB)")
    ax.set_title(f"H256 intermediates: {values.sum() / 1024:.2f} GiB logical materialization")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x")
    ax.text(0.0, -0.34, "Shape-derived lower bound, not an HBM hardware counter. y_diag + chunk_state contribute 66.3%.", transform=ax.transAxes, fontsize=8.5, color=SLATE)
    fig.subplots_adjust(bottom=0.30, top=0.84)
    save(fig, "data-movement.svg")


def vmamba_end_to_end(data: dict) -> None:
    result = data["vmamba2_end_to_end"]
    labels = ["Full network", "Mamba mixers", "SSD cores"]
    a100 = np.array([result["a100"]["full_network_ms"], result["a100"]["mamba_mixers_ms"], result["a100"]["ssd_cores_ms"]])
    npu = np.array([result["ascend_910b3"]["full_network_ms"], result["ascend_910b3"]["mamba_mixers_ms"], result["ascend_910b3"]["ssd_cores_ms"]])
    x, width = np.arange(3), 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    a100_bars = ax.bar(x - width / 2, a100, width, color=BLUE, label="A100 80GB")
    npu_bars = ax.bar(x + width / 2, npu, width, color=RED, label="Ascend 910B3")
    ax.bar_label(a100_bars, fmt="%.0f", padding=2, fontsize=8.3)
    ax.bar_label(npu_bars, fmt="%.0f", padding=2, fontsize=8.3)
    for index, ratio in enumerate(a100 / npu):
        ax.text(index, max(a100[index], npu[index]) + npu.max() * 0.05, f"{ratio:.2f}×", ha="center", fontweight="bold", fontsize=8.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median latency per batch (ms)")
    ax.set_ylim(0, npu.max() * 1.20)
    ax.set_title("Pure Vision-Mamba2 forward · batch 32 · 1024²")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.0, -0.19, "12 blocks × 4 scan directions = 48 SSD calls; component timings are instrumented independently and are not additive.", transform=ax.transAxes, fontsize=8.3, color=SLATE)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "vmamba2-end-to-end.svg")


def main() -> None:
    configure_style()
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    training_gate(data)
    pipeline(data)
    head_scaling(data)
    utilization(data)
    framework_comparison(data)
    backward_breakdown(data)
    data_movement(data)
    vmamba_end_to_end(data)


if __name__ == "__main__":
    main()
