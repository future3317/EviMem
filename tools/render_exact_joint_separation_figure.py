"""Render the exact state-feedback separation figure outside the code repo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matmem.exact_joint_separation import evaluate_exact_joint_separation


def _annotate_bar_values(axis: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _draw_action_path(axis: plt.Axes) -> None:
    axis.axis("off")
    nodes = {
        "start": (0.08, 0.53, "Start\nB=3, K=1"),
        "probe": (0.27, 0.72, "witness p\nreveals w"),
        "retain": (0.49, 0.72, "retain w\n(K=1)"),
        "groups": (0.72, 0.72, "g[w,a],\ng[w,b]"),
        "joint": (0.94, 0.72, "T=2"),
        "safe": (0.27, 0.25, "nonadaptive:\nsafe s"),
        "fixed": (0.60, 0.25, "fixed group pair\n(no state feedback)"),
        "nonadaptive": (0.91, 0.25, "E[T]=5/3"),
    }
    for key, (x, y, label) in nodes.items():
        facecolor = "#dbeafe" if key in {"probe", "retain", "groups", "joint"} else "#f3f4f6"
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.24", "fc": facecolor, "ec": "#4b5563", "lw": 0.8},
        )
    arrows = (
        ("start", "probe"),
        ("probe", "retain"),
        ("retain", "groups"),
        ("groups", "joint"),
        ("start", "safe"),
        ("safe", "fixed"),
        ("fixed", "nonadaptive"),
    )
    for source, target in arrows:
        x1, y1, _ = nodes[source]
        x2, y2, _ = nodes[target]
        axis.annotate(
            "",
            xy=(x2 - 0.06, y2),
            xytext=(x1 + 0.06, y1),
            arrowprops={"arrowstyle": "->", "color": "#4b5563", "lw": 1.0},
        )
    axis.text(
        0.5,
        0.98,
        "Reveal → retain → next action: state feedback is the only separation",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        transform=axis.transAxes,
    )


def render(output: Path) -> None:
    report = evaluate_exact_joint_separation()
    # Keep the theorem graphic compact enough to share a manuscript page with
    # its interpretation, while retaining readable labels at one-column width.
    figure = plt.figure(figsize=(7.0, 3.9), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(0.57, 1.0))
    path_axis = figure.add_subplot(grid[0, :])
    metric_axis = figure.add_subplot(grid[1, 0])
    null_axis = figure.add_subplot(grid[1, 1])

    _draw_action_path(path_axis)

    policies = ("State-feedback", "Best nonadaptive", "Myopic")
    evaluations = (report.joint, report.nonadaptive, report.myopic)
    metrics = (
        ("Terminal", [item.terminal_confirmations for item in evaluations], "#2563eb"),
        ("Final-causal", [item.final_causal_confirmations for item in evaluations], "#64748b"),
        ("False-stable", [item.false_stable for item in evaluations], "#dc2626"),
    )
    offsets = np.asarray([-0.24, 0.0, 0.24])
    positions = np.arange(len(policies))
    for (label, values, color), offset in zip(metrics, offsets, strict=True):
        bars = metric_axis.bar(positions + offset, values, width=0.23, label=label, color=color)
        _annotate_bar_values(metric_axis, list(bars))
    metric_axis.set_xticks(positions, policies, fontsize=8)
    metric_axis.set_ylim(0, 3.45)
    metric_axis.set_ylabel("Expected count per campaign")
    metric_axis.set_title("Exact objective decomposition", fontsize=10, fontweight="bold")
    metric_axis.legend(frameon=False, fontsize=8, loc="upper left")
    metric_axis.grid(axis="y", alpha=0.25)

    null_names = (
        "Joint\nK=1",
        "K=0",
        "Full\nhistory",
        "Free\naccess",
        "No\nsignal",
        "Protocol\nabstain",
    )
    null_values = (
        report.joint.terminal_confirmations,
        report.zero_memory.terminal_confirmations,
        report.full_history.terminal_confirmations,
        report.zero_access_cost.terminal_confirmations,
        report.uninformative_witness.terminal_confirmations,
        report.unsupported_witness.terminal_confirmations,
    )
    colors = ["#2563eb", "#94a3b8", "#2563eb", "#2563eb", "#94a3b8", "#94a3b8"]
    bars = null_axis.bar(np.arange(len(null_names)), null_values, color=colors)
    _annotate_bar_values(null_axis, list(bars))
    null_axis.axhline(
        report.nonadaptive.terminal_confirmations,
        color="#475569",
        lw=1.0,
        ls="--",
        label="Best nonadaptive = 5/3",
    )
    null_axis.set_xticks(np.arange(len(null_names)), null_names, fontsize=7)
    null_axis.set_ylim(0, 2.35)
    null_axis.set_ylabel("Expected terminal confirmations")
    null_axis.set_title("Registered null and safety checks", fontsize=10, fontweight="bold")
    null_axis.legend(frameon=False, fontsize=8, loc="upper right")
    null_axis.grid(axis="y", alpha=0.25)

    for axis in (metric_axis, null_axis):
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
