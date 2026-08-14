"""Render paper figures from frozen IC-SARR summaries and exact synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from publication_figure_style import PALETTE, apply_publication_style, finalize_figure

from matmem.controlled_delayed_label_benchmark import controlled_benchmark_grid

INK = PALETTE["charcoal"]
BLUE = PALETTE["blue_main"]
GREEN = PALETTE["teal"]
RED = PALETTE["red_strong"]
GRAY = PALETTE["neutral_dark"]


def _save(fig: plt.Figure, path: Path) -> None:
    finalize_figure(fig, path)


def delayed_adjudication(path: Path) -> None:
    apply_publication_style(7.4)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 3.85), facecolor="white")
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    def box(axis, x, y, text, *, edge=BLUE, fill="#EAF2F8", size=6.3):
        axis.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            bbox={"boxstyle": "round,pad=0.30", "fc": fill, "ec": edge, "lw": 0.8},
        )

    ax_a.axis("off")
    ax_a.set(xlim=(0, 1), ylim=(0, 1))
    ax_a.text(0.02, 0.94, "a", fontsize=8.5, fontweight="bold", transform=ax_a.transAxes)
    ax_a.text(0.50, 0.92, "Ordinary active search", ha="center", fontsize=7.2, color=GRAY)
    ax_a.text(0.50, 0.55, "Delayed structured-label search", ha="center", fontsize=7.2, color=BLUE)
    box(ax_a, 0.18, 0.78, "query $x$", edge=GRAY, fill="#F3F3F3")
    box(ax_a, 0.82, 0.78, "observe $Y_x$", edge=GRAY, fill="#F3F3F3")
    ax_a.annotate("", xy=(0.69, 0.78), xytext=(0.31, 0.78), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": GRAY})
    box(ax_a, 0.18, 0.38, "query $x$", edge=BLUE)
    box(ax_a, 0.50, 0.38, "observe $E_T(x)$", edge=BLUE)
    box(ax_a, 0.82, 0.38, "adjudicate $Y_x$\non complete pool", edge=RED, fill="#F8ECEB", size=5.8)
    ax_a.annotate("", xy=(0.40, 0.38), xytext=(0.29, 0.38), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": BLUE})
    ax_a.annotate("", xy=(0.72, 0.38), xytext=(0.61, 0.38), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": RED})
    ax_a.text(0.50, 0.16, "Immediate numeric observation; delayed globally coupled label", ha="center", fontsize=6.2, color=INK)

    ax_b.set(xlim=(-0.05, 1.05), ylim=(-0.32, 0.16))
    ax_b.axis("off")
    ax_b.text(0.02, 0.94, "b", fontsize=8.5, fontweight="bold", transform=ax_b.transAxes)
    comp = np.array([0.0, 0.34, 0.66, 1.0])
    ax_b.plot([0, 1], [0, 0], color=INK, lw=1.0)
    ax_b.scatter([0, 1], [0, 0], color=INK, s=18, zorder=3)
    ax_b.scatter([comp[1]], [-0.05], color=BLUE, s=32, zorder=4)
    ax_b.plot([0, comp[1], 1], [0, -0.05, 0], color=BLUE, lw=1.6, ls="--")
    ax_b.text(comp[1], -0.11, "queried $x$\nprovisional", ha="center", color=BLUE, fontsize=6.2)
    ax_b.scatter([comp[2]], [-0.16], color=RED, s=32, zorder=4)
    ax_b.plot([0, comp[2], 1], [0, -0.16, 0], color=RED, lw=1.6)
    ax_b.text(comp[2], -0.23, "unqueried $y$\nrevokes $x$", ha="center", color=RED, fontsize=6.2)
    ax_b.text(0.50, 0.10, "provisional hull", ha="center", color=BLUE, fontsize=6.4)
    ax_b.text(0.50, -0.30, "final complete-pool hull", ha="center", color=RED, fontsize=6.4)
    ax_b.set_xlabel("composition", labelpad=1)

    ax_c.axis("off")
    ax_c.set(xlim=(0, 1), ylim=(0, 1))
    ax_c.text(0.02, 0.94, "c", fontsize=8.5, fontweight="bold", transform=ax_c.transAxes)
    for y, label, color, fill in ((0.73, "$D$: reveal-time", BLUE, "#EAF2F8"), (0.51, "$F$: selected-history", GRAY, "#F3F3F3"), (0.29, "$T$: complete-pool", RED, "#F8ECEB")):
        box(ax_c, 0.52, y, label, edge=color, fill=fill)
    ax_c.annotate("", xy=(0.52, 0.60), xytext=(0.52, 0.69), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": GRAY})
    ax_c.annotate("", xy=(0.52, 0.38), xytext=(0.52, 0.47), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": GRAY})
    ax_c.text(0.13, 0.61, "$D-F$\nwithin-campaign", ha="center", va="center", fontsize=6.2, color=GRAY)
    ax_c.text(0.13, 0.38, "$F-T$\nunqueried competitors", ha="center", va="center", fontsize=6.2, color=RED)
    ax_c.text(0.50, 0.10, "Reward is $T$; $D$ and $F$ diagnose when\nprovisional discoveries are revoked.", ha="center", fontsize=6.0, color=INK)

    ax_d.axis("off")
    ax_d.set(xlim=(0, 1), ylim=(0, 1))
    ax_d.text(0.02, 0.94, "d", fontsize=8.5, fontweight="bold", transform=ax_d.transAxes)
    box(ax_d, 0.20, 0.53, "Target margin\n(proxy objective)", edge=GRAY, fill="#F3F3F3", size=6.0)
    box(ax_d, 0.50, 0.53, "Delta-Hull\n(complete-pool greedy)", edge=BLUE, fill="#EAF2F8", size=6.0)
    box(ax_d, 0.80, 0.53, "Lookahead\n(anchored rollout)", edge=RED, fill="#F8ECEB", size=6.0)
    ax_d.annotate("", xy=(0.40, 0.53), xytext=(0.30, 0.53), arrowprops={"arrowstyle": "->", "lw": 0.9, "color": BLUE})
    ax_d.annotate("", xy=(0.70, 0.53), xytext=(0.60, 0.53), arrowprops={"arrowstyle": "->", "lw": 0.9, "color": RED})
    ax_d.text(0.35, 0.70, "change objective", ha="center", fontsize=6.2, color=BLUE)
    ax_d.text(0.65, 0.70, "change solver", ha="center", fontsize=6.2, color=RED)
    ax_d.text(0.50, 0.22, "The study asks whether objective alignment\nmatters more than extra planning.", ha="center", fontsize=6.2, color=INK)

    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.96, wspace=0.16, hspace=0.18)
    _save(fig, path)


def rollout_method(path: Path) -> None:
    """Show common-world first-action branching and the IC-SARR fallback."""
    fig, axis = plt.subplots(figsize=(7.0, 2.05), constrained_layout=True)
    axis.set(xlim=(-0.01, 1.18), ylim=(0, 1))
    axis.axis("off")
    def box(x: float, y: float, text: str, *, color: str = BLUE, fill: str = "#EAF2F8", size: float = 7.0) -> None:
        axis.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, bbox={"boxstyle": "round,pad=0.32", "fc": fill, "ec": color, "lw": 0.9})

    box(0.07, 0.52, "Legal state\n$O_t$", size=6.2)
    box(0.22, 0.52, "Shared joint worlds\n$e^{(1)},\\ldots,e^{(M)}$", size=5.9)
    axis.annotate("", xy=(0.185, 0.54), xytext=(0.13, 0.54), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.9})
    branch_y = (0.74, 0.52, 0.30)
    for y, candidate in zip(branch_y, ("$x_1$", "$x_2$", "$x_n$"), strict=True):
        axis.annotate("", xy=(0.39, y), xytext=(0.32, 0.54), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8})
        box(0.40, y, f"first action\n{candidate}", size=5.8)
        box(0.56, y, "causal reveal\n+ source rollout", size=5.6)
        axis.annotate("", xy=(0.49, y), xytext=(0.445, y), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8})
        axis.annotate("", xy=(0.64, y), xytext=(0.615, y), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8})
    axis.text(0.56, 0.91, "Every legal first action uses the same posterior worlds and paired terminal returns", fontsize=5.8, ha="center", color=GRAY)
    box(0.73, 0.52, "paired $\\Delta Q$\nvs. source", color=GREEN, fill="#E9F5EE", size=5.7)
    for y in branch_y:
        axis.annotate("", xy=(0.685, 0.52), xytext=(0.64, y), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.75})
    box(0.87, 0.52, "two-stage\nlower bound", color=GREEN, fill="#E9F5EE", size=5.7)
    axis.annotate("", xy=(0.815, 0.52), xytext=(0.785, 0.52), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.85})
    axis.annotate("", xy=(0.95, 0.74), xytext=(0.925, 0.59), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 0.9})
    axis.annotate("", xy=(0.95, 0.30), xytext=(0.925, 0.45), arrowprops={"arrowstyle": "->", "color": RED, "lw": 0.9})
    axis.text(0.97, 0.75, "$L>0$:\nrollout", ha="left", va="center", fontsize=5.9, color=GREEN)
    axis.text(0.97, 0.29, "$L\\leq0$:\nsource", ha="left", va="center", fontsize=5.9, color=RED)
    axis.text(0.50, 0.06, "Sampled reveal + fixed source continuation + terminal complete-pool reward; independent stream only for unresolved stage-two candidates", ha="center", fontsize=5.2, color=INK)
    _save(fig, path)


def controlled_and_folds(path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    rows = [r for r in controlled_benchmark_grid() if r["source_signal"] == 0.5 and r["coupling"] == 1.0]
    budgets = [r["budget"] for r in rows]
    for key, label, color, marker in (("source_margin", "Source margin", GRAY, "o"), ("greedy_final", "Greedy final", BLUE, "s"), ("gated_source_rollout", "Source-anchored rollout", GREEN, "D"), ("optimal_dp", "Exact DP", RED, "^")):
        left.plot(budgets, [r[key] for r in rows], label=label, color=color, marker=marker, lw=1.5, ms=4)
    left.set(title="Exact controlled benchmark", xlabel="Budget", ylabel="Expected durable discoveries", xticks=[1, 2, 3])
    left.legend(frameon=False, fontsize=6.8)
    left.grid(axis="y", alpha=0.25)
    folds = np.arange(1, 6)
    effect = np.array([0.174, 0.196, 0.196, 0.130, 0.109])
    right.axvline(0, color=GRAY, lw=0.8)
    right.scatter(effect, folds, color=BLUE, s=33, zorder=3)
    right.errorbar(.161, 6, xerr=[[.161-.083], [.239-.161]], fmt="D", color=GREEN, capsize=3, label="All 230: 95% CI")
    right.set(
        yticks=[1, 2, 3, 4, 5, 6],
        yticklabels=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "All 230"],
        xlabel="IC-SARR minus source\nfull-pool confirmations/system",
        title="Cross-fitted MatPES effect",
    )
    right.set_ylim(6.5, 0.5)
    right.grid(axis="x", alpha=0.25)
    right.legend(frameon=False, fontsize=6.8, loc="lower right")
    _save(fig, path)


def dft_waterfall(path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 3.35), sharex=True, sharey=True, constrained_layout=True)
    values = {
        "Source margin": (4.322, 4.083, 3.622),
        "IC-SARR": (4.643, 4.096, 3.783),
    }
    labels = ("Provisional $D$", "$D-F$", "Selected-history $F$", "$F-T$", "Complete-pool $T$")
    for axis, (policy, counts) in zip(axes, values.items(), strict=True):
        d_value, f_value, t_value = counts
        x = np.arange(5)
        starts = (0.0, f_value, 0.0, t_value, 0.0)
        heights = (d_value, d_value - f_value, f_value, f_value - t_value, t_value)
        axis.bar(x, heights, bottom=starts, color=(BLUE, RED, BLUE, RED, GREEN), width=0.62, edgecolor="white", linewidth=0.55)
        axis.plot([0.31, 0.69], [d_value, d_value], color=GRAY, lw=0.8)
        axis.plot([1.31, 1.69], [f_value, f_value], color=GRAY, lw=0.8)
        axis.plot([2.31, 2.69], [f_value, f_value], color=GRAY, lw=0.8)
        axis.plot([3.31, 3.69], [t_value, t_value], color=GRAY, lw=0.8)
        axis.set_ylabel("Confirmations/system")
        axis.set_title(policy, loc="left", fontsize=9, fontweight="bold")
        axis.set_ylim(0, 5.25)
        axis.grid(axis="y", alpha=0.22)
        for index, (start, height) in enumerate(zip(starts, heights, strict=True)):
            if index in (0, 2, 4):
                axis.text(index, start + height + 0.10, f"{start + height:.3f}", ha="center", fontsize=7.0)
            else:
                axis.text(index, start + height / 2, f"−{height:.3f}", ha="center", va="center", fontsize=6.8, color="white")
    axes[-1].set_xticks(np.arange(5), labels, fontsize=6.8)
    fig.text(0.50, 0.50, "$\\Delta D=+0.322$    $\\Delta F=+0.013$    $\\Delta T=+0.161$    $\\Delta(F-T)=-0.148$", ha="center", va="center", fontsize=6.5, color=INK)
    _save(fig, path)


def controlled_grid_efficiency(path: Path) -> None:
    """Render the main exact regret heatmaps for three policy rows."""
    fig, axes = plt.subplots(3, 3, figsize=(6.8, 5.05), sharex=True, sharey=True, constrained_layout=True)
    rows = controlled_benchmark_grid()
    policies = (("source_margin", "Source margin"), ("greedy_final", "Greedy final"), ("gated_source_rollout", "Source-anchored rollout"))
    source_signal = (0.0, 0.5, 1.0)
    matrices: list[np.ndarray] = []
    for policy, _ in policies:
        for signal in source_signal:
            matrix = np.zeros((3, 3), dtype=float)
            for item in rows:
                if item["source_signal"] == signal:
                    budget_index = int(item["budget"]) - 1
                    coupling_index = int(round(item["coupling"] * 2))
                    matrix[budget_index, coupling_index] = item["optimal_dp"] - item[policy]
            matrices.append(matrix)
    vmax = max(float(matrix.max()) for matrix in matrices)
    image = None
    for row_index, (policy, label) in enumerate(policies):
        for column_index, signal in enumerate(source_signal):
            axis = axes[row_index, column_index]
            matrix = matrices[row_index * 3 + column_index]
            image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, origin="lower", aspect="auto")
            for budget_index in range(3):
                for coupling_index in range(3):
                    value = matrix[budget_index, coupling_index]
                    axis.text(coupling_index, budget_index, f"{value:.2f}", ha="center", va="center", fontsize=6.9, color="white" if value > vmax * 0.55 else INK)
            axis.set(xticks=[0, 1, 2], xticklabels=["0", "0.5", "1"], yticks=[0, 1, 2], yticklabels=["1", "2", "3"])
            if row_index == 0:
                axis.set_title(f"Source signal {signal:.1f}", fontsize=8)
            if column_index == 0:
                axis.set_ylabel(f"{label}\nBudget $B$", fontsize=7.5)
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes[:3, :], shrink=0.72, pad=0.015)
    colorbar.set_label("Exact DP value - policy value", fontsize=7.5)
    fig.suptitle("Exact regret from belief-state dynamic programming", fontsize=9, y=1.01)
    _save(fig, path)


def controlled_grid_rollout_source(path: Path) -> None:
    """Render the rollout-minus-source companion heatmaps for the appendix."""
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.15), sharex=True, sharey=True, constrained_layout=True)
    rows = controlled_benchmark_grid()
    source_signal = (0.0, 0.5, 1.0)
    gains = []
    for signal in source_signal:
        source = np.zeros((3, 3), dtype=float)
        rollout = np.zeros((3, 3), dtype=float)
        for item in rows:
            if item["source_signal"] != signal:
                continue
            budget_index = int(item["budget"]) - 1
            coupling_index = int(round(item["coupling"] * 2))
            source[budget_index, coupling_index] = item["source_margin"]
            rollout[budget_index, coupling_index] = item["gated_source_rollout"]
        gains.append(rollout - source)
    vmax = max(float(gain.max()) for gain in gains)
    image = None
    for column_index, (signal, gain) in enumerate(zip(source_signal, gains, strict=True)):
        axis = axes[column_index]
        image = axis.imshow(gain, cmap="Greens", vmin=0, vmax=vmax, origin="lower", aspect="auto")
        for budget_index in range(3):
            for coupling_index in range(3):
                value = gain[budget_index, coupling_index]
                axis.text(coupling_index, budget_index, f"{value:.1f}", ha="center", va="center", fontsize=6.8, color="white" if value > vmax * 0.55 else INK)
        axis.set(xticks=[0, 1, 2], xticklabels=["0", "0.5", "1"], yticks=[0, 1, 2], yticklabels=["1", "2", "3"], xlabel="Delayed-label coupling", title=f"Signal {signal:.1f}")
        if column_index == 0:
            axis.set_ylabel("Budget $B$")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.015)
    colorbar.set_label("Rollout value - source value", fontsize=7.5)
    fig.suptitle("Exact rollout gain over source margin", fontsize=9, y=1.03)
    _save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    delayed_adjudication(args.output_dir / "delayed_adjudication.pdf")
    rollout_method(args.output_dir / "gated_hull_rollout.pdf")
    controlled_and_folds(args.output_dir / "controlled_and_matpes_effects.pdf")
    dft_waterfall(args.output_dir / "matpes_dft_waterfall.pdf")
    controlled_grid_efficiency(args.output_dir / "controlled_grid_efficiency.pdf")
    controlled_grid_rollout_source(args.output_dir / "controlled_grid_rollout_source.pdf")


if __name__ == "__main__":
    main()
