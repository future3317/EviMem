"""Render the E52 objective-robustness and membership-calibration figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .publication_figure_style import (
        PALETTE,
        apply_publication_style,
        finalize_figure,
        style_axis,
    )
except ImportError:  # pragma: no cover - direct script execution
    from publication_figure_style import (
        PALETTE,
        apply_publication_style,
        finalize_figure,
        style_axis,
    )


def render(*, objective_path: Path, calibration_path: Path, output_path: Path) -> None:
    objective = json.loads(objective_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    groups = list(calibration["groups"].values())
    if len(groups) != 1 or groups[0]["policy"] != "delta_hull_active_search":
        raise ValueError("expected one Delta-Hull calibration group")
    group = groups[0]

    apply_publication_style(8.2)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.55),
        gridspec_kw={"width_ratios": (1.16, 0.84), "wspace": 0.34},
    )

    budgets = np.arange(1, 7)
    pool_styles = {
        "070": ("70% query pool", PALETTE["neutral_dark"], "s"),
        "085": ("85% query pool", PALETTE["teal"], "^"),
        "100": ("100% query pool", PALETTE["blue_main"], "o"),
    }
    for pool in ("070", "085", "100"):
        label, color, marker = pool_styles[pool]
        rows = objective["curves"][pool]["budgets"]
        values = np.asarray([rows[str(budget)]["paired_delta_T"] for budget in budgets])
        lower = np.asarray(
            [rows[str(budget)]["bootstrap_95"]["lower"] for budget in budgets]
        )
        upper = np.asarray(
            [rows[str(budget)]["bootstrap_95"]["upper"] for budget in budgets]
        )
        axes[0].plot(
            budgets,
            values,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=4.2,
            label=label,
            zorder=3,
        )
        axes[0].fill_between(budgets, lower, upper, color=color, alpha=0.12, linewidth=0)
    axes[0].axhline(0, color=PALETTE["charcoal"], linewidth=0.65, zorder=1)
    axes[0].set_xticks(budgets)
    axes[0].set_xlabel("Query budget $B$")
    axes[0].set_ylabel(r"Delta-Hull $-$ target margin, $\Delta T$")
    axes[0].set_title("a  Objective gain under query-pool shift", loc="left", fontweight="bold")
    axes[0].legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
    axes[0].set_ylim(-0.065, 0.205)
    style_axis(axes[0], grid=True)

    reliability = group["all_candidates"]["reliability_bins"]
    plotted = [row for row in reliability if row["record_count"]]
    predicted = np.asarray([row["mean_predicted_probability"] for row in plotted])
    empirical = np.asarray([row["empirical_frequency"] for row in plotted])
    counts = np.asarray([row["record_count"] for row in plotted], dtype=float)
    marker_sizes = 18.0 + 62.0 * np.sqrt(counts / counts.max())
    axes[1].plot([0, 1], [0, 1], color=PALETTE["neutral_dark"], linewidth=0.8, linestyle="--")
    axes[1].scatter(
        predicted,
        empirical,
        s=marker_sizes,
        color=PALETTE["blue_secondary"],
        edgecolor="white",
        linewidth=0.55,
        alpha=0.95,
        zorder=3,
    )
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Predicted final-hull probability")
    axes[1].set_ylabel("Observed complete-pool frequency")
    axes[1].set_title("b  Membership reliability", loc="left", fontweight="bold")
    metrics = group["all_candidates"]["metrics"]
    axes[1].text(
        0.04,
        0.96,
        (
            f"Brier {metrics['brier_score']:.3f}\n"
            f"NLL {metrics['bernoulli_nll']:.3f}\n"
            f"ROC-AUC {metrics['roc_auc']:.3f}"
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=PALETTE["charcoal"],
    )
    style_axis(axes[1], grid=False)

    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.21, top=0.89)
    finalize_figure(figure, output_path, dpi=300, pad_inches=0.035)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        objective_path=args.objective,
        calibration_path=args.calibration,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
