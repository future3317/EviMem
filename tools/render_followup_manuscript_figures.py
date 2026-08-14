"""Render manuscript figures for the completed delayed-label follow-up suites.

The inputs are derived summaries outside Git; this script only writes to the
explicit output directory supplied by the caller.  It does not recompute any
policy result or alter the registered estimands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from publication_figure_style import PALETTE, apply_publication_style, finalize_figure, style_axis

POLICY_LABELS = {
    "source_margin": "Source margin",
    "posterior_mean_target_margin": "Target margin",
    "posterior_current_hull_probability": "Current-hull probability",
    "delta_hull_active_search": "Delta-Hull",
    "ungated_source_rollout": "Ungated SARR",
    "source_rollout_delta_hull": "Source-rollout / Delta-Hull",
    "delta_hull_anchored_rollout": "Delta-Hull-anchored rollout",
    "independent_confirmation_source_rollout": "IC-SARR",
}


def _save(fig: plt.Figure, output: Path) -> None:
    finalize_figure(fig, output)


def _ci(entry: dict[str, Any]) -> tuple[float, float]:
    interval = entry.get("paired_bootstrap_95ci")
    if interval is None:
        return (float("nan"), float("nan"))
    return float(interval[0]), float(interval[1])


def render_e32(summary_path: Path, output_dir: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    budgets = np.asarray(summary["budgets"], dtype=int)
    curve = summary["curve_vs_source"]
    primary = (
        "delta_hull_active_search",
        "ungated_source_rollout",
        "source_rollout_delta_hull",
        "delta_hull_anchored_rollout",
    )
    apply_publication_style(8.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"wspace": 0.42})
    line_colors = {
        "delta_hull_active_search": PALETTE["blue_main"],
        "ungated_source_rollout": PALETTE["teal"],
        "source_rollout_delta_hull": PALETTE["violet"],
        "delta_hull_anchored_rollout": PALETTE["red_strong"],
    }
    line_styles = {
        "delta_hull_active_search": "-",
        "ungated_source_rollout": "-",
        "source_rollout_delta_hull": "--",
        "delta_hull_anchored_rollout": "-",
    }
    for policy in primary:
        means = np.asarray(
            [curve[str(b)][policy]["metrics"]["T"]["direct_paired"]["paired_mean_difference"] for b in budgets]
        )
        intervals = np.asarray(
            [_ci(curve[str(b)][policy]["metrics"]["T"]["direct_paired"]) for b in budgets]
        )
        axes[0].plot(
            budgets,
            means,
            marker="o",
            ms=3.2,
            lw=1.35,
            ls=line_styles[policy],
            color=line_colors[policy],
            label=POLICY_LABELS[policy],
        )
        axes[0].fill_between(
            budgets,
            intervals[:, 0],
            intervals[:, 1],
            color=line_colors[policy],
            alpha=0.08,
            linewidth=0,
        )

    axes[0].axhline(0.0, color=PALETTE["charcoal"], lw=0.65)
    axes[0].set(xlabel="Query budget $B$", ylabel="$\\Delta T$ vs source/system")
    axes[0].set_title("Terminal-$T$ contrast", loc="left", fontweight="bold")
    axes[0].set_xticks(budgets)
    axes[0].legend(fontsize=5.8, loc="upper right", handlelength=1.8)

    strata = summary["b6_posterior_only_rank_coupling_strata"]
    state = strata.get("state_summary", {})
    coupling = state.get("action_disagreement_by_coupling_quartile", {})
    rank = state.get("action_disagreement_by_rank_margin_quartile", {})
    labels = ["low", "middle", "high"]
    x = np.arange(3)
    width = 0.30
    coupling_values = [100.0 * coupling.get(label, {}).get("mean_action_disagreement", np.nan) for label in labels]
    rank_values = [100.0 * rank.get(label, {}).get("mean_action_disagreement", np.nan) for label in labels]
    axes[1].bar(
        x - width / 2,
        coupling_values,
        width,
        label="Coupling",
        color=PALETTE["blue_secondary"],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].bar(
        x + width / 2,
        rank_values,
        width,
        label="Rank margin",
        color=PALETTE["red_strong"],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_xticks(x, [label.capitalize() for label in labels])
    axes[1].set(xlabel="Posterior-only stratum", ylabel="Action disagreement (%)")
    axes[1].set_ylim(0, max(coupling_values + rank_values) * 1.18)
    axes[1].set_title("Action changes are not monotone", loc="left", fontweight="bold")
    axes[1].legend(fontsize=5.8, loc="upper right")
    for index, value in enumerate(coupling_values):
        axes[1].text(index - width / 2, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=5.4)
    for index, value in enumerate(rank_values):
        axes[1].text(index + width / 2, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=5.4)
    for axis in axes:
        style_axis(axis, grid=True)
    _save(fig, output_dir / "e32_delayed_label_mechanism.pdf")


def render_mad(summary_path: Path, output_dir: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    budgets = np.asarray(summary["budgets"], dtype=int)
    curves = summary["curves"]
    policies = tuple(summary["policies"])
    apply_publication_style(8.0)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.65),
        gridspec_kw={"width_ratios": (1.08, 1.02, 0.56), "wspace": 0.46},
    )
    policy_colors = {
        "source_margin": PALETTE["neutral_dark"],
        "delta_hull_active_search": PALETTE["blue_main"],
        "ungated_source_rollout": PALETTE["teal"],
        "independent_confirmation_source_rollout": PALETTE["red_strong"],
    }
    policy_styles = {
        "source_margin": ("--", "o"),
        "delta_hull_active_search": ("-", "o"),
        "ungated_source_rollout": ("-", "s"),
        "independent_confirmation_source_rollout": ("-", "D"),
    }
    for policy in policies:
        linestyle, marker = policy_styles.get(policy, ("-", "o"))
        axes[0].plot(
            budgets,
            curves[policy]["oracle_pool_confirmed_discoveries"],
            marker=marker,
            ms=3.2,
            lw=1.3,
            ls=linestyle,
            color=policy_colors.get(policy, PALETTE["charcoal"]),
            label=POLICY_LABELS[policy],
        )
    axes[0].set(xlabel="Query budget $B$", ylabel="$T$ / system")
    axes[0].set_title("Complete-pool proxy", loc="left", fontweight="bold")
    axes[0].set_xticks(budgets)
    axes[0].legend(fontsize=5.8, loc="upper left", handlelength=1.8)

    contrasts = (
        ("delta_hull_active_search_minus_source_margin", "Delta-Hull $-$ source"),
        ("ungated_source_rollout_minus_delta_hull_active_search", "Ungated SARR $-$ Delta-Hull"),
        (
            "independent_confirmation_source_rollout_minus_ungated_source_rollout",
            "IC-SARR $-$ ungated SARR",
        ),
        (
            "independent_confirmation_source_rollout_minus_delta_hull_active_search",
            "IC-SARR $-$ Delta-Hull",
        ),
    )
    # The MAD curve includes B=0 as the no-query reference, but direct
    # paired contrasts are registered only for B=1..6.  Keep B=0 on the
    # absolute curve and restrict the contrast panel to budgets with an
    # audited paired comparison.
    contrast_budgets = np.asarray(
        [budget for budget in budgets if str(int(budget)) in summary["direct_pairwise_by_budget"]],
        dtype=int,
    )
    contrast_colors = [
        PALETTE["blue_main"],
        PALETTE["teal"],
        PALETTE["red_strong"],
        PALETTE["violet"],
    ]
    # A horizontal forest plot keeps the paired uncertainty readable.  The
    # previous overlay of four ribbons made the zero crossing hard to audit,
    # especially at B=3--6 where the contrasts are close.
    y_base = np.arange(len(contrast_budgets), dtype=float) + 1.0
    y_offsets = np.linspace(-0.27, 0.27, len(contrasts))
    for (key, label), color, offset in zip(contrasts, contrast_colors, y_offsets, strict=True):
        means = []
        lows = []
        highs = []
        for budget in contrast_budgets:
            entry = summary["direct_pairwise_by_budget"][str(budget)][key]["T"]
            means.append(entry["mean"])
            low, high = _ci(entry)
            lows.append(low)
            highs.append(high)
        means = np.asarray(means, dtype=float)
        xerr = np.vstack((means - np.asarray(lows), np.asarray(highs) - means))
        axes[1].errorbar(
            means,
            y_base + offset,
            xerr=xerr,
            fmt="o",
            ms=3.2,
            lw=0.9,
            elinewidth=0.9,
            capsize=2.0,
            color=color,
            ecolor=color,
            label=label,
        )
    axes[1].axvline(0.0, color=PALETTE["charcoal"], lw=0.7, zorder=0)
    axes[1].set(xlabel="Paired $\\Delta T$ / system", ylabel="Query budget $B$")
    axes[1].set_title("Direct paired contrasts", loc="left", fontweight="bold")
    axes[1].set_yticks(y_base, [str(int(budget)) for budget in contrast_budgets])
    axes[1].set_xticks(
        [-0.05, 0.00, 0.05, 0.10, 0.15],
        ["−0.05", "0.00", "0.05", "0.10", "0.15"],
    )
    axes[1].set_ylim(0.45, len(contrast_budgets) + 0.55)
    axes[1].invert_yaxis()
    legend_handles, legend_labels = axes[1].get_legend_handles_labels()
    axes[2].axis("off")
    axes[2].legend(
        legend_handles,
        legend_labels,
        fontsize=5.0,
        loc="center left",
        handlelength=1.4,
        labelspacing=1.0,
        frameon=False,
    )
    for axis in axes[:2]:
        style_axis(axis, grid=True)
    _save(fig, output_dir / "mad15_direct_mechanism_curve.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e32-summary", type=Path)
    parser.add_argument("--mad-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.e32_summary is None and args.mad_summary is None:
        raise ValueError("at least one derived summary is required")
    if args.e32_summary is not None:
        render_e32(args.e32_summary, args.output_dir)
    if args.mad_summary is not None:
        render_mad(args.mad_summary, args.output_dir)


if __name__ == "__main__":
    main()
