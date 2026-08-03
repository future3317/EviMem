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

import matplotlib.pyplot as plt
import numpy as np

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
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
    for policy in primary:
        means = np.asarray(
            [curve[str(b)][policy]["metrics"]["T"]["direct_paired"]["paired_mean_difference"] for b in budgets]
        )
        intervals = np.asarray(
            [_ci(curve[str(b)][policy]["metrics"]["T"]["direct_paired"]) for b in budgets]
        )
        axes[0].plot(budgets, means, marker="o", label=POLICY_LABELS[policy])
        axes[0].fill_between(budgets, intervals[:, 0], intervals[:, 1], alpha=0.10)

    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set(xlabel="Query budget $B$", ylabel="$\\Delta T$ vs source/system")
    axes[0].set_title("Delayed full-pool objective")
    axes[0].legend(fontsize=7, frameon=False)

    strata = summary["b6_posterior_only_rank_coupling_strata"]
    state = strata.get("state_summary", {})
    coupling = state.get("action_disagreement_by_coupling_quartile", {})
    rank = state.get("action_disagreement_by_rank_margin_quartile", {})
    labels = ["low", "middle", "high"]
    x = np.arange(3)
    width = 0.36
    axes[1].bar(
        x - width / 2,
        [100.0 * coupling.get(label, {}).get("mean_action_disagreement", np.nan) for label in labels],
        width,
        label="Coupling quartile",
    )
    axes[1].bar(
        x + width / 2,
        [100.0 * rank.get(label, {}).get("mean_action_disagreement", np.nan) for label in labels],
        width,
        label="Rank-margin quartile",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set(xlabel="Posterior-only stratum", ylabel="Action disagreement (%)")
    axes[1].set_title("Where lookahead changes the action")
    axes[1].legend(fontsize=7, frameon=False)
    _save(fig, output_dir / "e32_delayed_label_mechanism.pdf")


def render_mad(summary_path: Path, output_dir: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    budgets = np.asarray(summary["budgets"], dtype=int)
    curves = summary["curves"]
    policies = tuple(summary["policies"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
    for policy in policies:
        axes[0].plot(
            budgets,
            curves[policy]["oracle_pool_confirmed_discoveries"],
            marker="o",
            label=POLICY_LABELS[policy],
        )
    axes[0].set(xlabel="Query budget $B$", ylabel="$T$ / system")
    axes[0].set_title("MAD-1.5 complete-pool proxy")
    axes[0].legend(fontsize=7, frameon=False)

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
    for key, label in contrasts:
        means = []
        lows = []
        highs = []
        for budget in contrast_budgets:
            entry = summary["direct_pairwise_by_budget"][str(budget)][key]["T"]
            means.append(entry["mean"])
            low, high = _ci(entry)
            lows.append(low)
            highs.append(high)
        axes[1].plot(contrast_budgets, means, marker="o", label=label)
        axes[1].fill_between(contrast_budgets, lows, highs, alpha=0.10)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set(xlabel="Query budget $B$", ylabel="Paired $\\Delta T$ / system")
    axes[1].set_title("Direct paired mechanism contrasts")
    axes[1].legend(fontsize=7, frameon=False)
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
