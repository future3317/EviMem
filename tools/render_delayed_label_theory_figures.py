"""Render the revised delayed-label main and exact-DP theory figures.

Inputs are external derived summaries.  The renderer never recomputes policy
outcomes; it only maps frozen summary fields to the manuscript figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from publication_figure_style import (  # noqa: E402
    PALETTE,
    apply_publication_style,
    finalize_figure,
    style_axis,
)

GRAY = PALETTE["neutral_dark"]
BLUE = PALETTE["blue_main"]
BLUE_LIGHT = "#B9D2E8"
ORANGE = PALETTE["red_strong"]
MAGENTA = PALETTE["violet"]
GREEN = PALETTE["teal"]
BLACK = PALETTE["charcoal"]


def _ci(entry: dict[str, Any]) -> tuple[float, float]:
    interval = entry.get("paired_bootstrap_95ci", (float("nan"), float("nan")))
    if isinstance(interval, str):
        interval = interval.split()
    return float(interval[0]), float(interval[1])


def render_main_figure_two(summary_path: Path, output: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    budgets = np.arange(1, 7)
    direct = summary["direct_pairwise_contrasts"]
    target_name = "posterior_mean_target_margin"
    delta_name = "delta_hull_active_search"
    pair_key = f"{delta_name}_vs_{target_name}"

    delta_t = []
    t_low = []
    t_high = []
    delta_f = []
    delta_revocation = []
    for budget in budgets:
        entry = direct[str(int(budget))][pair_key]["metrics"]
        t = entry["T"]["direct_paired"]
        delta_t.append(float(t["paired_mean_difference"]))
        low, high = _ci(t)
        t_low.append(low)
        t_high.append(high)
        delta_f.append(float(entry["F"]["direct_paired"]["paired_mean_difference"]))
        delta_revocation.append(
            float(entry["F_minus_T"]["direct_paired"]["paired_mean_difference"])
        )

    apply_publication_style(8.0)
    fig = plt.figure(figsize=(7.2, 2.65), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.00, 1.28), wspace=0.52)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    means = np.asarray(delta_t)
    ax_a.plot(budgets, means, color=BLUE, marker="o", ms=3.5, lw=1.4)
    ax_a.fill_between(budgets, t_low, t_high, color=BLUE_LIGHT, alpha=0.75, linewidth=0)
    ax_a.axhline(0.0, color=BLACK, lw=0.65)
    ax_a.annotate("+0.087", (2, means[1]), xytext=(2.1, means[1] + 0.027), fontsize=6.2, color=BLUE)
    ax_a.annotate("-0.004", (6, means[-1]), xytext=(5.0, means[-1] - 0.035), fontsize=6.2, color=BLUE)
    ax_a.text(
        0.04,
        0.95,
        "AUC +0.300",
        transform=ax_a.transAxes,
        va="top",
        fontsize=6.2,
        color=BLACK,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.82},
    )
    ax_a.set(xlabel="Query budget $B$", ylabel="$\\Delta T$ / system")
    ax_a.set_xticks(budgets)
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=8)

    b2 = summary["curve_vs_source"]["2"]
    policies = [(target_name, "Target margin", GRAY), (delta_name, "Delta-Hull", BLUE)]
    y = np.asarray([0.0, 1.0])
    for row, (policy, label, color) in enumerate(policies):
        metrics = b2[policy]["metrics"]
        f_value = float(metrics["F"]["policy_mean"])
        t_value = float(metrics["T"]["policy_mean"])
        ax_b.hlines(row, t_value, f_value, color=MAGENTA, lw=2.5, alpha=0.72, zorder=1)
        ax_b.scatter(t_value, row, color=color, s=26, zorder=3)
        ax_b.scatter(f_value, row, facecolors="white", edgecolors=color, linewidths=1.1, s=26, zorder=3)
        ax_b.text(t_value - 0.04, row - 0.16, f"{t_value:.3f}", ha="right", va="top", fontsize=5.8, color=color)
        ax_b.text(f_value + 0.04, row - 0.16, f"{f_value:.3f}", ha="left", va="top", fontsize=5.8, color=color)
    ax_b.set_yticks(y, ["Target margin", "Delta-Hull"])
    ax_b.set(xlabel="Confirmations / system")
    ax_b.set_xlim(0.0, 2.15)
    ax_b.set_ylim(-0.42, 1.42)
    ax_b.set_title("b", loc="left", fontweight="bold", fontsize=8)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=4.2, label="$T$ filled"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=BLACK, markersize=4.2, label="$F$ open"),
            Line2D([0], [0], color=MAGENTA, lw=2.5, label="$F-T$"),
        ],
        frameon=False,
        fontsize=5.2,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.58, 1.04),
        columnspacing=0.45,
        handlelength=1.3,
    )

    kappas = np.linspace(0.0, 1.2, 121)
    utility = np.asarray(delta_f)[:, None] - np.asarray(delta_revocation)[:, None] * kappas[None, :]
    x_edges = np.arange(0.5, 6.6, 1.0)
    y_edges = np.linspace(-0.005, 1.205, len(kappas) + 1)
    cmap = LinearSegmentedColormap.from_list("revocation", [ORANGE, "white", BLUE])
    vmax = max(abs(float(utility.min())), abs(float(utility.max())))
    mesh = ax_c.pcolormesh(
        x_edges,
        y_edges,
        utility.T,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
    )
    for edge in x_edges[1:-1]:
        ax_c.axvline(edge, color="white", lw=0.8, zorder=2)
    kstars = np.asarray(delta_f) / np.asarray(delta_revocation)
    finite_kstars = np.where((kstars >= 0.0) & (kstars <= 1.2), kstars, np.nan)
    ax_c.plot(budgets, finite_kstars, color=BLACK, lw=0.8, marker="x", ms=3.6, mew=0.8)
    ax_c.annotate("$\\kappa^\\star=0.23$", (2, kstars[1]), xytext=(2.3, 0.36), fontsize=5.8, color=BLACK)
    if kstars[-1] > 1.2:
        ax_c.text(5.98, 1.14, "$B=6$: $>1$", ha="right", fontsize=6, color=BLACK)
    ax_c.set(xlabel="Query budget $B$", ylabel=r"Revocation weight $\kappa$")
    ax_c.set_xticks(budgets)
    ax_c.set_ylim(0.0, 1.2)
    ax_c.set_title("c", loc="left", fontweight="bold", fontsize=8)
    cbar = fig.colorbar(mesh, ax=ax_c, fraction=0.045, pad=0.03)
    cbar.set_label(r"$\Delta U_\kappa$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    for axis in (ax_a, ax_b, ax_c):
        style_axis(axis)
    finalize_figure(fig, output)


def render_main_figure_two_clean(summary_path: Path, output: Path) -> None:
    """Render the objective-first main figure with absolute and direct views."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    budgets = np.arange(1, 7)
    curve = summary["curve_vs_source"]
    direct = summary["direct_pairwise_contrasts"]
    target = "posterior_mean_target_margin"
    delta = "delta_hull_active_search"

    apply_publication_style(8.0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), facecolor="white")
    ax_a, ax_b, ax_c = axes
    colors = {"source_margin": GRAY, target: MAGENTA, delta: BLUE}
    labels = {"source_margin": "Source", target: "Target margin", delta: "Delta-Hull"}
    for policy in ("source_margin", target, delta):
        values = np.asarray([curve[str(int(b))][policy]["metrics"]["T"]["policy_mean"] for b in budgets])
        ax_a.plot(budgets, values, marker="o", ms=3.0, lw=1.25, color=colors[policy], label=labels[policy])
    ax_a.set(xlabel="Query budget $B$", ylabel="$T$ / system", title="a")
    ax_a.set_xticks(budgets)
    ax_a.legend(fontsize=5.4, loc="upper left", handlelength=1.4)

    pair_key = f"{delta}_vs_{target}"
    means, lows, highs = [], [], []
    for b in budgets:
        entry = direct[str(int(b))][pair_key]["metrics"]["T"]["direct_paired"]
        means.append(float(entry["paired_mean_difference"]))
        low, high = _ci(entry)
        lows.append(low)
        highs.append(high)
    ax_b.plot(budgets, means, marker="o", ms=3.0, lw=1.25, color=BLUE)
    ax_b.fill_between(budgets, lows, highs, color=BLUE_LIGHT, alpha=0.75, linewidth=0)
    ax_b.axhline(0.0, color=BLACK, lw=0.65)
    ax_b.set(xlabel="Query budget $B$", ylabel="$\\Delta T$ / system", title="b")
    ax_b.text(
        0.04,
        0.72,
        "Delta-Hull $-$ target margin",
        transform=ax_b.transAxes,
        fontsize=5.9,
        color=BLUE,
    )
    ax_b.set_xticks(budgets)
    ax_b.text(
        0.04,
        0.95,
        "AUC $\\Delta=+0.300$\n95\\% CI $[+0.052,+0.559]$\n$p=0.0247$",
        transform=ax_b.transAxes,
        va="top",
        fontsize=5.7,
        color=BLACK,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.84},
    )

    for policy, color, label in (("source_margin", GRAY, "Source"), (target, MAGENTA, "Target margin"), (delta, BLUE, "Delta-Hull")):
        values = np.asarray([
            curve[str(int(b))][policy]["metrics"]["F"]["policy_mean"]
            - curve[str(int(b))][policy]["metrics"]["T"]["policy_mean"]
            for b in budgets
        ])
        ax_c.plot(budgets, values, marker="o", ms=3.0, lw=1.25, color=color, label=label)
    ax_c.set(xlabel="Query budget $B$", ylabel="$F-T$ / system", title="c")
    ax_c.set_xticks(budgets)
    ax_c.legend(fontsize=5.1, loc="upper right", handlelength=1.4)
    ax_c.annotate("39.4% lower at $B=2$", (2, 0.174), xytext=(2.25, 0.28), fontsize=5.8, color=BLUE, arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.6})

    for axis in axes:
        style_axis(axis, grid=True)
        axis.title.set_fontweight("bold")
        axis.title.set_ha("left")
        axis.title.set_position((0.0, 1.0))
    fig.subplots_adjust(wspace=0.43, bottom=0.23, top=0.86)
    finalize_figure(fig, output)


def _flatten_candidate_terms(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    penalty: list[float] = []
    gain: list[float] = []
    term: list[float] = []
    for row in rows:
        for candidate in row.get("two_step_candidate_terms", []):
            if int(candidate["rank"]) < 3:
                continue
            penalty.append(float(candidate["rank_penalty"]))
            gain.append(float(candidate["information_gain"]))
            term.append(float(candidate["headroom_term"]))
    return np.asarray(penalty), np.asarray(gain), np.asarray(term)


def render_main_figure_three(audit_path: Path, output: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = [row for row in audit["rows"] if "two_step_headroom" in row]
    penalty, gain, term = _flatten_candidate_terms(rows)
    headroom = np.asarray([float(row["two_step_headroom"]) for row in rows])
    top_two_difference = np.asarray(
        [float(row.get("two_step_top_two_information_difference", 0.0)) for row in rows]
    )
    boundary = np.asarray([float(row["boundary_mass_delta_0_02"]) for row in rows])
    bound = np.asarray([float(row["certificate_bound_2n_epsilon"]) for row in audit["rows"]])
    gap = np.asarray([float(row["exact_dp_minus_greedy"]) for row in audit["rows"]])

    apply_publication_style(8.0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), facecolor="white")
    ax_a, ax_b, ax_c = axes
    positive = term > 1e-12
    ax_a.scatter(penalty[~positive], gain[~positive], s=5, color=GRAY, alpha=0.32, linewidths=0)
    ax_a.scatter(penalty[positive], gain[positive], s=10, color=ORANGE, alpha=0.9, linewidths=0)
    top_two_positive = top_two_difference > 1e-12
    if np.any(top_two_positive):
        ax_a.scatter(
            np.zeros(np.count_nonzero(top_two_positive)),
            top_two_difference[top_two_positive],
            s=18,
            color=ORANGE,
            edgecolors=BLACK,
            linewidths=0.45,
            marker="D",
            zorder=4,
        )
    lo = 0.0
    hi = max(float(penalty.max()), float(gain.max()), 1e-6)
    ax_a.plot([lo, hi], [lo, hi], color=BLACK, lw=0.7)
    ax_a.axhline(0, color=BLACK, lw=0.5)
    ax_a.axvline(0, color=BLACK, lw=0.5)
    ax_a.set(xlabel=r"Rank penalty $p_2-p_j$", ylabel=r"Information gain $I_j-I_1$")
    ax_a.set_xlim(left=-0.04)
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=8)
    ax_a.text(0.04, 0.94, f"overall positive $H_2$: {int(np.count_nonzero(headroom > 1e-12))}/{len(rows)} roots", transform=ax_a.transAxes, fontsize=6.0, va="top")

    ax_b.scatter(boundary, headroom, s=7, color=BLUE, alpha=0.32, linewidths=0)
    ax_b.scatter(boundary[headroom > 1e-12], headroom[headroom > 1e-12], s=13, color=ORANGE, alpha=0.92, linewidths=0)
    ax_b.set(xlabel=r"Boundary mass, $|m|\leq0.04$", ylabel=r"Exact two-step headroom $H_2$")
    ax_b.set_title("b", loc="left", fontweight="bold", fontsize=8)

    positive_bound = bound > 1e-12
    positive_gap = gap > 1e-12
    nonzero = positive_bound & positive_gap
    ax_c.scatter(bound[nonzero], gap[nonzero], s=7, color=GREEN, alpha=0.44, linewidths=0)
    zero_gap = positive_bound & ~positive_gap
    if np.any(zero_gap):
        ax_c.scatter(
            bound[zero_gap],
            np.full(np.count_nonzero(zero_gap), 1e-5),
            s=8,
            color=ORANGE,
            alpha=0.55,
            linewidths=0,
            marker="_",
        )
    upper = max(float(bound.max()), float(gap.max()), 1e-3) * 1.05
    ax_c.plot([1e-5, upper], [1e-5, upper], color=BLACK, lw=0.7)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set(xlabel=r"Certificate $2n\epsilon$", ylabel=r"Exact $V^\star-V^\Delta$")
    ax_c.set_title("c", loc="left", fontweight="bold", fontsize=8)
    zero_fraction = float(np.mean(gap <= 1e-12))
    ax_c.text(0.04, 0.94, f"coverage 100%; zero gap {zero_fraction:.1%}", transform=ax_c.transAxes, fontsize=6.0, va="top")

    for axis in axes:
        style_axis(axis)
    fig.subplots_adjust(wspace=0.48, bottom=0.22, top=0.86)
    finalize_figure(fig, output)


def render_appendix_certificate(audit_path: Path, output: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = [row for row in audit["rows"] if "two_step_headroom" in row]
    headroom = np.asarray([float(row["two_step_headroom"]) for row in rows])
    bound = np.asarray([float(row["certificate_bound_2n_epsilon"]) for row in audit["rows"]])
    gap = np.asarray([float(row["exact_dp_minus_greedy"]) for row in audit["rows"]])

    apply_publication_style(8.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), facecolor="white")
    axes[0].hist(headroom, bins=24, color=GRAY, edgecolor="white", linewidth=0.35)
    axes[0].hist(headroom[headroom > 1e-12], bins=8, color=ORANGE, edgecolor="white", linewidth=0.35)
    axes[0].set_yscale("log")
    axes[0].set(xlabel=r"Two-step headroom $H_2$", ylabel="Instances (log scale)")
    axes[0].text(0.96, 0.94, "11 positive", transform=axes[0].transAxes, ha="right", va="top", fontsize=6.5, color=ORANGE)
    axes[0].set_title("a", loc="left", fontweight="bold", fontsize=8)
    positive = bound > 1e-12
    nonzero = positive & (gap > 1e-12)
    axes[1].scatter(bound[nonzero], gap[nonzero], s=6, color=GREEN, alpha=0.3, linewidths=0)
    zero_gap = positive & (gap <= 1e-12)
    if np.any(zero_gap):
        axes[1].scatter(
            bound[zero_gap],
            np.full(np.count_nonzero(zero_gap), 1e-5),
            s=6,
            color=ORANGE,
            alpha=0.45,
            linewidths=0,
            marker="_",
        )
    upper = max(float(bound.max()), float(gap.max()), 1e-3) * 1.05
    axes[1].plot([1e-5, upper], [1e-5, upper], color=BLACK, lw=0.7)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(xlabel=r"Certificate $2n\epsilon$", ylabel=r"Exact DP--greedy gap (nonzero)")
    axes[1].text(
        0.04,
        0.94,
        f"zero gap {float(np.mean(gap <= 1e-12)):.1%}",
        transform=axes[1].transAxes,
        fontsize=6.0,
        va="top",
    )
    axes[1].set_title("b", loc="left", fontweight="bold", fontsize=8)
    for axis in axes:
        style_axis(axis)
    fig.subplots_adjust(wspace=0.45, bottom=0.24, top=0.88)
    finalize_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e32-summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_main_figure_two_clean(args.e32_summary, args.output_dir / "matpes_budget_curve.pdf")
    render_main_figure_three(args.audit, args.output_dir / "matpes_dft_waterfall.pdf")
    render_appendix_certificate(args.audit, args.output_dir / "exact_certificate_audit.pdf")


if __name__ == "__main__":
    main()
