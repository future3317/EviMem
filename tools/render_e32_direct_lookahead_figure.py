"""Render the evaluator-only E32 lookahead mechanism audit figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from publication_figure_style import PALETTE, apply_publication_style, finalize_figure, style_axis


def _bootstrap_mean(values: list[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def render(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    states = data["state_rows"]
    systems = list(data["system_rows"].values())

    qgap = np.asarray([row["q_gap_over_delta_action"] for row in states], dtype=float)
    disagreement = np.asarray([row["action_disagreement"] for row in states], dtype=float)
    q25, q75 = np.quantile(qgap, [0.25, 0.75])
    state_masks = [qgap <= q25, (qgap > q25) & (qgap < q75), qgap >= q75]
    state_means = [float(disagreement[mask].mean()) for mask in state_masks]
    state_counts = [int(mask.sum()) for mask in state_masks]

    system_qgap = np.asarray(
        [row["max_q_gap_over_delta_action"] for row in systems], dtype=float
    )
    system_t = np.asarray([row["anchored_minus_delta_T"] for row in systems], dtype=float)
    tq25, tq75 = np.quantile(system_qgap, [0.25, 0.75])
    t_masks = [system_qgap <= tq25, (system_qgap > tq25) & (system_qgap < tq75), system_qgap >= tq75]
    t_labels = ["Low", "Middle", "High"]

    system_rank = np.asarray(
        [row["max_rank_switch_selected_action"] for row in systems], dtype=float
    )
    rank_masks = [system_rank <= 0.5, system_rank > 0.5]
    rank_labels = [r"$\max r_h(x)\leq0.5$", r"$\max r_h(x)>0.5$"]

    apply_publication_style(8.0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"wspace": 0.42})

    axes[0].bar(
        np.arange(3),
        state_means,
        color=[PALETTE["neutral_dark"], PALETTE["blue_secondary"], PALETTE["red_strong"]],
        edgecolor="white",
        linewidth=0.5,
        width=0.62,
    )
    axes[0].set_xticks(np.arange(3), ["Low", "Middle", "High"])
    axes[0].set_ylim(0, 1.24)
    axes[0].set_ylabel("Action disagreement")
    axes[0].set_title("Action change by Q-gap", loc="left", fontweight="bold")
    for i, (value, count) in enumerate(zip(state_means, state_counts)):
        axes[0].text(i, value + 0.04, f"{100 * value:.1f}%\n$n={count}$", ha="center", fontsize=5.8)

    means = []
    errors = []
    counts = []
    for mask in t_masks:
        values = system_t[mask]
        means.append(float(values.mean()))
        lo, hi = _bootstrap_mean(values.tolist(), 20260808 + len(means))
        errors.append((means[-1] - lo, hi - means[-1]))
        counts.append(int(mask.sum()))
    axes[1].errorbar(
        np.arange(3), means,
        yerr=np.asarray(errors).T,
        fmt="o", color=PALETTE["teal"], ecolor=PALETTE["gold"], capsize=2.5, markersize=4,
    )
    axes[1].axhline(0, color=PALETTE["charcoal"], linewidth=0.6)
    axes[1].set_xticks(np.arange(3), [f"{label}\n(n={count})" for label, count in zip(t_labels, counts)])
    axes[1].set_ylabel(r"Realized $\Delta T$")
    axes[1].set_title("Realized $\\Delta T$ by Q-gap", loc="left", fontweight="bold")

    means = []
    errors = []
    counts = []
    for mask in rank_masks:
        values = system_t[mask]
        means.append(float(values.mean()))
        lo, hi = _bootstrap_mean(values.tolist(), 20260811 + len(means))
        errors.append((means[-1] - lo, hi - means[-1]))
        counts.append(int(mask.sum()))
    axes[2].errorbar(
        np.arange(2), means,
        yerr=np.asarray(errors).T,
        fmt="o", color=PALETTE["red_strong"], ecolor=PALETTE["gold"], capsize=2.5, markersize=4,
    )
    axes[2].axhline(0, color=PALETTE["charcoal"], linewidth=0.6)
    axes[2].set_xticks(np.arange(2), [f"{label}\n(n={count})" for label, count in zip(rank_labels, counts)])
    axes[2].set_title("Realized $\\Delta T$ by rank switch", loc="left", fontweight="bold")

    all_means = np.asarray(means, dtype=float)
    # The final panel is recomputed below; use both panels' confidence ranges
    # to keep the realized-utility axes directly comparable.
    qgap_means = []
    qgap_errors = []
    for mask in t_masks:
        values = system_t[mask]
        qgap_means.append(float(values.mean()))
        lo, hi = _bootstrap_mean(values.tolist(), 20260808 + len(qgap_means))
        qgap_errors.append((qgap_means[-1] - lo, hi - qgap_means[-1]))
    utility_extent = max(
        0.03,
        float(np.max(np.abs(np.asarray(qgap_means)[:, None] + np.asarray(qgap_errors)))) if qgap_errors else 0.03,
        float(np.max(np.abs(all_means[:, None] + np.asarray(errors)))) if errors else 0.03,
    )
    axes[1].set_ylim(-1.15 * utility_extent, 1.15 * utility_extent)
    axes[2].set_ylim(-1.15 * utility_extent, 1.15 * utility_extent)

    for axis in axes:
        style_axis(axis, grid=True)
    fig.subplots_adjust(top=0.78, bottom=0.28)
    fig.text(
        0.5, 0.01,
        "$B=6$; 220 systems with complete state diagnostics; error bars are paired system bootstrap intervals",
        ha="center", fontsize=5.8, color=PALETTE["neutral_dark"],
    )
    finalize_figure(fig, output_path)


def render_main(summary_path: Path, audit_path: Path, output_path: Path) -> None:
    """Render the main-text solver comparison from the frozen E32 summary."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    budgets = np.asarray(summary["budgets"], dtype=int)
    curve = summary["curve_vs_source"]
    direct = summary["direct_pairwise_contrasts"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    delta = "delta_hull_active_search"
    anchored = "delta_hull_anchored_rollout"
    pair = f"{anchored}_vs_{delta}"

    def interval(entry: dict) -> tuple[float, float]:
        value = entry["paired_bootstrap_95ci"]
        if isinstance(value, str):
            value = value.split()
        return float(value[0]), float(value[1])

    apply_publication_style(8.6)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.72), facecolor="white")
    ax_a, ax_b, ax_c = axes

    for policy, color, label, marker in (
        (delta, PALETTE["blue_main"], "Delta-Hull", "o"),
        (anchored, PALETTE["red_strong"], "Anchored rollout", "D"),
    ):
        values = np.asarray([curve[str(int(b))][policy]["metrics"]["T"]["policy_mean"] for b in budgets])
        ax_a.plot(budgets, values, color=color, marker=marker, ms=3.0, lw=1.3, label=label)
    ax_a.set(xlabel="Query budget $B$", ylabel="$T$ / system", title="a  Terminal utility")
    ax_a.set_xticks(budgets)
    ax_a.legend(fontsize=6.0, loc="upper left", handlelength=1.4)

    means, lows, highs = [], [], []
    for budget in budgets:
        entry = direct[str(int(budget))][pair]["metrics"]["T"]["direct_paired"]
        means.append(float(entry["paired_mean_difference"]))
        low, high = interval(entry)
        lows.append(low)
        highs.append(high)
    means = np.asarray(means)
    ax_b.plot(budgets, means, color=PALETTE["red_strong"], marker="D", ms=3.0, lw=1.3)
    ax_b.fill_between(budgets, lows, highs, color=PALETTE["red_strong"], alpha=0.13, linewidth=0)
    ax_b.axhline(0.0, color=PALETTE["charcoal"], lw=0.65)
    ax_b.annotate("$B=6$: +0.039; $p=0.064$", (6, means[-1]), xytext=(2.6, 0.072), fontsize=6.2, color=PALETTE["red_strong"], arrowprops={"arrowstyle": "-", "color": PALETTE["red_strong"], "lw": 0.6})
    ax_b.set(xlabel="Query budget $B$", ylabel="Anchored $-$ Delta $\\Delta T$", title="b  Paired gain")
    ax_b.set_xticks(budgets)

    b6 = direct["6"][pair]["metrics"]["T"]["direct_paired"]
    system_count = int(b6["system_count"])
    changed_systems = sum(
        float(row["max_q_gap_over_delta_action"]) > 0.0
        for row in audit["system_rows"].values()
    )
    rows = (
        ("Action", (("changed", changed_systems, PALETTE["blue_main"]),
                    ("unchanged", system_count - changed_systems, PALETTE["neutral"]))),
        ("Terminal $T$", (("win", int(b6["wins"]), PALETTE["green_3"]),
                          ("tie", int(b6["ties"]), PALETTE["neutral_dark"]),
                          ("loss", int(b6["losses"]), PALETTE["red_strong"]))),
    )
    for row_index, (_, segments) in enumerate(rows):
        left = 0
        for label, count, color in segments:
            ax_c.barh(
                row_index,
                count,
                left=left,
                height=0.44,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=label,
            )
            if count >= 28:
                ax_c.text(
                    left + count / 2,
                    row_index,
                    f"{count} {label}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    fontweight="bold",
                    color="white",
                )
            else:
                annotation_color = (
                    PALETTE["neutral_dark"] if label == "unchanged" else color
                )
                ax_c.annotate(
                    f"{count} {label}",
                    (left + count / 2, row_index),
                    xytext=(0, 18 if row_index == 0 else 12),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=annotation_color,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": annotation_color,
                        "lw": 0.55,
                    },
                )
            left += count
    ax_c.set(
        xlim=(0, system_count),
        yticks=np.arange(len(rows)),
        yticklabels=[row[0] for row in rows],
        xlabel="Systems at $B=6$",
        title="c  Action change vs terminal gain",
    )
    ax_c.invert_yaxis()
    ax_c.set_xticks([0, 115, 230])
    ax_c.set_xticklabels(["0", "50%", "230"])
    for axis in axes:
        style_axis(axis, grid=True)
        axis.title.set_fontweight("bold")
        axis.title.set_ha("left")
        axis.title.set_position((0.0, 1.0))
    fig.subplots_adjust(wspace=0.47, bottom=0.25, top=0.86)
    finalize_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.summary is None:
        render(args.input, args.output)
    else:
        render_main(args.summary, args.input, args.output)


if __name__ == "__main__":
    main()
