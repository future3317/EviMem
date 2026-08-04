"""Render the evaluator-only E32 lookahead mechanism audit figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAVY = "#123A63"
BLUE = "#2D6FA3"
GOLD = "#D8B26A"
SLATE = "#6F7D8C"
RED = "#B45F5F"


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

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"wspace": 0.42})

    axes[0].bar(np.arange(3), state_means, color=[SLATE, BLUE, GOLD], width=0.65)
    axes[0].set_xticks(np.arange(3), ["Low", "Middle", "High"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Action disagreement")
    axes[0].set_title("Q-gap predicts action change", color=NAVY, fontsize=9)
    for i, (value, count) in enumerate(zip(state_means, state_counts)):
        axes[0].text(i, value + 0.04, f"{100 * value:.1f}%\n(n={count})", ha="center", fontsize=7)

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
        fmt="o", color=NAVY, ecolor=GOLD, capsize=3, markersize=4,
    )
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_xticks(np.arange(3), [f"{label}\n(n={count})" for label, count in zip(t_labels, counts)])
    axes[1].set_ylabel(r"Realized $\Delta T$")
    axes[1].set_title("Q-gap: weak realized calibration", color=NAVY, fontsize=9)

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
        fmt="o", color=RED, ecolor=GOLD, capsize=3, markersize=4,
    )
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_xticks(np.arange(2), [f"{label}\n(n={count})" for label, count in zip(rank_labels, counts)])
    axes[2].set_ylabel(r"Realized $\Delta T$")
    axes[2].set_title("Rank-switch: weak realized calibration", color=NAVY, fontsize=9)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9DEE4", linewidth=0.5, alpha=0.7)
        axis.tick_params(labelsize=7)
        axis.yaxis.label.set_size(8)
    fig.subplots_adjust(top=0.78, bottom=0.28)
    fig.text(
        0.5, 0.01,
        "E32-A B=6; 220 systems with complete state diagnostics; bars/points use frozen traces",
        ha="center", fontsize=7, color=SLATE,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.input, args.output)


if __name__ == "__main__":
    main()
