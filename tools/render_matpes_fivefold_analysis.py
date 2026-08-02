"""Render audited MatPES B=6 development figures from archived five-fold outputs.

The input directory is intentionally outside Git.  These plots perform no new
policy evaluation: they only summarize the complete frozen system records from
the five IC-SARR/source-margin B=6 runs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.6,
    }
)


INK = "#172033"
BLUE = "#2D6F9F"
GREEN = "#1D7A5B"
GRAY = "#6B7C93"
RED = "#A53A3A"


def load_system_rows(archive_dir: Path) -> list[dict[str, Any]]:
    """Load the 230 paired exact-system records and retain their fold IDs."""
    paths = sorted(archive_dir.glob("matpes-ic-sarr-crossfit-fold*-b6-v1.json"))
    if len(paths) != 5:
        raise ValueError(f"Expected five archived fold summaries, found {len(paths)}.")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold = int(payload["config"]["crossfit_fold_index"])
        if payload["active_policies"] != [
            "source_margin",
            "independent_confirmation_source_rollout",
        ]:
            raise ValueError(f"Unexpected policy comparison in {path.name}.")
        for system, result in payload["systems"].items():
            source = result["strategies"]["source_margin"]
            rollout = result["strategies"]["independent_confirmation_source_rollout"]
            rows.append(
                {
                    "system": system,
                    "fold": fold,
                    "order": len(system.split("-")),
                    "source_t": float(source["oracle_pool_confirmed_discoveries"]),
                    "rollout_t": float(rollout["oracle_pool_confirmed_discoveries"]),
                    "source_f": float(source["final_causal_confirmed_discoveries"]),
                    "rollout_f": float(rollout["final_causal_confirmed_discoveries"]),
                    "source_d": float(source["causal_discoveries"]),
                    "rollout_d": float(rollout["causal_discoveries"]),
                    "ceiling": float(source["oracle_pool_discovery_ceiling"]),
                    "source_time": float(source["wall_seconds"]),
                    "rollout_time": float(rollout["wall_seconds"]),
                    "rollout_rounds": rollout["policy_decision_rounds"],
                }
            )
    if len(rows) != 230 or len({row["system"] for row in rows}) != 230:
        raise ValueError("Archived records are not a complete disjoint 230-system panel.")
    return rows


def bootstrap_interval(values: np.ndarray, seed: int, repetitions: int = 20_000) -> tuple[float, float]:
    """Return a deterministic paired-system percentile bootstrap interval."""
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[sample_indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_effects(rows: list[dict[str, Any]], output: Path) -> None:
    """Render fold forest and a discrete paired-count matrix for B=6."""
    figure, (forest, matrix_axis) = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    all_delta = np.array([row["rollout_t"] - row["source_t"] for row in rows])
    estimates: list[tuple[str, float, tuple[float, float], int]] = []
    for fold in range(1, 6):
        fold_delta = np.array([row["rollout_t"] - row["source_t"] for row in rows if row["fold"] == fold])
        estimates.append((f"Fold {fold}", float(fold_delta.mean()), bootstrap_interval(fold_delta, 20260722 + fold), len(fold_delta)))
    estimates.append(("All 230", float(all_delta.mean()), bootstrap_interval(all_delta, 20260722), len(all_delta)))
    y = np.arange(len(estimates))
    forest.axvline(0, color=GRAY, lw=0.8, zorder=0)
    for index, (_, estimate, interval, _) in enumerate(estimates):
        color, marker = (GREEN, "D") if index == len(estimates) - 1 else (BLUE, "o")
        forest.errorbar(
            estimate,
            index,
            xerr=[[estimate - interval[0]], [interval[1] - estimate]],
            fmt=marker,
            color=color,
            markersize=5,
            capsize=2.5,
            lw=1.15,
            zorder=2,
        )
        forest.text(interval[1] + 0.008, index, f"{estimate:+.3f}", va="center", fontsize=7)
    forest.set(
        yticks=y,
        yticklabels=[name for name, _, _, _ in estimates],
        xlabel="IC-SARR minus source $T$ / system",
        title="Five-fold paired effect",
        xlim=(-0.12, 0.32),
    )
    forest.invert_yaxis()
    forest.grid(axis="x", alpha=0.22)

    counts = np.zeros((7, 7), dtype=int)
    for row in rows:
        source_t = int(round(row["source_t"]))
        rollout_t = int(round(row["rollout_t"]))
        counts[rollout_t, source_t] += 1
    image = matrix_axis.imshow(counts, cmap="Blues", vmin=0, vmax=int(counts.max()), origin="lower", aspect="equal")
    for rollout_t in range(7):
        for source_t in range(7):
            count = counts[rollout_t, source_t]
            if count:
                matrix_axis.text(source_t, rollout_t, str(count), ha="center", va="center", fontsize=7.0, color="white" if count > counts.max() * 0.52 else INK)
    matrix_axis.plot([-0.5, 6.5], [-0.5, 6.5], color=GRAY, lw=0.9)
    wins = int(np.tril(counts, k=-1).sum())
    losses = int(np.triu(counts, k=1).sum())
    ties = int(np.trace(counts))
    matrix_axis.set(
        xticks=np.arange(7),
        yticks=np.arange(7),
        xlim=(-0.5, 6.5),
        ylim=(-0.5, 6.5),
        xlabel="Source-margin complete-pool $T$",
        ylabel="IC-SARR complete-pool $T$",
        title="Paired exact-system counts (B=6)",
    )
    matrix_axis.text(0.02, 0.98, f"{wins} wins / {ties} ties / {losses} losses", transform=matrix_axis.transAxes, va="top", fontsize=6.7, color=INK, bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85})
    figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.03, label="Systems")
    _save(figure, output)


def render_headroom(rows: list[dict[str, Any]], output: Path) -> None:
    """Render descriptive B=6 terminal headroom and MC gate usage diagnostics."""
    figure, (headroom_axis, gate_axis) = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)
    bins: list[tuple[str, Iterable[dict[str, Any]]]] = [
        ("0", (row for row in rows if row["ceiling"] - row["source_t"] <= 0)),
        ("1", (row for row in rows if row["ceiling"] - row["source_t"] == 1)),
        ("2", (row for row in rows if row["ceiling"] - row["source_t"] == 2)),
        ("3+", (row for row in rows if row["ceiling"] - row["source_t"] >= 3)),
    ]
    labels, means, counts, errors = [], [], [], []
    for index, (label, group) in enumerate(bins):
        group_rows = list(group)
        delta = np.array([row["rollout_t"] - row["source_t"] for row in group_rows], dtype=float)
        labels.append(label)
        means.append(float(delta.mean()) if len(delta) else np.nan)
        counts.append(len(delta))
        errors.append(bootstrap_interval(delta, 20260730 + index) if len(delta) > 1 else (np.nan, np.nan))
    x = np.arange(len(labels))
    lower = [mean - interval[0] if np.isfinite(mean) else 0 for mean, interval in zip(means, errors, strict=True)]
    upper = [interval[1] - mean if np.isfinite(mean) else 0 for mean, interval in zip(means, errors, strict=True)]
    headroom_axis.axhline(0, color=GRAY, lw=0.8)
    headroom_axis.bar(x, means, color=[GRAY, BLUE, GREEN, RED], alpha=0.9)
    headroom_axis.errorbar(x, means, yerr=[lower, upper], fmt="none", color=INK, capsize=2.5, lw=0.85)
    for index, count in enumerate(counts):
        headroom_axis.text(index, means[index] + (0.03 if means[index] >= 0 else -0.05), f"n={count}", ha="center", fontsize=7)
    headroom_axis.set(
        xticks=x,
        xticklabels=labels,
        xlabel="Source remaining finite-pool headroom",
        ylabel="Mean IC-SARR minus source $T$",
        title="Descriptive headroom strata",
    )
    headroom_axis.grid(axis="y", alpha=0.2)

    changed_by_round = np.zeros(6, dtype=int)
    stage_one_accepted = np.zeros(6, dtype=int)
    stage_two_evaluated = np.zeros(6, dtype=int)
    stage_two_passed = np.zeros(6, dtype=int)
    for row in rows:
        for decision in row["rollout_rounds"]:
            diagnostics = decision.get("selection_diagnostics")
            if not diagnostics:
                continue
            index = int(decision["round_index"]) - 1
            if diagnostics.get("selected_pair_id") != diagnostics.get("source_pair_id"):
                changed_by_round[index] += 1
            if diagnostics.get("fallback_reason") == "stage_one_accepted":
                stage_one_accepted[index] += 1
            if diagnostics.get("stage_two_used"):
                stage_two_evaluated[index] += 1
            if diagnostics.get("fallback_reason") is None:
                stage_two_passed[index] += 1
    rounds = np.arange(1, 7)
    gate_axis.plot(rounds, changed_by_round, marker="o", color=GREEN, label="action differs from source")
    gate_axis.plot(rounds, stage_one_accepted, marker="s", color=BLUE, label="stage-one accepted")
    gate_axis.plot(rounds, stage_two_evaluated, marker="^", color=RED, label="stage-two evaluated")
    gate_axis.plot(rounds, stage_two_passed, marker="x", color=GRAY, label="stage-two passed")
    gate_axis.set(
        xticks=rounds,
        xlabel="Query round",
        ylabel="States (out of 230)",
        title="MC-gate execution diagnostic",
        ylim=(0, 250),
    )
    gate_axis.grid(axis="y", alpha=0.2)
    gate_axis.legend(frameon=False, fontsize=6.25, ncol=2, loc="upper left")
    _save(figure, output)


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print auditable quantities used in captions and tables."""
    for name, source_key, rollout_key in (
        ("T", "source_t", "rollout_t"),
        ("final-causal", "source_f", "rollout_f"),
        ("D", "source_d", "rollout_d"),
        ("wall seconds", "source_time", "rollout_time"),
    ):
        source = np.mean([row[source_key] for row in rows])
        rollout = np.mean([row[rollout_key] for row in rows])
        print(f"{name}: source={source:.6f}; ic_sarr={rollout:.6f}; delta={rollout - source:+.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_system_rows(args.archive_dir)
    render_effects(rows, args.output_dir / "matpes_b6_effects.pdf")
    render_headroom(rows, args.output_dir / "matpes_b6_headroom_gate.pdf")
    print_summary(rows)


if __name__ == "__main__":
    main()
