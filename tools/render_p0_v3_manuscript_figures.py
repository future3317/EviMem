"""Render manuscript figures from the frozen MatPES P0-v3 artifacts.

This is post-processing only.  It reads complete raw core records and the
registered external summary/calibration JSONs; it never runs a policy or
changes an experiment protocol.  All inputs and outputs must remain outside
the Git repository except for the exported figures copied into the paper
repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

INK = "#172033"
GRID = "#D9DEE5"
SOURCE = "#62748A"
IC = "#2D6F9F"
DELTA = "#B57A19"
POSITIVE = "#1D7A5B"
NEGATIVE = "#A53A3A"
CAUTION = "#B57A19"
PALE_BLUE = "#DDEBF7"
PALE_GREEN = "#DDEFE7"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
    }
)


def _finish(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#98A4B2")
    axis.tick_params(colors=INK)
    axis.grid(axis="y", color=GRID, lw=0.55)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _paired_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def _metric(summary: dict[str, Any], budget: int, policy: str, metric: str) -> dict[str, Any]:
    return summary["core_curve"][str(budget)][policy]["metrics"][metric]


def load_core_curve_rows(core_root: Path) -> dict[int, list[dict[str, Any]]]:
    """Load the frozen per-system rows needed for budget-level contrasts."""
    rows_by_budget: dict[int, list[dict[str, Any]]] = {}
    expected_policies = {
        "source_margin",
        "delta_hull_active_search",
        "independent_confirmation_source_rollout",
    }
    for budget in range(1, 7):
        paths = sorted(core_root.glob(f"matpes-p0v2-core-fold*-b{budget}-main.json"))
        if len(paths) != 5:
            raise ValueError(f"expected five complete core files for B={budget}, found {len(paths)}")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if set(payload.get("active_policies", ())) != expected_policies:
                raise ValueError(f"unexpected policy roster in {path.name}")
            for system, result in payload["systems"].items():
                if system in seen:
                    raise ValueError(f"duplicate exact system at B={budget}: {system}")
                seen.add(system)
                rows.append(
                    {
                        "system": system,
                        "source": result["strategies"]["source_margin"],
                        "delta": result["strategies"]["delta_hull_active_search"],
                        "ic": result["strategies"]["independent_confirmation_source_rollout"],
                    }
                )
        if len(rows) != 230:
            raise ValueError(f"expected 230 exact systems at B={budget}, found {len(rows)}")
        rows_by_budget[budget] = rows
    roster = {row["system"] for row in rows_by_budget[1]}
    if any({row["system"] for row in rows} != roster for rows in rows_by_budget.values()):
        raise ValueError("budget curve system rosters are not identical")
    return rows_by_budget


def load_core_b6_rows(core_root: Path) -> list[dict[str, Any]]:
    paths = sorted(core_root.glob("matpes-p0v2-core-fold*-b6-main.json"))
    if len(paths) != 5:
        raise ValueError(f"expected five complete B=6 core files, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    expected = [
        "source_margin",
        "delta_hull_active_search",
        "independent_confirmation_source_rollout",
    ]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["active_policies"] != expected:
            raise ValueError(f"unexpected policy roster in {path.name}")
        fold = int(payload["config"]["crossfit_fold_index"])
        for system, result in payload["systems"].items():
            source = result["strategies"]["source_margin"]
            rollout = result["strategies"]["independent_confirmation_source_rollout"]
            rows.append(
                {
                    "system": system,
                    "fold": fold,
                    "source_t": float(source["oracle_pool_confirmed_discoveries"]),
                    "rollout_t": float(rollout["oracle_pool_confirmed_discoveries"]),
                }
            )
    if len(rows) != 230 or len({row["system"] for row in rows}) != 230:
        raise ValueError("B=6 core records are not a disjoint 230-system panel")
    return rows


def render_effects(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    figure, (forest, matrix) = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    estimates: list[tuple[str, float, tuple[float, float]]] = []
    for fold in range(5):
        values = np.asarray(
            [row["rollout_t"] - row["source_t"] for row in rows if row["fold"] == fold], dtype=float
        )
        estimates.append((f"Fold {fold + 1}", float(values.mean()), _paired_interval(values, 20260810 + fold)))
    pooled = _metric(summary, 6, "independent_confirmation_source_rollout", "T")["paired_vs_source"]
    estimates.append(("All 230", float(pooled["paired_mean_difference"]), tuple(pooled["paired_bootstrap_95ci"])))
    forest.axvline(0, color=SOURCE, lw=0.8)
    for index, (label, estimate, interval) in enumerate(estimates):
        color = POSITIVE if index == len(estimates) - 1 else IC
        forest.errorbar(
            estimate,
            index,
            xerr=[[estimate - interval[0]], [interval[1] - estimate]],
            fmt="D" if index == len(estimates) - 1 else "o",
            color=color,
            markersize=5,
            capsize=2.5,
            lw=1.15,
        )
        forest.text(interval[1] + 0.008, index, f"{estimate:+.3f}", va="center", fontsize=7)
    forest.set(
        yticks=np.arange(len(estimates)),
        yticklabels=[item[0] for item in estimates],
        xlabel="IC-SARR minus source complete-pool $T$ / system",
        title="Five-fold paired effect",
        xlim=(-0.08, 0.34),
    )
    forest.invert_yaxis()
    forest.grid(axis="x", alpha=0.22)

    counts = np.zeros((7, 7), dtype=int)
    for row in rows:
        counts[int(round(row["rollout_t"])), int(round(row["source_t"]))] += 1
    image = matrix.imshow(counts, cmap="Blues", vmin=0, vmax=int(counts.max()), origin="lower", aspect="equal")
    for rollout_t in range(7):
        for source_t in range(7):
            count = counts[rollout_t, source_t]
            if count:
                matrix.text(
                    source_t,
                    rollout_t,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if count > counts.max() * 0.52 else INK,
                )
    matrix.plot([-0.5, 6.5], [-0.5, 6.5], color=SOURCE, lw=0.9)
    wins = int(np.tril(counts, k=-1).sum())
    losses = int(np.triu(counts, k=1).sum())
    ties = int(np.trace(counts))
    matrix.set(
        xticks=np.arange(7),
        yticks=np.arange(7),
        xlim=(-0.5, 6.5),
        ylim=(-0.5, 6.5),
        xlabel="Source complete-pool $T$",
        ylabel="IC-SARR complete-pool $T$",
        title="Paired exact-system counts (B=6)",
    )
    matrix.text(
        0.02,
        0.98,
        f"{wins} wins / {ties} ties / {losses} losses",
        transform=matrix.transAxes,
        va="top",
        fontsize=6.7,
        color=INK,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85},
    )
    figure.colorbar(image, ax=matrix, fraction=0.046, pad=0.03, label="Systems")
    _save(figure, output)


def render_waterfall(summary: dict[str, Any], output: Path) -> None:
    source = _metric(summary, 6, "source_margin", "T")
    ic = _metric(summary, 6, "independent_confirmation_source_rollout", "T")
    source_d = _metric(summary, 6, "source_margin", "D")
    ic_d = _metric(summary, 6, "independent_confirmation_source_rollout", "D")
    source_f = _metric(summary, 6, "source_margin", "F")
    ic_f = _metric(summary, 6, "independent_confirmation_source_rollout", "F")
    values = {
        "Source margin": (source_d["source_mean"], source_f["source_mean"], source["source_mean"]),
        "IC-SARR": (ic_d["policy_mean"], ic_f["policy_mean"], ic["policy_mean"]),
    }
    figure, axes = plt.subplots(2, 1, figsize=(6.7, 3.35), sharex=True, sharey=True, constrained_layout=True)
    labels = ("Provisional $D$", "$D-F$", "Selected-history $F$", "$F-T$", "Complete-pool $T$")
    for axis, (policy, counts) in zip(axes, values.items(), strict=True):
        d_value, f_value, t_value = counts
        starts = (0.0, f_value, 0.0, t_value, 0.0)
        heights = (d_value, d_value - f_value, f_value, f_value - t_value, t_value)
        axis.bar(
            np.arange(5),
            heights,
            bottom=starts,
            color=(IC, NEGATIVE, IC, NEGATIVE, POSITIVE),
            width=0.62,
            edgecolor="white",
            linewidth=0.55,
        )
        axis.plot([0.31, 0.69], [d_value, d_value], color=SOURCE, lw=0.8)
        axis.plot([1.31, 1.69], [f_value, f_value], color=SOURCE, lw=0.8)
        axis.plot([2.31, 2.69], [f_value, f_value], color=SOURCE, lw=0.8)
        axis.plot([3.31, 3.69], [t_value, t_value], color=SOURCE, lw=0.8)
        axis.set_ylabel("Confirmations/system")
        axis.set_title(policy, loc="left", fontsize=9, fontweight="bold")
        axis.set_ylim(0, 5.25)
        axis.grid(axis="y", alpha=0.22)
        for index, (start, height) in enumerate(zip(starts, heights, strict=True)):
            if index in (0, 2, 4):
                axis.text(index, start + height + 0.10, f"{start + height:.3f}", ha="center", fontsize=7)
            elif height > 0.03:
                axis.text(index, start + height / 2, f"{height:+.3f}", ha="center", va="center", fontsize=6.8, color="white")
    axes[-1].set_xticks(np.arange(5), labels, fontsize=6.8)
    delta_t = ic["policy_mean"] - source["source_mean"]
    delta_f = ic_f["policy_mean"] - source_f["source_mean"]
    delta_d = ic_d["policy_mean"] - source_d["source_mean"]
    delta_ft = (ic_f["policy_mean"] - ic["policy_mean"]) - (source_f["source_mean"] - source["source_mean"])
    figure.text(
        0.50,
        0.50,
        f"$\\Delta D={delta_d:+.3f}$    $\\Delta F={delta_f:+.3f}$    $\\Delta T={delta_t:+.3f}$    $\\Delta(F-T)={delta_ft:+.3f}$",
        ha="center",
        va="center",
        fontsize=6.5,
        color=INK,
    )
    _save(figure, output)


def render_budget_curve(summary: dict[str, Any], core_root: Path, output: Path) -> None:
    budgets = np.arange(1, 7)
    policies = {
        "source_margin": ("Source margin", SOURCE, "-"),
        "delta_hull_active_search": ("Delta-Hull", DELTA, ":"),
        "independent_confirmation_source_rollout": ("IC-SARR", IC, "--"),
    }
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 4.85), constrained_layout=True)
    for axis, metric, title in zip(axes.flat[:3], ("T", "F", "D"), ("Complete-pool $T$", "Selected-history $F$", "Provisional $D$"), strict=True):
        for policy, (label, color, style) in policies.items():
            values = [_metric(summary, int(budget), policy, metric)["policy_mean"] if policy != "source_margin" else _metric(summary, int(budget), policy, metric)["source_mean"] for budget in budgets]
            axis.plot(budgets, values, label=label, color=color, linestyle=style, marker="o", ms=3.5, lw=1.3)
        axis.set(xticks=budgets, xlabel="Query budget $B$", ylabel="Mean confirmations/system", title=title)
        _finish(axis)
    axes.flat[0].legend(frameon=False, loc="upper left", handlelength=2.4)
    axis = axes.flat[3]
    for metric, label, color in (("T", "$\\Delta T$", POSITIVE), ("F", "$\\Delta F$", CAUTION), ("D", "$\\Delta D$", IC)):
        means = []
        lower = []
        upper = []
        for budget in budgets:
            paired = _metric(summary, int(budget), "independent_confirmation_source_rollout", metric)["paired_vs_source"]
            mean = float(paired["paired_mean_difference"])
            ci = paired["paired_bootstrap_95ci"]
            means.append(mean)
            lower.append(mean - float(ci[0]))
            upper.append(float(ci[1]) - mean)
        axis.errorbar(budgets, means, yerr=[lower, upper], label=label, color=color, marker="o", ms=3.2, lw=1.1, capsize=2.2)
    axis.axhline(0, color=SOURCE, lw=0.8)
    axis.set(xticks=budgets, xlabel="Query budget $B$", ylabel="IC-SARR minus source", title="Paired differences (95% CI)")
    axis.legend(frameon=False, ncol=1, loc="upper left")
    _finish(axis)
    curve_rows = load_core_curve_rows(core_root)
    axis = axes.flat[4]
    means = []
    lower = []
    upper = []
    for budget in budgets:
        values = np.asarray(
            [
                row["ic"]["oracle_pool_confirmed_discoveries"]
                - row["delta"]["oracle_pool_confirmed_discoveries"]
                for row in curve_rows[int(budget)]
            ],
            dtype=float,
        )
        mean = float(values.mean())
        interval = _paired_interval(values, 20260820 + int(budget))
        means.append(mean)
        lower.append(mean - interval[0])
        upper.append(interval[1] - mean)
    axis.errorbar(
        budgets,
        means,
        yerr=[lower, upper],
        color=DELTA,
        marker="o",
        ms=3.2,
        lw=1.2,
        capsize=2.2,
    )
    axis.axhline(0, color=SOURCE, lw=0.8)
    axis.set(
        xticks=budgets,
        xlabel="Query budget $B$",
        ylabel="IC-SARR minus Delta-Hull",
        title="Direct $T$ contrast (95% CI)",
    )
    _finish(axis)
    axis = axes.flat[5]
    means = []
    lower = []
    upper = []
    for budget in budgets:
        paired = _metric(summary, int(budget), "independent_confirmation_source_rollout", "wall_seconds")["paired_vs_source"]
        mean = float(paired["paired_mean_difference"])
        ci = paired["paired_bootstrap_95ci"]
        means.append(mean)
        lower.append(mean - float(ci[0]))
        upper.append(float(ci[1]) - mean)
    axis.errorbar(budgets, means, yerr=[lower, upper], color=NEGATIVE, marker="s", ms=3.2, lw=1.2, capsize=2.2)
    axis.axhline(0, color=SOURCE, lw=0.8)
    axis.set(xticks=budgets, xlabel="Query budget $B$", ylabel="Seconds/system", title="Additional wall time (95% CI)")
    _finish(axis)
    figure.suptitle("MatPES cross-fitted budget curve (230 development systems)", fontsize=10)
    _save(figure, output)


def render_ablation(summary: dict[str, Any], output: Path) -> None:
    audit = summary["reduced_b6_targeted_mechanism_audit"]
    order = [
        ("delta_hull_active_search", "Delta-Hull"),
        ("source_rollout_delta_hull", "Source rollout"),
        ("diagonal_ic_sarr", "Diagonal covariance"),
        ("ungated_source_rollout", "Ungated rollout"),
        ("independent_confirmation_source_rollout", "IC-SARR"),
    ]
    labels = [label for _, label in order]
    means = []
    lower = []
    upper = []
    for policy, _ in order:
        paired = audit[policy]["metrics"]["T"]["paired_vs_source"]
        means.append(paired["paired_mean_difference"])
        lower.append(paired["paired_mean_difference"] - paired["paired_bootstrap_95ci"][0])
        upper.append(paired["paired_bootstrap_95ci"][1] - paired["paired_mean_difference"])
    figure, axis = plt.subplots(figsize=(7.0, 2.8), constrained_layout=True)
    positions = np.arange(len(labels))
    axis.axhline(0, color=SOURCE, lw=0.8)
    axis.errorbar(positions, means, yerr=[lower, upper], fmt="none", color=INK, capsize=3, lw=1.0)
    axis.scatter(positions[:-1], means[:-1], color=SOURCE, s=28, zorder=3)
    axis.scatter(positions[-1], means[-1], color=POSITIVE, s=42, marker="D", zorder=3)
    for position, value in zip(positions, means, strict=True):
        axis.text(position, value + 0.017, f"{value:+.3f}", ha="center", fontsize=7)
    axis.set(xticks=positions, xticklabels=labels, ylabel="B=6 paired $\\Delta T$ / system", title="Targeted MatPES mechanism audit (5 folds, 230 systems)")
    axis.set_ylim(-0.02, 0.31)
    axis.grid(axis="y", alpha=0.22)
    _save(figure, output)


def render_direct_mechanism_comparisons(direct_summary: dict[str, Any], output: Path) -> None:
    order = [
        ("ic_sarr_vs_delta_hull", "IC-SARR\nvs Delta-Hull"),
        ("ic_sarr_vs_ungated_sarr", "IC-SARR\nvs Ungated SARR"),
        ("ic_sarr_vs_diagonal_covariance", "IC-SARR\nvs Diagonal covariance"),
    ]
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    p_values: list[float] = []
    for key, _ in order:
        paired = direct_summary["comparisons"][key]["metrics"]["T"]["direct_paired"]
        means.append(float(paired["paired_mean_difference"]))
        lower.append(means[-1] - float(paired["paired_bootstrap_95ci"][0]))
        upper.append(float(paired["paired_bootstrap_95ci"][1]) - means[-1])
        p_values.append(float(paired["two_sided_sign_flip_p"]))
    figure, axis = plt.subplots(figsize=(7.0, 2.8), constrained_layout=True)
    positions = np.arange(len(order))
    colors = [POSITIVE, NEGATIVE, IC]
    axis.axhline(0, color=SOURCE, lw=0.8)
    axis.errorbar(
        positions,
        means,
        yerr=[lower, upper],
        fmt="none",
        color=INK,
        capsize=3,
        lw=1.0,
    )
    axis.scatter(positions, means, color=colors, s=40, zorder=3)
    for position, mean, p_value in zip(positions, means, p_values, strict=True):
        label = f"{mean:+.3f}\np={p_value:.3f}"
        offset = 0.012 if mean >= 0.0 else -0.016
        axis.text(position, mean + offset, label, ha="center", va="bottom" if mean >= 0.0 else "top", fontsize=6.7)
    axis.set(
        xticks=positions,
        xticklabels=[label for _, label in order],
        ylabel=r"Direct paired $\,\Delta T$ / system",
        title="Direct MatPES mechanism contrasts (B=6; 230 systems)",
        ylim=(-0.10, 0.15),
    )
    axis.grid(axis="y", alpha=0.22)
    _save(figure, output)


def render_calibration(calibration_path: Path, output: Path) -> None:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    deciles = summary["deciles"]
    deciles = sorted(deciles, key=lambda row: row["decile"])
    figure, (calibration, strata) = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    x = np.asarray([row["predicted_mean"] for row in deciles], dtype=float)
    t = np.asarray([row["actual_t_mean"] for row in deciles], dtype=float)
    f = np.asarray([row["actual_f_mean"] for row in deciles], dtype=float)
    calibration.axhline(0, color=SOURCE, lw=0.8)
    calibration.plot(x, t, color=POSITIVE, marker="o", ms=3.8, lw=1.3, label="Realized $T$ advantage")
    calibration.plot(x, f, color=CAUTION, marker="s", ms=3.8, lw=1.3, label="Realized $F$ advantage")
    calibration.set(xlabel="Predicted rollout advantage (decile mean)", ylabel="Realized advantage", title="Predicted vs. realized value")
    calibration.legend(frameon=False, fontsize=6.6)
    _finish(calibration)
    labels = ["Accepted", "Rejected/screened"]
    accepted = summary["strata"]["accepted"]
    rejected = summary["strata"]["rejected_or_screened"]
    t_means = [accepted["actual_t_mean"], rejected["actual_t_mean"]]
    f_means = [accepted["actual_f_mean"], rejected["actual_f_mean"]]
    positions = np.arange(2)
    width = 0.34
    strata.axhline(0, color=SOURCE, lw=0.8)
    strata.bar(positions - width / 2, t_means, width, color=POSITIVE, label="Realized $T$")
    strata.bar(positions + width / 2, f_means, width, color=CAUTION, label="Realized $F$")
    strata.set(xticks=positions, xticklabels=labels, ylabel="Realized advantage", title="Gate strata")
    strata.legend(frameon=False, fontsize=6.6)
    _finish(strata)
    figure.suptitle(
        f"Post-hoc development calibration audit (n={summary['decision_state_count']}; accepted={summary['accepted_deviation_count']})",
        fontsize=9.5,
    )
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--direct-summary", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    direct_summary = json.loads(args.direct_summary.read_text(encoding="utf-8"))
    rows = load_core_b6_rows(args.core_root)
    render_effects(rows, summary, args.output_dir / "matpes_b6_effects.pdf")
    render_waterfall(summary, args.output_dir / "matpes_dft_waterfall.pdf")
    render_budget_curve(summary, args.core_root, args.output_dir / "matpes_budget_curve.pdf")
    render_ablation(summary, args.output_dir / "matpes_mechanism_ablation.pdf")
    render_direct_mechanism_comparisons(direct_summary, args.output_dir / "matpes_direct_mechanism_comparisons.pdf")
    render_calibration(args.calibration, args.output_dir / "matpes_rollout_calibration.pdf")


if __name__ == "__main__":
    main()
