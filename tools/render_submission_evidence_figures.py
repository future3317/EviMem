"""Render the frozen-evidence figures used by the manuscript.

This renderer does not execute an experiment or infer a new result.  Its MAD
panel reads the already frozen curve summary; the evidence ladder and MatPES
D/F/T panel use the exact aggregate values recorded in the manuscript and
experiment ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

INK = "#172033"
GRID = "#D9DEE5"
SOURCE = "#62748A"
IC = "#2D6F9F"
POSITIVE = "#1D7A5B"
NEGATIVE = "#A53A3A"
CAUTION = "#B57A19"
PALE_BLUE = "#DDEBF7"
PALE_GREEN = "#DDEFE7"
PALE_RED = "#F7E1E1"
PALE_GOLD = "#F6ECD7"


def _finish(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#98A4B2")
    axis.tick_params(colors=INK)
    axis.grid(axis="y", color=GRID, lw=0.55)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_evidence_ladder(output: Path) -> None:
    """Render the claim map without implying that its evidence layers add up."""

    rows = (
        (
            "Exact finite-pool separation",
            "Can retained state matter\nfor later acquisition?",
            "Yes, against the declared\nnonadaptive comparator\n(B=3, K=1: 2 vs 5/3).",
            "SUPPORTED",
            PALE_BLUE,
            POSITIVE,
        ),
        (
            "WBM homogeneous null",
            "Is bounded posterior memory\na measured bottleneck here?",
            "No. Full history is the\ncorrect operating null.",
            "NULL",
            PALE_GOLD,
            CAUTION,
        ),
        (
            "JARVIS--MP transport",
            "Does certified protocol transport\nbeat simple reuse on fresh systems?",
            "No. The certificate is non-degenerate,\nbut superiority and coverage gates fail.",
            "NO-GO",
            PALE_RED,
            NEGATIVE,
        ),
        (
            "MatPES development",
            "Can delayed full-pool utility\nchange acquisition behavior?",
            "Yes: oracle-final +0.161/system\nover 230 development systems.",
            "MECHANISM",
            PALE_GREEN,
            POSITIVE,
        ),
        (
            "MAD protocol shift",
            "Does that mechanism transport\nunder a frozen public protocol task?",
            "Small oracle AUC +0.224; no\nfinal-causal or cost-aware advantage.",
            "BOUNDARY",
            PALE_GOLD,
            CAUTION,
        ),
    )
    figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.95, len(rows) - 0.42)
    axis.axis("off")
    column_x = (0.02, 0.30, 0.56, 0.89)
    headers = ("Evidence layer", "Question", "Warranted conclusion", "Status")
    for x, header in zip(column_x, headers, strict=True):
        axis.text(x, len(rows) - 0.17, header, fontsize=8.1, fontweight="bold", color=INK)
    for index, (title, question, conclusion, status, face, accent) in enumerate(rows):
        y = len(rows) - 1 - index
        axis.add_patch(
            plt.Rectangle((0.0, y - 0.40), 0.995, 0.80, facecolor="#FFFFFF", edgecolor=GRID, lw=0.65)
        )
        axis.text(
            column_x[0], y, title, va="center", ha="left", fontsize=8.1, fontweight="bold", color=INK
        )
        axis.text(column_x[1], y, question, va="center", ha="left", fontsize=7.8, color=INK)
        axis.text(column_x[2], y, conclusion, va="center", ha="left", fontsize=7.8, color=INK)
        axis.text(
            column_x[3],
            y,
            status,
            va="center",
            ha="center",
            fontsize=7.0,
            fontweight="bold",
            color=accent,
            bbox={"boxstyle": "round,pad=0.28", "fc": face, "ec": accent, "lw": 0.95},
        )
    axis.text(
        0.0,
        len(rows) + 0.27,
        "What the evidence establishes - and what it does not",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.0,
        -0.79,
        "The rows are evidence boundaries, not additive proof of bounded-memory or deployment superiority.",
        fontsize=7.8,
        color=INK,
    )
    _save(figure, output)


def render_matpes_dft(output: Path) -> None:
    """Render the frozen MatPES D/F/T mechanism decomposition."""

    labels = ("Online\nannouncements D", "Selected-history\nretained F", "Full-pool\nadjudicated T")
    source = np.asarray((4.322, 4.083, 3.622))
    ic = np.asarray((4.643, 4.096, 3.783))
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    positions = np.arange(3)
    width = 0.34
    left.bar(positions - width / 2, source, width, label="Source margin", color=SOURCE)
    left.bar(positions + width / 2, ic, width, label="IC-SARR", color=IC)
    left.set_xticks(positions, labels)
    left.set_ylim(0, 5.1)
    left.set_ylabel("Confirmations per system")
    left.set_title("MatPES development: D/F/T counts (n=230)")
    left.legend(frameon=False, loc="lower right")
    for index, value in enumerate(ic - source):
        left.text(
            index,
            max(source[index], ic[index]) + 0.16,
            f"Δ={value:+.3f}",
            ha="center",
            fontsize=8,
            color=INK,
        )
    _finish(left)

    loss_labels = ("Within-campaign\nrevocation D−F", "Unqueried-pool\ninvalidation F−T")
    source_loss = np.asarray((0.239, 0.461))
    ic_loss = np.asarray((0.547, 0.313))
    positions = np.arange(2)
    right.bar(positions - width / 2, source_loss, width, label="Source margin", color=SOURCE)
    right.bar(positions + width / 2, ic_loss, width, label="IC-SARR", color=IC)
    right.set_xticks(positions, loss_labels)
    right.set_ylim(0, 0.72)
    right.set_ylabel("Revoked confirmations/system")
    right.set_title("Where the terminal difference comes from")
    for index, value in enumerate(ic_loss - source_loss):
        right.text(
            index,
            max(source_loss[index], ic_loss[index]) + 0.035,
            f"Δ={value:+.3f}",
            ha="center",
            fontsize=8,
            color=INK,
        )
    right.text(
        0.5,
        -0.31,
        "ΔT = +0.161: more provisional announcements, but fewer invalidations by unqueried competitors.",
        transform=right.transAxes,
        ha="center",
        fontsize=7.7,
        color=INK,
    )
    _finish(right)
    _save(figure, output)


def _curve(summary: dict[str, Any], policy: str, metric: str) -> np.ndarray:
    return np.asarray(summary["curves"][policy][metric], dtype=float)


def render_mad_curve(summary_path: Path, output: Path) -> None:
    """Render frozen paired MAD differences with their uncertainty intervals."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "development_only_mad15_curve_summary":
        raise ValueError("expected the frozen MAD development curve summary")
    budgets = np.asarray(summary["budgets"], dtype=int)
    nonzero_budgets = budgets[budgets > 0]
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), constrained_layout=True)
    paired = summary["paired_by_budget"]
    panels = (
        ("oracle_difference_ic_minus_source", "IC-SARR minus source $T$ / system", "Complete-pool $\\Delta T$", "+0.2240 AUC\n95% CI [+0.0365, +0.4167]\np=0.0230"),
        ("final_causal_difference_ic_minus_source", "IC-SARR minus source $F$ / system", "Selected-history $\\Delta F$", "+0.0260 AUC\n95% CI [−0.0313, +0.0833]\np=0.5320"),
    )
    for axis, (metric, ylabel, title, annotation) in zip(axes[:2], panels, strict=True):
        means = np.asarray([paired[str(budget)][metric]["mean"] for budget in nonzero_budgets], dtype=float)
        intervals = np.asarray([paired[str(budget)][metric]["bootstrap_95pct"] for budget in nonzero_budgets], dtype=float)
        error = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
        axis.axhline(0, color=SOURCE, lw=0.85, zorder=0)
        axis.errorbar(nonzero_budgets, means, yerr=error, color=IC, marker="s", ms=3.8, lw=1.3, capsize=2.2)
        axis.set_xticks(nonzero_budgets)
        axis.set_xlabel("Query budget B")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.text(0.04, 0.95, annotation, transform=axis.transAxes, va="top", fontsize=7.0, color=INK)
        _finish(axis)
    time_delta = np.asarray([paired[str(budget)]["mean_wall_seconds_difference_ic_minus_source"] for budget in nonzero_budgets], dtype=float)
    axes[2].axhline(0, color=SOURCE, lw=0.85, zorder=0)
    axes[2].plot(nonzero_budgets, time_delta, marker="s", ms=3.8, lw=1.5, color=NEGATIVE)
    axes[2].set(xticks=nonzero_budgets, xlabel="Query budget B", ylabel="IC-SARR minus source\nseconds / system", title="Wall-time $\\Delta$")
    axes[2].text(0.04, 0.95, "+8.5739 AUC\n$U_{\\mathrm{AUC}}=-0.6334$", transform=axes[2].transAxes, va="top", fontsize=7.0, color=INK)
    _finish(axes[2])
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mad-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_evidence_ladder(args.output_dir / "evidence_ladder.pdf")
    render_matpes_dft(args.output_dir / "matpes_dft_decomposition.pdf")
    render_mad_curve(args.mad_summary, args.output_dir / "mad15_frozen_curve.pdf")


if __name__ == "__main__":
    main()
