"""Render the paper-facing E53 matched-adjudicator comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tools.publication_figure_style import (
    PALETTE,
    apply_publication_style,
    finalize_figure,
    style_axis,
)

POLICIES = (
    "posterior_mean_target_margin",
    "matched_local_hull_probability",
    "delta_hull_active_search",
)
LABELS = {
    "posterior_mean_target_margin": "Target margin",
    "matched_local_hull_probability": "Local-prob",
    "delta_hull_active_search": "Delta-Hull",
}
STYLES = {
    "posterior_mean_target_margin": (PALETTE["neutral_dark"], "--", "^"),
    "matched_local_hull_probability": (PALETTE["red_strong"], "-.", "s"),
    "delta_hull_active_search": (PALETTE["blue_main"], "-", "o"),
}


def _validated_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "e53_objective_isolation_complete":
        raise ValueError("expected a complete E53 objective-isolation summary")
    if tuple(payload.get("policies", ())) != POLICIES:
        raise ValueError("unexpected E53 policy roster")
    panels = payload.get("panels", {})
    if set(panels) != {"development", "secondary"}:
        raise ValueError("E53 figure requires development and secondary panels")
    for panel in panels.values():
        if set(panel.get("budgets", {})) != {str(value) for value in range(1, 7)}:
            raise ValueError("E53 figure requires all six budgets")
    return payload


def render(*, summary_path: Path, output_path: Path) -> None:
    summary = _validated_summary(summary_path)
    budgets = np.arange(1, 7)
    apply_publication_style(font_size=8.2)
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.55))

    development = summary["panels"]["development"]["budgets"]
    for policy in POLICIES:
        color, linestyle, marker = STYLES[policy]
        axes[0].plot(
            budgets,
            [development[str(b)]["absolute_mean_T"][policy] for b in budgets],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.0,
            linewidth=1.65,
            label=LABELS[policy],
        )
    axes[0].set(
        xlabel="Query budget $B$",
        ylabel="Complete-pool confirmations $T$",
        xticks=budgets,
        title="(a) Matched objective comparison",
    )
    axes[0].legend(loc="upper left", ncol=1, handlelength=2.6)
    style_axis(axes[0], grid=True)

    axes[1].axhline(0.0, color=PALETTE["neutral_dark"], linewidth=0.8, zorder=0)
    panel_styles = {
        "development": ("230-system development", PALETTE["blue_main"], "-", "o"),
        "secondary": ("94-system secondary", PALETTE["teal"], "--", "s"),
    }
    for panel_name, (label, color, linestyle, marker) in panel_styles.items():
        rows = summary["panels"][panel_name]["budgets"]
        means = np.asarray(
            [rows[str(b)]["contrasts"]["delta_minus_local"]["mean_effect"] for b in budgets]
        )
        lows = np.asarray(
            [rows[str(b)]["contrasts"]["delta_minus_local"]["ci_low"] for b in budgets]
        )
        highs = np.asarray(
            [rows[str(b)]["contrasts"]["delta_minus_local"]["ci_high"] for b in budgets]
        )
        axes[1].plot(
            budgets,
            means,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.0,
            linewidth=1.65,
            label=label,
        )
        axes[1].fill_between(budgets, lows, highs, color=color, alpha=0.13, linewidth=0)
    axes[1].set(
        xlabel="Query budget $B$",
        ylabel=r"$\Delta T$: Delta-Hull $-$ Local-prob",
        xticks=budgets,
        title="(b) Effect of complete-pool adjudication",
    )
    axes[1].legend(loc="upper left", ncol=1, handlelength=2.6)
    style_axis(axes[1], grid=True)

    figure.tight_layout(pad=1.2, w_pad=2.0)
    finalize_figure(figure, output_path, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(summary_path=args.summary, output_path=args.output)
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
