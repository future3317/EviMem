"""Shared publication plotting conventions for delayed-label figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#767676",
    "charcoal": "#272727",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#D8B26A",
    "paper": "#FAF7F0",
}


def apply_publication_style(font_size: float = 8.0) -> None:
    """Apply a compact, print-safe matplotlib style used by the paper."""

    plt.rcParams.update(
        {
            # DejaVu Sans is bundled with the analysis environment and is a
            # metrically stable Arial/Helvetica substitute for the PDF build.
            "font.family": ["DejaVu Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 0.5,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["charcoal"],
            "axes.labelcolor": PALETTE["charcoal"],
            "xtick.color": PALETTE["charcoal"],
            "ytick.color": PALETTE["charcoal"],
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.frameon": False,
            "legend.fontsize": font_size - 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes, *, grid: bool = False) -> None:
    """Apply shared axes details after plotting."""

    axis.spines["left"].set_linewidth(0.85)
    axis.spines["bottom"].set_linewidth(0.85)
    axis.tick_params(width=0.75, length=3, pad=2)
    if grid:
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=PALETTE["neutral"], linewidth=0.45, alpha=0.65)


def finalize_figure(
    figure: plt.Figure,
    output: Path,
    *,
    dpi: int = 300,
    pad_inches: float = 0.035,
) -> None:
    """Write stable PDF and PNG siblings with a tight, white canvas."""

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    figure.savefig(
        output.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )
    plt.close(figure)
