"""Figures for the SAGE audit.

The sample grids contain patient images, so everything drawn here must stay in the
git-ignored outputs/ folder. Figures use matplotlib's object API (no pyplot), so
importing this module never changes the active backend.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from sage_shift.data import SPLITS

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
SURFACE = "#ffffff"
SERIES = {"train": "#2a78d6", "test": "#eb6834"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STYLE = {
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}


def _luminance(colour: tuple[float, float, float]) -> float:
    """WCAG relative luminance of an sRGB colour with channels in 0-1."""
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in colour]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def plot_class_bars(label_counts: pd.DataFrame, n_images: dict[str, int], path: Path) -> None:
    """Grouped horizontal bars: share of each split's images that carry each label."""
    rows = label_counts.iloc[::-1].reset_index(drop=True)  # largest class at the top
    positions = np.arange(len(rows))
    thickness = 0.34
    with matplotlib.rc_context(STYLE):
        figure = Figure(figsize=(8.0, 0.42 * len(rows) + 1.5), dpi=150)
        axis = figure.subplots()
        for split, side in zip(SPLITS, (1, -1)):
            axis.barh(
                positions + side * (thickness / 2 + 0.02),
                rows[f"{split}_pct"],
                height=thickness,
                color=SERIES[split],
                linewidth=0,
                label=f"{split} ({n_images[split]:,} images)",
            )
        for position, row in zip(positions, rows.itertuples()):
            if row.low_test_support:
                axis.text(
                    max(row.train_pct, row.test_pct) + 0.5,
                    position,
                    f"low test support: {row.test_images} images, {row.test_patients} patients",
                    va="center",
                    fontsize=8,
                    color=INK_SECONDARY,
                )
        axis.set_yticks(positions, rows["label"])
        axis.set_ylim(-0.7, len(rows) - 0.3)
        axis.set_xlabel("Images carrying the label (% of the split)")
        axis.set_title("SAGE label prevalence, train vs test", loc="left")
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(length=0)
        for side_name in ("top", "right", "bottom"):
            axis.spines[side_name].set_visible(False)
        axis.legend(loc="lower right", frameon=False)
        figure.savefig(path, bbox_inches="tight")


def plot_cooccurrence(cooccurrence: pd.DataFrame, path: Path) -> None:
    """Lower-triangle heatmap of label co-occurrence counts; the diagonal shows class totals."""
    labels = list(cooccurrence.index)
    counts = cooccurrence.to_numpy()
    size = len(labels)
    lower = np.tril(np.ones((size, size), dtype=bool), k=-1)
    shown = np.ma.masked_where(~lower | (counts == 0), counts)
    colours = LinearSegmentedColormap.from_list("sage_blue", BLUE_RAMP)
    peak = max(int(counts[lower].max()), 1)
    with matplotlib.rc_context(STYLE):
        figure = Figure(figsize=(9.0, 7.6), dpi=150)
        axis = figure.subplots()
        mesh = axis.pcolormesh(shown, cmap=colours, vmin=0, vmax=peak, edgecolors=SURFACE, linewidth=1.5)
        for index in range(size):
            axis.add_patch(Rectangle((index, index), 1, 1, facecolor=NEUTRAL, edgecolor=SURFACE, linewidth=1.5))
            axis.text(
                index + 0.5, index + 0.5, str(counts[index, index]),
                ha="center", va="center", fontsize=6.5, color=INK_SECONDARY,
            )
        for row, column in zip(*np.nonzero(lower & (counts > 0))):
            fill = to_rgb(colours(counts[row, column] / peak))
            axis.text(
                column + 0.5, row + 0.5, str(counts[row, column]),
                ha="center", va="center", fontsize=6.5, color=SURFACE if _luminance(fill) < 0.2 else INK,
            )
        ticks = np.arange(size) + 0.5
        axis.set_xticks(ticks, labels, rotation=45, ha="right", rotation_mode="anchor")
        axis.set_yticks(ticks, labels)
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title("SAGE label co-occurrence", loc="left", pad=20)
        axis.text(
            0.0, 1.012, "Images carrying both labels. Grey diagonal: images carrying the label.",
            transform=axis.transAxes, fontsize=8.5, color=INK_SECONDARY, va="bottom",
        )
        colourbar = figure.colorbar(mesh, ax=axis, shrink=0.55, pad=0.02)
        colourbar.set_label("Images carrying both labels")
        colourbar.outline.set_visible(False)
        figure.savefig(path, bbox_inches="tight")


def pick_grid_samples(table: pd.DataFrame, split: str, seed: int, n: int = 16) -> pd.DataFrame:
    """Draw `n` rows of one split at random, reproducibly for a given seed."""
    pool = table[table["split"] == split].sort_values("sample_uuid")
    chosen = np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False)
    return pool.iloc[np.sort(chosen)]


def plot_sample_grid(
    image_paths: Sequence[Path], captions: Sequence[str], title: str, path: Path, n_columns: int = 4
) -> None:
    """Grid of images with a caption under each one. The output contains patient images."""
    n_rows = -(-len(image_paths) // n_columns)
    with matplotlib.rc_context(STYLE):
        figure = Figure(figsize=(3.3 * n_columns, 3.0 * n_rows + 0.5), dpi=110, layout="constrained")
        axes = figure.subplots(n_rows, n_columns, squeeze=False).ravel()
        for axis in axes:
            axis.set_visible(False)
        for axis, image_path, caption in zip(axes, image_paths, captions):
            axis.set_visible(True)
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_xlabel("\n".join(textwrap.wrap(caption, 36)), fontsize=9, color=INK)
        figure.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
        figure.savefig(path)
