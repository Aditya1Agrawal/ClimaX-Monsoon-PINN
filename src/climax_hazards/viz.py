"""Shared matplotlib styling and grid-map drawing for the analysis script and the demo app."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# validated categorical slots 1-3 (blue, orange, aqua) and a neutral grey for reference lines
C_CLIMAX, C_LOGISTIC, C_CAL, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc3ee", "#2a78d6", "#0d3b75"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2481a", "#eb6834", "#e6e5e1", "#2a78d6", "#0d3b75"])
OBS = LinearSegmentedColormap.from_list("obs", ["#f3f2ef", "#3a3936"])
WARM = LinearSegmentedColormap.from_list("seq_orange", ["#fdf0e9", "#f5b08f", "#eb6834", "#8f330f"])  # temperatures
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
ALERT = "#e34948"
NICE = {"heatwave": "Heatwave", "extreme_wind": "Extreme wind", "extreme_precip": "Extreme rainfall"}


def apply_style() -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": "white"})


def cell_edges(c: np.ndarray) -> np.ndarray:
    d = np.diff(c).mean()
    return np.concatenate([c - d / 2, [c[-1] + d / 2]])


def draw_map(ax, field, lat, lon, cmap, vmin, vmax, annotate=True, fmt="{:.2f}", outline=None, alerts=None):
    """Cell map with values written in each cell. ``outline``: (i, j) cells to box; ``alerts``: bool (H, W) to mark with a dot."""
    im = ax.pcolormesh(cell_edges(lon), cell_edges(lat), np.ma.masked_invalid(field), cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors="white", linewidth=1.5)
    if annotate:
        for i in range(len(lat)):
            for j in range(len(lon)):
                v = field[i, j]
                if np.isfinite(v):
                    frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
                    dark = abs(v) > 0.6 * vmax if cmap is DIV else frac > 0.6
                    ax.text(lon[j], lat[i], fmt.format(v), ha="center", va="center", fontsize=7, color="white" if dark else TEXT)
    d = np.diff(lon).mean()
    for i, j in outline or []:
        ax.add_patch(plt.Rectangle((lon[j] - d / 2, lat[i] - d / 2), d, d, fill=False, ec=TEXT, lw=1.4))
    if alerts is not None:  # thick red frame inside each alert cell
        for i, j in zip(*np.nonzero(alerts)):
            ax.add_patch(plt.Rectangle((lon[j] - d * 0.44, lat[i] - d * 0.44), d * 0.88, d * 0.88, fill=False, ec=ALERT, lw=2.4, zorder=5))
    ax.set_aspect("equal")
    ax.set_xticks(lon[::2]); ax.set_yticks(lat[::2])
    ax.set_xticklabels([f"{x:.0f}°E" for x in lon[::2]]); ax.set_yticklabels([f"{y:.0f}°N" for y in lat[::2]])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im
