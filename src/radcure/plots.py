"""
Shared figure helpers, so every figure in `analysis/01`-`07` uses the same
style, size conventions and colours.

Design rules kept deliberately small:
  - matplotlib only, no additional plotting dependency;
  - every figure is saved to `figures/` AND shown, so the same code works
    when a script is run headless and when its `# %%` cells are executed
    interactively in VS Code;
  - one consistent colour for "event / positive / worse" and one for
    "non-event / negative / better", used the same way everywhere.

Figures are presentation output. Nothing here computes a scientific quantity
or invents a value -- every number plotted is supplied by the caller, whether
freshly computed in the artefact-producing stage (`analysis/01`) or reloaded
from a persisted artefact in a later stage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import config

# A small, consistent palette. EVENT is used wherever the two-year
# mortality event / positive class / poorer operating point is shown;
# NON_EVENT for its counterpart; ACCENT for a selected or highlighted item.
EVENT_COLOR = "#c0392b"
NON_EVENT_COLOR = "#2c7fb8"
ACCENT_COLOR = "#d95f02"
NEUTRAL_COLOR = "#7f7f7f"
MODEL_COLORS = ["#2c7fb8", "#7fcdbb", "#d95f02", "#756bb1", "#31a354"]

FIGURE_DPI = 150


def apply_project_style() -> None:
    """Apply the shared matplotlib style. Called once per analysis script."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def save_figure(fig, name: str, *, show: bool = True) -> Path:
    """Save `fig` to `figures/<name>` and display it.

    `plt.show()` renders the figure inline when the script is run as
    `# %%` cells in VS Code, and is a harmless no-op under a
    non-interactive backend when the script is run headless.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[figure] {path.relative_to(config.PROJECT_ROOT)}")
    return path


def annotated_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    value_format: str = "{:.4f}",
    cmap: str = "viridis",
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Draw a small heatmap with every cell value written in.

    Used for the SVC `C` x `gamma` grid-search surface and the out-of-fold
    score-correlation matrix -- both small enough that printing the numbers
    is more informative than a colour scale alone.

    `vmin`/`vmax` pin the colour scale. Pass them whenever several panels
    are meant to be compared with each other: with per-panel autoscaling,
    the brightest cell of a weak panel looks identical to the brightest
    cell of a strong one.
    """
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), labels=col_labels)
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.grid(False)

    # Choose each label's colour from the LUMINANCE of the cell actually
    # rendered behind it, rather than from where the value sits in the range.
    # The latter silently assumes a dark-to-light colormap and produces white
    # text on a pale background for any colormap that runs the other way.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(
                j, i, value_format.format(value),
                ha="center", va="center", fontsize=8,
                color="white" if luminance < 0.5 else "black",
            )
    if colorbar_label:
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)


def confusion_matrix_panel(ax, tn: int, fp: int, fn: int, tp: int, title: str) -> None:
    """Draw one 2x2 confusion matrix with counts and row percentages.

    Cells are shaded by their share of the ROW (the actual class), so the
    panel reads as "of the actual events, how many were caught" rather than
    being dominated by the majority class.
    """
    counts = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)

    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1], labels=["Predicted\nnon-event", "Predicted\nevent"])
    ax.set_yticks([0, 1], labels=["Actual\nnon-event", "Actual\nevent"])
    ax.set_title(title)
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{int(counts[i, j])}\n({shares[i, j] * 100:.1f}%)",
                ha="center", va="center", fontsize=11,
                color="white" if shares[i, j] > 0.5 else "black",
            )
