"""Shared matplotlib helpers for consistent styling and color schemes."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def setup_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[float, float] = (12.0, 10.0),
) -> tuple[Figure, Axes | np.ndarray]:
    """
    Create a matplotlib figure with consistent sizing and font settings.

    - Figure size: 12 x 10 inches by default.
    - Global font sizes: ~24 for labels/titles, slightly smaller for ticks/legend.
    """
    mpl.rcParams.update(
        {
            "figure.figsize": figsize,
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
        }
    )
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, axes


def color_cycle(n: int, cmap_name: str = "tab10") -> list:
    """
    Generate a list of visually distinct colors.

    Uses a qualitative palette (default 'tab10') for up to ~10 series,
    and samples evenly from the chosen colormap beyond that.
    """
    if n <= 0:
        return []

    cmap = plt.get_cmap(cmap_name)
    # For qualitative maps like tab10/tab20, cycle through discrete colors.
    if cmap_name in {"tab10", "tab20", "Dark2", "Set2"}:
        return [cmap(i % cmap.N) for i in range(n)]

    # Otherwise sample evenly from 0..1.
    if n == 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]

