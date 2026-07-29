"""Plotting utilities for coordination environment analysis (RDF, cage time)."""

from __future__ import annotations

import math
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np

from py_oats.analyzers.coordination import CoordinationAnalyzer, TOTAL_RDF_KEY
from py_oats.schemas.coordination import CoordinationAnalysisDoc, RDFResult
from py_oats.utils.plotting import color_cycle, setup_figure

SourceLike = Union[CoordinationAnalyzer, CoordinationAnalysisDoc]


def _as_doc(obj: SourceLike) -> CoordinationAnalysisDoc:
    if isinstance(obj, CoordinationAnalyzer):
        return CoordinationAnalysisDoc.from_analyzer(obj)
    return obj


def _pair_keys(doc: CoordinationAnalysisDoc, include_total: bool = False) -> list[str]:
    """Sorted pair keys from doc.rdfs, optionally including 'total'."""
    keys = [k for k in doc.rdfs if k != TOTAL_RDF_KEY]
    keys.sort()
    if include_total and TOTAL_RDF_KEY in doc.rdfs:
        keys.append(TOTAL_RDF_KEY)
    return keys


def _add_cn_axis(ax: Any, result: RDFResult) -> Any:
    """Add a secondary y-axis with the running coordination number."""
    ax2 = ax.twinx()
    ax2.plot(result.r, result.coordination_number, color="0.6", linewidth=1.2,
             linestyle="--", label="CN")
    ax2.set_ylabel("CN", fontsize=16)
    ax2.tick_params(labelsize=14)
    return ax2


def plot_rdf(
    source: SourceLike,
    pairs: list[str] | None = None,
    *,
    show_coordination_number: bool = False,
    ax: Any | None = None,
) -> Any:
    """
    Plot g(r) for one or more species pairs on a single axes.

    Args:
        source: A post-``analyze()`` ``CoordinationAnalyzer`` or a
            ``CoordinationAnalysisDoc``.
        pairs: List of pair keys to plot (e.g. ``["Li-O", "Mn-O"]``).
            Defaults to all species pairs excluding ``"total"``.
            Pass ``["total"]`` or include it explicitly to show the total g(r).
        show_coordination_number: If True, overlay the running coordination
            number on a secondary y-axis.  Only meaningful when plotting a
            single pair; ignored when plotting multiple.
        ax: Existing ``Axes`` to draw on.  A new figure is created if None.

    Returns:
        The ``Axes`` object.
    """
    doc = _as_doc(source)

    if pairs is None:
        pairs = _pair_keys(doc)
    if not pairs:
        raise ValueError("No pairs to plot.")

    missing = [p for p in pairs if p not in doc.rdfs]
    if missing:
        raise KeyError(f"Pairs not found in source: {missing}. Available: {list(doc.rdfs)}")

    if ax is None:
        _, ax = setup_figure()

    colors = color_cycle(len(pairs))
    for color, key in zip(colors, pairs):
        result = doc.rdfs[key]
        ax.plot(result.r, result.rdf, label=f"g(r) {key}", color=color, linewidth=2)

    if show_coordination_number and len(pairs) == 1:
        _add_cn_axis(ax, doc.rdfs[pairs[0]])

    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.axhline(1.0, color="0.7", linewidth=1, linestyle=":")
    ax.legend()
    ax.figure.tight_layout()
    return ax


def plot_rdf_grid(
    source: SourceLike,
    *,
    include_total: bool = True,
    show_coordination_number: bool = False,
) -> Any:
    """
    Plot every species pair in a grid of subplots, one panel per pair.

    Args:
        source: A post-``analyze()`` ``CoordinationAnalyzer`` or a
            ``CoordinationAnalysisDoc``.
        include_total: If True (default), add a final panel for the total g(r).
        show_coordination_number: If True, overlay the running coordination
            number on a secondary y-axis in each panel.

    Returns:
        The matplotlib ``Figure``.
    """
    doc = _as_doc(source)
    keys = _pair_keys(doc, include_total=include_total)

    n = len(keys)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    subplot_w, subplot_h = 7.0, 6.0
    fig, axes = setup_figure(
        nrows=nrows, ncols=ncols,
        figsize=(subplot_w * ncols, subplot_h * nrows),
    )
    axes_flat = np.array(axes).flatten()

    colors = color_cycle(n)
    for idx, (key, color) in enumerate(zip(keys, colors)):
        ax = axes_flat[idx]
        result = doc.rdfs[key]
        ax.plot(result.r, result.rdf, color=color, linewidth=2)
        ax.axhline(1.0, color="0.7", linewidth=1, linestyle=":")

        # Mark peaks
        for pr, ph in zip(result.peak_r, result.peak_rdf):
            ax.axvline(pr, color=color, linewidth=0.8, linestyle="--", alpha=0.5)

        if show_coordination_number:
            _add_cn_axis(ax, result)

        ax.set_title(key, fontsize=20)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")

    # Turn off any unused subplot panels
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig
