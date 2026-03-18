"""Plotting utilities for transport analysis (Onsager coefficients, correlations)."""

from __future__ import annotations

from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from pymatgen.core import Element

from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.schemas.transport import TransportDoc
from py_oats.utils.plotting import color_cycle, setup_figure

AnalyzerLike = Union[TransportAnalyzer, TransportDoc]


def _as_doc(obj: AnalyzerLike) -> TransportDoc:
    """Accept either TransportAnalyzer or TransportDoc and return a TransportDoc."""
    if isinstance(obj, TransportAnalyzer):
        return TransportDoc.from_analyzer(obj)
    return obj


def _resolve_index(doc: TransportDoc, s: Union[str, int, Element]) -> int:
    """Map species label / Element / index to integer index in doc.species."""
    if isinstance(s, int):
        return s
    if isinstance(s, Element):
        label = s.symbol
    else:
        label = str(s)
    if doc.mapping is None or label not in doc.mapping:
        raise KeyError(f"Species {label!r} not in mapping")
    return doc.mapping[label]


def plot_correlation_pair(
    analyzer_or_doc: AnalyzerLike,
    i: Union[str, int, Element],
    j: Union[str, int, Element],
    *,
    log: bool = False,
    ax: Any | None = None,
) -> Any:
    """
    Plot time correlation function(s) for a pair of species (i, j).

    - If i == j: plot both "total" and "self".
    - If i != j: plot "distinct".
    """
    doc = _as_doc(analyzer_or_doc)
    if doc.correlation_functions is None:
        raise ValueError("correlation_functions not present on TransportDoc/Analyzer.")

    idx_i = _resolve_index(doc, i)
    idx_j = _resolve_index(doc, j)
    s_i = doc.species[idx_i]
    s_j = doc.species[idx_j]
    key = (s_i, s_j)
    if key not in doc.correlation_functions:
        raise KeyError(f"No correlation data for pair {key}")

    cf = doc.correlation_functions[key]
    times = np.asarray(doc.times, dtype=float)
    if doc.time_step is not None:
        times = times * float(doc.time_step)
    if doc.step_skip is not None:
        times = times * int(doc.step_skip)

    if ax is None:
        fig, ax = setup_figure()
    else:
        fig = ax.figure

    colors = color_cycle(2)
    if idx_i == idx_j:
        if "total" in cf:
            ax.plot(times, cf["total"], label=f"{s_i}-{s_i} total", color=colors[0], linewidth=2)
        if "self" in cf:
            ax.plot(times, cf["self"], label=f"{s_i}-{s_i} self", color=colors[1], linewidth=2)
    else:
        if "distinct" not in cf:
            raise KeyError(f"No 'distinct' correlation for pair {key}")
        ax.plot(times, cf["distinct"], label=f"{s_i}-{s_j} distinct", color=colors[0], linewidth=2)

    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Correlation")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return ax


def plot_correlation_pairwise(
    analyzer_or_doc: AnalyzerLike,
    *,
    log: bool = False,
) -> Any:
    """
    Plot correlation functions for all distinct species pairs in one figure.

    - Auto terms: (s, s) use "total".
    - Cross terms: (s_i, s_j) use "distinct".
    """
    doc = _as_doc(analyzer_or_doc)
    if doc.correlation_functions is None:
        raise ValueError("correlation_functions not present on TransportDoc/Analyzer.")

    n_species = len(doc.species)
    fig, ax = setup_figure()
    colors = color_cycle(n_species * (n_species - 1) // 2 + n_species)
    c_idx = 0

    times = np.asarray(doc.times, dtype=float)
    if doc.time_step is not None:
        times = times * float(doc.time_step)
    if doc.step_skip is not None:
        times = times * int(doc.step_skip)

    for i in range(n_species):
        s_i = doc.species[i]
        # auto
        cf_auto = doc.correlation_functions.get((s_i, s_i))
        if cf_auto and "total" in cf_auto:
            ax.plot(times, cf_auto["total"], label=f"{s_i}-{s_i} total", color=colors[c_idx], linewidth=2)
            c_idx += 1
        # cross
        for j in range(i + 1, n_species):
            s_j = doc.species[j]
            cf = doc.correlation_functions.get((s_i, s_j))
            if cf and "distinct" in cf:
                ax.plot(times, cf["distinct"], label=f"{s_i}-{s_j} distinct", color=colors[c_idx], linewidth=2)
                c_idx += 1

    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Correlation")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return ax


def plot_all_correlations(analyzer_or_doc: AnalyzerLike) -> Any:
    """
    Plot all correlation functions (auto + cross) in a grid of subplots.

    Returns the matplotlib figure.
    """
    doc = _as_doc(analyzer_or_doc)
    if doc.correlation_functions is None:
        raise ValueError("correlation_functions not present on TransportDoc/Analyzer.")

    species = list(doc.species)
    n = len(species)
    subplot_inches = 7.0
    figsize = (subplot_inches * n, subplot_inches * n)
    fig, axes = setup_figure(nrows=n, ncols=n, figsize=figsize)
    axes_arr = np.atleast_2d(axes)

    times = np.asarray(doc.times, dtype=float)
    if doc.time_step is not None:
        times = times * float(doc.time_step)
    if doc.step_skip is not None:
        times = times * int(doc.step_skip)

    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            ax = axes_arr[i, j]
            key = (si, sj)
            cf = doc.correlation_functions.get(key)
            if not cf:
                ax.axis("off")
                continue
            colors = color_cycle(3)
            if i == j:
                if "total" in cf:
                    ax.plot(times, cf["total"], label="total", color=colors[0], linewidth=2)
                if "self" in cf:
                    ax.plot(times, cf["self"], label="self", color=colors[1], linewidth=2)
            else:
                if "distinct" in cf:
                    ax.plot(times, cf["distinct"], label="distinct", color=colors[0], linewidth=2)
            ax.set_title(f"{si}-{sj}")
            if i == n - 1:
                ax.set_xlabel("Time (fs)")
            if j == 0:
                ax.set_ylabel("Correlation")
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

    fig.tight_layout()
    return fig

