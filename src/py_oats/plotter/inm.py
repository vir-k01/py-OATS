"""Plotting utilities for INM density of states (total + element-resolved)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from py_oats.analyzers.inm import INMAnalyzer
from py_oats.schemas.inm import INMDoc, INMDOS
from py_oats.utils.plotting import color_cycle, setup_figure


AnalyzerLike = Union[INMAnalyzer, INMDoc]
DOSBranch = Literal["total", "stable", "unstable"]


def _as_doc(
    obj: AnalyzerLike,
    *,
    dos_bins: int = 400,
    dos_omega_ps_range: tuple[float, float] | None = None,
) -> INMDoc:
    """Accept either INMAnalyzer or INMDoc and return an INMDoc."""
    if isinstance(obj, INMAnalyzer):
        return INMDoc.from_analyzer(obj, dos_bins=dos_bins, dos_omega_ps_range=dos_omega_ps_range)
    return obj


def _select_dos(doc: INMDoc, frame_index: Optional[int]) -> INMDOS:
    """
    Pick a DOS entry from doc.

    - frame_index=None -> trajectory-average (stored with frame_index=None)
    - otherwise -> first match of that frame index
    """
    if frame_index is None:
        for d in doc.dos:
            if d.frame_index is None:
                return d
        raise KeyError("No trajectory-average DOS entry (frame_index=None) found.")

    for d in doc.dos:
        if d.frame_index == frame_index:
            return d
    raise KeyError(f"No DOS entry found for frame_index={frame_index}.")


def plot_inm_dos(
    analyzer_or_doc: AnalyzerLike,
    *,
    frame_index: int | None = None,
    branch: DOSBranch = "total",
    elements: Sequence[str] | None = None,
    plot_total: bool = True,
    plot_elements: bool = True,
    ax: Any | None = None,
    dos_bins: int = 400,
    dos_omega_ps_range: tuple[float, float] | None = None,
) -> Any:
    """
    Plot INM DOS (total + element-resolved).

    Args:
        analyzer_or_doc: INMAnalyzer (will be converted to INMDoc) or INMDoc.
        frame_index: Which sampled frame to plot. If None, plot trajectory-average DOS.
        branch: "total", "stable", or "unstable".
        elements: Optional list of element symbols to include (default: all in doc).
        plot_total: If True, plot the global DOS curve for the branch.
        plot_elements: If True, plot per-element DOS curves (weighted by participation).
        ax: Optional matplotlib axis to draw on.
        dos_bins: Used only when input is an analyzer (controls doc construction).
        dos_omega_ps_range: Used only when input is an analyzer (controls doc construction).
    """
    doc = _as_doc(analyzer_or_doc, dos_bins=dos_bins, dos_omega_ps_range=dos_omega_ps_range)
    d = _select_dos(doc, frame_index)

    edges = np.asarray(d.omega_bin_edges_ps, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if branch == "total":
        rho = np.asarray(d.rho_total, dtype=float)
        element_rho = {k: np.asarray(v, dtype=float) for k, v in d.element_rho_total.items()}
    elif branch == "stable":
        rho = np.asarray(d.rho_stable, dtype=float)
        element_rho = {k: np.asarray(v, dtype=float) for k, v in d.element_rho_stable.items()}
    elif branch == "unstable":
        rho = np.asarray(d.rho_unstable, dtype=float)
        element_rho = {k: np.asarray(v, dtype=float) for k, v in d.element_rho_unstable.items()}
    else:
        raise ValueError("branch must be one of: 'total', 'stable', 'unstable'")

    if ax is None:
        fig, ax = setup_figure()
    else:
        fig = ax.figure

    title_frame = "trajectory-average" if d.frame_index is None else f"frame {d.frame_index}"
    ax.set_title(f"INM DOS ({branch}) — {title_frame}")

    if plot_total:
        ax.plot(centers, rho, color="black", linewidth=2.5, label=f"{branch} DOS")

    if plot_elements:
        if elements is None:
            elements = sorted(element_rho.keys())
        colors = color_cycle(len(elements))
        for c, el in zip(colors, elements):
            if el not in element_rho:
                continue
            ax.plot(centers, element_rho[el], color=c, linewidth=2, alpha=0.9, label=f"{el}")

    ax.set_xlabel(r"$\omega$ (ps$^{-1}$)")
    ax.set_ylabel(r"$\rho(\omega)$")
    ax.legend()
    fig.tight_layout()
    return ax

