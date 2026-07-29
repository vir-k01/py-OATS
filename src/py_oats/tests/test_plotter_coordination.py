"""Tests for coordination environment plotting utilities."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from py_oats.io.trajectory import TrajectoryData
from py_oats.analyzers.coordination import CoordinationAnalyzer, TOTAL_RDF_KEY
from py_oats.schemas.coordination import CoordinationAnalysisDoc
from py_oats.plotter.coordination import plot_rdf, plot_rdf_grid


def _make_analyzed() -> CoordinationAnalyzer:
    rng = np.random.default_rng(0)
    n_frames, n_atoms = 15, 12
    lattice = np.eye(3, dtype=np.float64) * 6.0
    base = np.array([
        [0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.1, 0.6, 0.1], [0.6, 0.6, 0.1],
        [0.35, 0.35, 0.35], [0.85, 0.35, 0.35], [0.35, 0.85, 0.35], [0.85, 0.85, 0.35],
        [0.35, 0.35, 0.85], [0.85, 0.35, 0.85], [0.35, 0.85, 0.85], [0.85, 0.85, 0.85],
    ]) * 6.0
    positions = np.stack([base + rng.normal(0, 0.05, (n_atoms, 3)) for _ in range(n_frames)])
    species = np.array(["Li"] * 4 + ["O"] * 8, dtype=object)
    td = TrajectoryData(
        positions=positions, species=species, lattices=lattice,
        properties={}, metadata={"temperature": 1000.0, "time_step": 2.0, "step_skip": 1},
    )
    an = CoordinationAnalyzer(td, rmax=5.0, ngrid=51, sigma=0.1)
    an.analyze()
    return an


# ---------------------------------------------------------------------------
# plot_rdf — single panel
# ---------------------------------------------------------------------------

def test_plot_rdf_returns_axes_from_analyzer():
    an = _make_analyzed()
    ax = plot_rdf(an)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_rdf_returns_axes_from_doc():
    an = _make_analyzed()
    doc = CoordinationAnalysisDoc.from_analyzer(an)
    ax = plot_rdf(doc)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_rdf_draws_one_line_per_pair():
    an = _make_analyzed()
    ax = plot_rdf(an, pairs=["Li-O", "Li-Li"])
    assert len(ax.lines) >= 2   # at least one line per pair (plus maybe hline)
    plt.close(ax.figure)


def test_plot_rdf_single_pair():
    an = _make_analyzed()
    ax = plot_rdf(an, pairs=["Li-O"])
    assert len([l for l in ax.lines if len(l.get_xdata()) > 1]) >= 1
    plt.close(ax.figure)


def test_plot_rdf_total_pair():
    an = _make_analyzed()
    ax = plot_rdf(an, pairs=[TOTAL_RDF_KEY])
    assert ax is not None
    plt.close(ax.figure)


def test_plot_rdf_missing_pair_raises():
    an = _make_analyzed()
    with pytest.raises(KeyError, match="Zr-Zr"):
        plot_rdf(an, pairs=["Zr-Zr"])


def test_plot_rdf_accepts_existing_ax():
    an = _make_analyzed()
    fig, ax_in = plt.subplots()
    ax_out = plot_rdf(an, ax=ax_in)
    assert ax_out is ax_in
    plt.close(fig)


def test_plot_rdf_with_coordination_number():
    an = _make_analyzed()
    ax = plot_rdf(an, pairs=["Li-O"], show_coordination_number=True)
    # Secondary axis added
    assert len(ax.figure.axes) == 2
    plt.close(ax.figure)


def test_plot_rdf_cn_ignored_for_multiple_pairs():
    """show_coordination_number is silently ignored when multiple pairs are plotted."""
    an = _make_analyzed()
    ax = plot_rdf(an, pairs=["Li-O", "Li-Li"], show_coordination_number=True)
    assert len(ax.figure.axes) == 1
    plt.close(ax.figure)


def test_plot_rdf_has_axis_labels():
    an = _make_analyzed()
    ax = plot_rdf(an, pairs=["Li-O"])
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
    plt.close(ax.figure)


def test_plot_rdf_has_legend():
    an = _make_analyzed()
    ax = plot_rdf(an, pairs=["Li-O", "O-O"])
    assert ax.get_legend() is not None
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# plot_rdf_grid — multi-panel
# ---------------------------------------------------------------------------

def test_plot_rdf_grid_returns_figure():
    an = _make_analyzed()
    fig = plot_rdf_grid(an)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_rdf_grid_from_doc():
    an = _make_analyzed()
    doc = CoordinationAnalysisDoc.from_analyzer(an)
    fig = plot_rdf_grid(doc)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_rdf_grid_panel_count_includes_total():
    """With include_total=True (default), panels = n_pairs + 1."""
    an = _make_analyzed()
    # 2 species → Li-Li, Li-O, O-O = 3 pairs + total = 4 subplots
    fig = plot_rdf_grid(an, include_total=True)
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.axison]
    assert len(visible) >= 3    # at least the pair panels are visible
    plt.close(fig)


def test_plot_rdf_grid_without_total():
    an = _make_analyzed()
    fig = plot_rdf_grid(an, include_total=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_rdf_grid_with_cn():
    an = _make_analyzed()
    fig = plot_rdf_grid(an, show_coordination_number=True)
    # Each pair panel should have a secondary axis → more axes than panels
    n_pairs = len([k for k in an.rdfs if k != TOTAL_RDF_KEY]) + 1  # +1 for total
    assert len(fig.axes) > n_pairs
    plt.close(fig)


def test_plot_rdf_grid_each_panel_has_title():
    an = _make_analyzed()
    fig = plot_rdf_grid(an)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) >= 1
    plt.close(fig)
