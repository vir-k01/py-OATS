"""Tests for transport plotting utilities."""

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np

from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.plotter.transport import (
    plot_all_correlations,
    plot_correlation_pair,
    plot_correlation_pairwise,
)
from py_oats.reader.trajectory import TrajectoryData


def _make_analyzer() -> TransportAnalyzer:
    n_frames, n_atoms = 20, 3
    np.random.seed(0)
    positions = np.cumsum(
        np.random.randn(n_frames, n_atoms, 3).astype(np.float64) * 0.1, axis=0
    )
    species = np.array(["Li", "Mn", "O"], dtype=object)
    lattice = np.eye(3, dtype=np.float64) * 4.0
    td = TrajectoryData(
        positions=positions,
        species=species,
        lattices=lattice,
        properties={},
        metadata={"temperature": 1000.0, "time_step": 2.0, "step_skip": 1},
    )
    a = TransportAnalyzer(td)
    a.analyze()
    return a


def test_plot_correlation_pair_smoke():
    """plot_correlation_pair returns an Axes and adds at least one line."""
    a = _make_analyzer()
    ax = plot_correlation_pair(a, "Li", "Li")
    assert len(ax.lines) >= 1
    plt.close(ax.figure)


def test_plot_correlation_pairwise_smoke():
    """plot_correlation_pairwise runs without error and returns an Axes."""
    a = _make_analyzer()
    ax = plot_correlation_pairwise(a)
    assert ax is not None
    assert ax.figure is not None
    plt.close(ax.figure)


def test_plot_all_correlations_smoke():
    """plot_all_correlations returns a Figure with multiple Axes."""
    a = _make_analyzer()
    fig = plot_all_correlations(a)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)

