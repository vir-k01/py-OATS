"""Tests for TransportAnalyzer."""

import numpy as np
import pytest

from py_oats.reader.trajectory import TrajectoryData
from py_oats.analyzers.base import BaseAnalyzer
from py_oats.analyzers.transport import TransportAnalyzer


@pytest.fixture
def traj_data_transport():
    """TrajectoryData with metadata for transport (temperature, time_step, etc.)."""
    n_frames, n_atoms = 40, 4
    np.random.seed(123)
    positions = np.cumsum(np.random.randn(n_frames, n_atoms, 3).astype(np.float64) * 0.1, axis=0)
    species = np.array(["Li", "Mn", "O", "O"], dtype=object)
    lattice = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64)
    return TrajectoryData(
        positions=positions,
        species=species,
        lattices=lattice,
        properties={},
        metadata={"temperature": 1000.0, "time_step": 2.0, "step_skip": 1, "smoothing": "best_fit"},
    )


def test_transport_analyzer_inherits_base(traj_data_transport):
    """TransportAnalyzer subclasses BaseAnalyzer and sets trajectory/name."""
    assert issubclass(TransportAnalyzer, BaseAnalyzer)
    a = TransportAnalyzer(traj_data_transport)
    assert a.trajectory is traj_data_transport
    assert a.name == "transport_analyzer"


def test_transport_analyzer_requires_temperature():
    """TrajectoryData without metadata['temperature'] raises."""
    td = TrajectoryData(
        positions=np.random.randn(5, 2, 3).astype(np.float64),
        species=np.array(["A", "B"], dtype=object),
        lattices=np.eye(3, dtype=np.float64),
        properties={},
        metadata={},
    )
    with pytest.raises(ValueError, match="temperature"):
        TransportAnalyzer(td)


def test_transport_analyzer_runs(traj_data_transport):
    """TransportAnalyzer builds and computes L_tensor, fit_dicts (indexed by species order)."""
    a = TransportAnalyzer(traj_data_transport)
    a.analyze()
    assert a.temperature == 1000.0
    assert a.time_step == 2.0
    assert a.species == ["Li", "Mn", "O"]
    assert a.L_tensor.shape == (3, 3)
    assert a.L_tensor_self.shape == (3, 3)
    assert a.L_tensor_dis.shape == (3, 3)
    assert a.mapping == {"Li": 0, "Mn": 1, "O": 2}
    # fit_dicts indexed by (i, j) with i, j = mapping[species]
    assert "fit_err" in a.fit_dicts[0, 0]
    assert a.fit_dicts[0, 1] is a.fit_dicts[1, 0]  # same dict for (Li,Mn) and (Mn,Li)
    assert "fit_err" in a.fit_dicts_self[0, 0]
    # L_tensor_dis = L_tensor - L_tensor_self
    np.testing.assert_array_almost_equal(a.L_tensor_dis, a.L_tensor - a.L_tensor_self)
    # correlation_functions keyed by (species_i, species_j) for plotting
    assert ("Li", "Li") in a.correlation_functions
    assert set(a.correlation_functions[("Li", "Li")].keys()) == {"total", "self"}
    assert a.correlation_functions[("Li", "Li")]["total"].shape == (40,)
    assert ("Li", "Mn") in a.correlation_functions
    assert set(a.correlation_functions[("Li", "Mn")].keys()) == {"distinct"}
    assert ("O", "O") in a.correlation_functions
