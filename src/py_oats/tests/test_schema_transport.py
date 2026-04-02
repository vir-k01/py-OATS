"""Tests for TransportDoc schema."""

import json
from pathlib import Path

import numpy as np

from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.io.trajectory import TrajectoryData
from py_oats.schemas.transport import TransportDoc


def _make_simple_analyzer() -> TransportAnalyzer:
    """Build a small TransportAnalyzer with minimal but valid data."""
    n_frames, n_atoms = 10, 3
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


def test_transport_doc_from_analyzer_basic():
    """TransportDoc.from_analyzer populates core fields from TransportAnalyzer."""
    a = _make_simple_analyzer()
    doc = TransportDoc.from_analyzer(a)

    assert doc.temperature == a.temperature
    assert doc.species == list(a.species)
    assert doc.mapping == a.mapping
    assert doc.L_tensor.shape == a.L_tensor.shape
    assert doc.L_tensor_self.shape == a.L_tensor_self.shape
    assert doc.L_tensor_dis.shape == a.L_tensor_dis.shape
    assert doc.volume == a.volume
    assert doc.times == list(a.times)
    assert doc.time_step == a.time_step
    assert doc.step_skip == a.step_skip
    assert doc.num_atoms == a.trajectory.n_atoms


def test_transport_doc_get_transport_coefficient():
    """get_transport_coefficient returns the expected tensor elements."""
    a = _make_simple_analyzer()
    doc = TransportDoc.from_analyzer(a)

    # diagonal (self) and off-diagonal access by species labels
    li_idx = a.mapping["Li"]
    mn_idx = a.mapping["Mn"]

    li_self = doc.get_transport_coefficient("Li", self_transport=True)
    assert np.isclose(li_self, doc.L_tensor_self[li_idx, li_idx])

    li_mn = doc.get_transport_coefficient(["Li", "Mn"], self_transport=False)
    assert np.isclose(li_mn, doc.L_tensor[li_idx, mn_idx])


def test_transport_doc_roundtrip_dict():
    """as_dict and from_dict provide a stable round-trip."""
    a = _make_simple_analyzer()
    doc = TransportDoc.from_analyzer(a)

    d = doc.as_dict()
    doc2 = TransportDoc.from_dict(d)

    assert doc2.temperature == doc.temperature
    assert doc2.species == doc.species
    assert doc2.mapping == doc.mapping
    assert doc2.L_tensor.shape == doc.L_tensor.shape


def test_transport_doc_from_monty_numpy_dict():
    """Monty-encoded numpy arrays (dumpfn) deserialize to ndarrays."""
    d = {
        "species": ["A"],
        "L_tensor": {
            "@module": "numpy",
            "@class": "array",
            "dtype": "float64",
            "data": [[1.0, 2.0], [3.0, 4.0]],
        },
        "L_tensor_self": {
            "@module": "numpy",
            "@class": "array",
            "dtype": "float64",
            "data": [[1.0, 0.0], [0.0, 1.0]],
        },
        "L_tensor_dis": {
            "@module": "numpy",
            "@class": "array",
            "dtype": "float64",
            "data": [[0.0, 2.0], [3.0, 3.0]],
        },
        "mapping": {"A": 0},
        "temperature": 300.0,
    }
    doc = TransportDoc.from_dict(d)
    assert isinstance(doc.L_tensor, np.ndarray)
    assert doc.L_tensor.shape == (2, 2)
    np.testing.assert_allclose(doc.L_tensor, [[1.0, 2.0], [3.0, 4.0]])


def test_transport_doc_example_json_file():
    """examples/transport_doc.json (Monty-style) loads if present."""
    path = Path(__file__).resolve().parents[3] / "examples" / "transport_doc.json"
    if not path.is_file():
        return
    d = json.loads(path.read_text())
    doc = TransportDoc.from_dict(d)
    assert doc.L_tensor.shape == (3, 3)
    assert doc.correlation_functions is not None

