"""Tests for CoordinationAnalyzer (RDF and cage time)."""

from __future__ import annotations

import numpy as np
import pytest

from py_oats.io.trajectory import TrajectoryData
from py_oats.analyzers.base import BaseAnalyzer
from py_oats.analyzers.coordination import CoordinationAnalyzer, rdf_pair_key, TOTAL_RDF_KEY
from py_oats.schemas.coordination import RDFResult, CageResult, CoordinationAnalysisDoc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sc_traj(n_frames=20, seed=0) -> TrajectoryData:
    """
    Simple cubic supercell: 4 Li + 8 O at random positions inside a 6 Å cube.
    Lattice is constant.  No unwrapping needed (positions stay near 0).
    """
    rng = np.random.default_rng(seed)
    n_atoms = 12
    lattice = np.eye(3, dtype=np.float64) * 6.0
    base = np.array([
        [0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.1, 0.6, 0.1], [0.6, 0.6, 0.1],  # Li (frac)
        [0.35, 0.35, 0.35], [0.85, 0.35, 0.35], [0.35, 0.85, 0.35], [0.85, 0.85, 0.35],
        [0.35, 0.35, 0.85], [0.85, 0.35, 0.85], [0.35, 0.85, 0.85], [0.85, 0.85, 0.85],
    ]) * 6.0  # Cartesian
    positions = np.stack(
        [base + rng.normal(0, 0.05, (n_atoms, 3)) for _ in range(n_frames)]
    )
    species = np.array(["Li"] * 4 + ["O"] * 8, dtype=object)
    return TrajectoryData(
        positions=positions,
        species=species,
        lattices=lattice,
        properties={},
        metadata={"temperature": 1000.0, "time_step": 2.0, "step_skip": 1},
    )


@pytest.fixture
def traj_lmo():
    return _make_sc_traj(n_frames=20)


@pytest.fixture
def traj_lmo_long():
    return _make_sc_traj(n_frames=60, seed=42)


@pytest.fixture
def an_analyzed(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51, sigma=0.1)
    an.analyze()
    return an


# ---------------------------------------------------------------------------
# rdf_pair_key helper
# ---------------------------------------------------------------------------

def test_rdf_pair_key_sorted():
    assert rdf_pair_key("O", "Li") == "Li-O"
    assert rdf_pair_key("Li", "O") == "Li-O"


def test_rdf_pair_key_self():
    assert rdf_pair_key("Li", "Li") == "Li-Li"


# ---------------------------------------------------------------------------
# Construction and inheritance
# ---------------------------------------------------------------------------

def test_inherits_base(traj_lmo):
    assert issubclass(CoordinationAnalyzer, BaseAnalyzer)
    an = CoordinationAnalyzer(traj_lmo)
    assert an.trajectory is traj_lmo
    assert an.name == "coordination_analyzer"


def test_default_grid(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    assert an.rmax == 8.0
    assert an.ngrid == 201
    assert len(an.r) == 201
    assert an.r[0] == pytest.approx(0.0)
    assert an.r[-1] == pytest.approx(8.0)


def test_custom_params(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51, sigma=0.05)
    assert an.rmax == 5.0
    assert an.ngrid == 51
    assert an.r[-1] == pytest.approx(5.0)


def test_invalid_ngrid(traj_lmo):
    with pytest.raises(ValueError, match="ngrid"):
        CoordinationAnalyzer(traj_lmo, ngrid=1)


def test_invalid_rmax(traj_lmo):
    with pytest.raises(ValueError, match="rmax"):
        CoordinationAnalyzer(traj_lmo, rmax=0.0)


def test_get_rdf_before_analyze_raises(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    with pytest.raises(RuntimeError, match="analyze"):
        an.get_rdf("Li", "O")


def test_rdfs_empty_before_analyze(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    assert an.rdfs == {}


# ---------------------------------------------------------------------------
# analyze() — neighbor lists
# ---------------------------------------------------------------------------

def test_analyze_populates_neighbor_lists(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    an.analyze()
    assert len(an._distances) == len(an.frame_indices)
    assert all(isinstance(d, np.ndarray) for d in an._distances)


def test_analyze_frame_sample(traj_lmo_long):
    an = CoordinationAnalyzer(traj_lmo_long, frame_sample=10)
    assert len(an.frame_indices) == 10
    an.analyze()
    assert len(an._distances) == 10


# ---------------------------------------------------------------------------
# analyze() — pre-computed rdfs dict
# ---------------------------------------------------------------------------

def test_analyze_rdfs_has_all_pairs(an_analyzed):
    """Every unordered species pair should have an entry."""
    species = an_analyzed.unique_species  # ["Li", "O"]
    expected_keys = {rdf_pair_key(s1, s2) for s1 in species for s2 in species}
    for key in expected_keys:
        assert key in an_analyzed.rdfs, f"Missing key: {key}"


def test_analyze_rdfs_has_total(an_analyzed):
    assert TOTAL_RDF_KEY in an_analyzed.rdfs


def test_analyze_rdfs_values_are_rdf_results(an_analyzed):
    for result in an_analyzed.rdfs.values():
        assert isinstance(result, RDFResult)


def test_analyze_rdfs_pair_count(an_analyzed):
    """2 species → 3 unordered pairs (Li-Li, Li-O, O-O) + total = 4 entries."""
    assert len(an_analyzed.rdfs) == 4


def test_analyze_rdfs_pair_key_consistent(an_analyzed):
    """rdfs["Li-O"] and get_rdf("Li","O") should return identical rdf arrays."""
    cached = an_analyzed.rdfs["Li-O"]
    fresh = an_analyzed.get_rdf("Li", "O")
    np.testing.assert_array_equal(cached.rdf, fresh.rdf)


def test_analyze_rdfs_total_is_all_species(an_analyzed):
    """total RDF ref_species and species should cover all unique species."""
    total = an_analyzed.rdfs[TOTAL_RDF_KEY]
    assert set(total.ref_species) == set(an_analyzed.unique_species)
    assert set(total.species) == set(an_analyzed.unique_species)


def test_analyze_rdfs_rerun_updates(traj_lmo):
    """Calling analyze() twice should overwrite rdfs cleanly."""
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51)
    an.analyze()
    first_rdf = an.rdfs["Li-O"].rdf.copy()
    an.analyze()
    np.testing.assert_array_equal(an.rdfs["Li-O"].rdf, first_rdf)


# ---------------------------------------------------------------------------
# RDF correctness
# ---------------------------------------------------------------------------

def test_get_rdf_returns_rdf_result(an_analyzed):
    assert isinstance(an_analyzed.get_rdf("Li", "O"), RDFResult)


def test_rdf_grid_shape(an_analyzed):
    result = an_analyzed.get_rdf("Li", "O")
    assert result.r.shape == (51,)
    assert result.rdf.shape == (51,)
    assert result.coordination_number.shape == (51,)


def test_rdf_nonnegative(an_analyzed):
    result = an_analyzed.get_rdf("Li", "O")
    assert np.all(result.rdf >= -1e-10)


def test_rdf_per_frame_shape(an_analyzed, traj_lmo):
    result = an_analyzed.get_rdf("Li", "O")
    assert result.rdf_per_frame.shape == (traj_lmo.n_frames, 51)


def test_rdf_zero_at_origin(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=101, sigma=0.0)
    an.analyze()
    result = an.get_rdf("Li", "O")
    assert result.rdf[0] == pytest.approx(0.0, abs=1e-10)


def test_rdf_self_pair(an_analyzed):
    result = an_analyzed.get_rdf("Li", "Li")
    assert result.rdf.shape == (51,)
    assert np.all(result.rdf >= -1e-10)


def test_rdf_string_or_list_species(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51, sigma=0.0)
    an.analyze()
    r1 = an.get_rdf("Li", "O")
    r2 = an.get_rdf(["Li"], ["O"])
    np.testing.assert_array_equal(r1.rdf, r2.rdf)


def test_rdf_peak_positions_sensible(an_analyzed):
    result = an_analyzed.get_rdf("Li", "O")
    for p in result.peak_r:
        assert 0.0 < p <= 5.0


def test_coordination_number_monotone(an_analyzed):
    _, cn = an_analyzed.get_coordination_number("Li", "O")
    assert np.all(np.diff(cn) >= -1e-10)


def test_coordination_number_positive_at_rmax(an_analyzed):
    _, cn = an_analyzed.get_coordination_number("Li", "O")
    assert cn[-1] > 0.0


# ---------------------------------------------------------------------------
# Cage correlation / cage time
# ---------------------------------------------------------------------------

def test_cage_correlation_returns_cage_result(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5)
    assert isinstance(result, CageResult)


def test_cage_result_shapes(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5)
    assert result.lag_times.shape == (traj_lmo.n_frames // 2 + 1,)
    assert result.cage_correlation.shape == (traj_lmo.n_frames // 2 + 1,)


def test_cage_correlation_starts_at_one(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5)
    assert result.cage_correlation[0] == pytest.approx(1.0, abs=1e-6)


def test_cage_correlation_bounded(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5)
    assert np.all(result.cage_correlation >= -1e-10)
    assert np.all(result.cage_correlation <= 1.0 + 1e-10)


def test_cage_time_positive(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5)
    assert result.cage_time > 0.0


def test_cage_time_1e_type(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5)
    assert isinstance(result.cage_time_1e, float)


def test_cage_correlation_max_lag(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5, max_lag=5)
    assert len(result.lag_times) == 6
    assert len(result.cage_correlation) == 6


def test_cage_lag_times_use_metadata(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5, max_lag=3)
    np.testing.assert_allclose(result.lag_times, [0.0, 2.0, 4.0, 6.0])


def test_cage_invalid_mobile_species(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    with pytest.raises(ValueError, match="mobile_species"):
        an.get_cage_correlation("Zr", "O", cutoff=3.0)


def test_cage_invalid_cage_species(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    with pytest.raises(ValueError, match="cage_species"):
        an.get_cage_correlation("Li", "Zr", cutoff=3.0)


# ---------------------------------------------------------------------------
# Schema import paths
# ---------------------------------------------------------------------------

def test_schema_importable_from_schemas_module():
    from py_oats.schemas.coordination import RDFResult as R, CageResult as C
    assert R is RDFResult
    assert C is CageResult


def test_rdf_result_species_labels(an_analyzed):
    result = an_analyzed.get_rdf("Li", "O")
    assert result.ref_species == ["Li"]
    assert result.species == ["O"]


def test_cage_result_labels(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo)
    result = an.get_cage_correlation("Li", "O", cutoff=3.5)
    assert result.mobile_species == ["Li"]
    assert result.cage_species == ["O"]
    assert result.cutoff == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# CoordinationAnalysisDoc
# ---------------------------------------------------------------------------

def test_doc_from_analyzer(an_analyzed):
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    assert isinstance(doc, CoordinationAnalysisDoc)


def test_doc_before_analyze_raises(traj_lmo):
    an = CoordinationAnalyzer(traj_lmo, rmax=5.0, ngrid=51)
    with pytest.raises(RuntimeError, match="analyze"):
        CoordinationAnalysisDoc.from_analyzer(an)


def test_doc_fields(an_analyzed, traj_lmo):
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    assert doc.unique_species == an_analyzed.unique_species
    assert doc.rmax == pytest.approx(5.0)
    assert doc.ngrid == 51
    assert doc.sigma == pytest.approx(0.1)
    assert doc.n_frames == traj_lmo.n_frames


def test_doc_rdfs_keys_match_analyzer(an_analyzed):
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    assert set(doc.rdfs.keys()) == set(an_analyzed.rdfs.keys())


def test_doc_rdfs_are_copies(an_analyzed):
    """rdfs dict is a shallow copy — modifying doc.rdfs does not affect analyzer."""
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    doc.rdfs["extra"] = doc.rdfs["Li-O"]
    assert "extra" not in an_analyzed.rdfs


def test_doc_get_rdf_order_insensitive(an_analyzed):
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    r1 = doc.get_rdf("Li", "O")
    r2 = doc.get_rdf("O", "Li")
    np.testing.assert_array_equal(r1.rdf, r2.rdf)


def test_doc_get_rdf_missing_pair_raises(an_analyzed):
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    with pytest.raises(KeyError, match="Zr"):
        doc.get_rdf("Li", "Zr")


def test_doc_get_rdf_returns_correct_result(an_analyzed):
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    result = doc.get_rdf("Li", "O")
    assert isinstance(result, RDFResult)
    np.testing.assert_array_equal(result.rdf, an_analyzed.rdfs["Li-O"].rdf)


def test_doc_cage_results_default_empty(an_analyzed):
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    assert doc.cage_results == {}


def test_doc_cage_results_stored(an_analyzed):
    cage = an_analyzed.get_cage_correlation("Li", "O", cutoff=3.5)
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed, cage_results={"Li-O": cage})
    assert "Li-O" in doc.cage_results
    assert doc.cage_results["Li-O"] is cage


def test_doc_total_rdf_present(an_analyzed):
    doc = CoordinationAnalysisDoc.from_analyzer(an_analyzed)
    assert TOTAL_RDF_KEY in doc.rdfs
    assert isinstance(doc.rdfs[TOTAL_RDF_KEY], RDFResult)
