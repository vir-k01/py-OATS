"""Tests for ChargeStateAnalyzer."""

import numpy as np
import pytest

from py_oats.io.trajectory import TrajectoryData
from py_oats.analyzers.base import BaseAnalyzer
from py_oats.analyzers.charge_state import ChargeStateAnalyzer
from py_oats.utils.analyzers.charge_state.charge import (
    species_string_from_element_and_oxidation,
    species_array_from_elements_and_oxi_states,
    decorate_species_naive,
)
from py_oats.utils.analyzers.charge_state.oxidation import (
    assign_oxidation_states,
    composition_from_species_array,
)


@pytest.fixture
def traj_for_charge():
    """TrajectoryData with metadata, no oxidation states."""
    n_frames, n_atoms = 5, 4
    np.random.seed(42)
    positions = np.cumsum(np.random.randn(n_frames, n_atoms, 3).astype(np.float64) * 0.1, axis=0)
    species = np.array(["Li", "Mn", "O", "O"], dtype=object)
    lattice = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64)
    return TrajectoryData(
        positions=positions,
        species=species,
        lattices=lattice,
        properties={},
        metadata={"temperature": 1000.0},
    )


def test_species_string_from_element_and_oxidation():
    """Canonical formatting: Li+1, Mn+3.5, O-2 (no Structure round-trip)."""
    assert species_string_from_element_and_oxidation("Li", 1) == "Li+1"
    assert species_string_from_element_and_oxidation("Mn", 3.5) == "Mn+3.5"
    assert species_string_from_element_and_oxidation("O", -2) == "O-2"
    assert species_string_from_element_and_oxidation("Li", 0) == "Li0"


def test_composition_from_species_array():
    """Composition from species array (no Structure)."""
    species = np.array(["Li", "Mn", "O", "O"], dtype=object)
    comp = composition_from_species_array(species)
    assert comp == {"Li": 1, "Mn": 1, "O": 2}
    species2 = np.array(["Li+1", "O-2", "O-2"], dtype=object)
    comp2 = composition_from_species_array(species2)
    assert comp2 == {"Li": 1, "O": 2}


def test_assign_oxidation_states_by_magmoms_species_array():
    """assign_oxidation_states_by_magmoms accepts species_array and magmoms (no Structure)."""
    species = np.array(["Li", "Mn", "O", "O"], dtype=object)
    magmoms = [0.1, 3.5, 0.01, 0.01]  # Li low, Mn mid, O low
    ox = {"Li": [1], "Mn": [3, 4], "O": [-2]}
    bounds = {"Mn": {4: 3.08, 3: 4.2, 2: 6}}
    result = assign_oxidation_states(
        species, ox, magmoms, adaptive_boundaries=False, bounds=bounds
    )
    assert result[0] == 1 and result[2] == result[3] == -2
    assert result[1] in (3, 4)


def test_decorate_species_naive_li_mn_o():
    """Naive decoration for LiMn2O4-style: Li+1, Mn+3.5, O-2."""
    species = np.array(["Li", "Mn", "O", "O"], dtype=object)
    ox = {"Li": [1], "Mn": [3, 4], "O": [-2]}
    out = decorate_species_naive(species, ox)
    assert out[0] == "Li+1"
    assert out[1] == "Mn+3.5"
    assert out[2] == out[3] == "O-2"


def test_charge_state_analyzer_inherits_base(traj_for_charge):
    """ChargeStateAnalyzer subclasses BaseAnalyzer and sets name."""
    assert issubclass(ChargeStateAnalyzer, BaseAnalyzer)
    a = ChargeStateAnalyzer(traj_for_charge, oxidation_states={"Li": [1], "Mn": [3], "O": [-2]})
    assert a.trajectory is traj_for_charge
    assert a.name == "charge_state_analyzer"


def test_charge_state_analyzer_naive_decorate_first(traj_for_charge):
    """Analyze returns list of TrajectoryData with decorated species (Naive: one chunk)."""
    ox = {"Li": [1], "Mn": [3], "O": [-2]}
    a = ChargeStateAnalyzer(traj_for_charge, oxidation_states=ox)
    result = a.analyze(decorate_freq=0)
    assert len(result) == 1
    td = result[0]
    species_strs = [str(s) for s in td.species]
    assert any("+" in s or (s.startswith("O") and "-" in s) for s in species_strs), species_strs
    assert td.n_frames == 5 and td.n_atoms == 4
    np.testing.assert_array_equal(td.positions, traj_for_charge.positions)


def test_charge_state_analyzer_default_oxidation_states(traj_for_charge):
    """When oxidation_states is None, defaults to unique elements with [0]."""
    a = ChargeStateAnalyzer(traj_for_charge)
    result = a.analyze(decorate_freq=0)
    assert a.oxidation_states is not None
    assert "Li" in a.oxidation_states and "Mn" in a.oxidation_states and "O" in a.oxidation_states
    assert len(result) == 1


def test_charge_state_analyzer_with_magmoms_returns_list(traj_for_charge):
    """When trajectory has magmoms, analyze uses them and returns list with metadata tag."""
    n_frames, n_atoms = 5, 4
    magmoms = np.random.randn(n_frames, n_atoms).astype(np.float64) * 0.5 + 3.0  # positive
    td = TrajectoryData(
        positions=traj_for_charge.positions,
        species=traj_for_charge.species,
        lattices=traj_for_charge.lattices,
        properties={"magmoms": magmoms},
        metadata=traj_for_charge.metadata,
    )
    a = ChargeStateAnalyzer(td, oxidation_states={"Li": [1], "Mn": [3, 4], "O": [-2]})
    result = a.analyze(decorate_freq=0)
    assert len(result) == 1
    assert result[0].metadata.get("decorated_with") == "user_magmoms"
    assert result[0].metadata.get("decorate_freq") == 0


