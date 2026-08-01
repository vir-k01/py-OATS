"""Tests for py_oats.structure_generator.generator."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Composition, Molecule, Structure

from py_oats.structure_generator.generator import (
    get_amorphous_structure,
    get_average_volume_from_mp,
    get_random_packed_structure,
)


# ---------------------------------------------------------------------------
# get_average_volume_from_mp (cached — no API calls)
# ---------------------------------------------------------------------------

def test_avg_vol_mp_cached_returns_float():
    vol = get_average_volume_from_mp(Composition("SiO2"), use_cached=True)
    assert isinstance(vol, float)
    assert vol > 0


def test_avg_vol_mp_cached_reasonable_range():
    vol = get_average_volume_from_mp(Composition("SiO2"), use_cached=True)
    # SiO2 vol/atom is ~8-15 Å³ in crystalline phases
    assert 5.0 < vol < 25.0


def test_avg_vol_mp_cached_mgo():
    vol = get_average_volume_from_mp(Composition("MgO"), use_cached=True)
    assert 5.0 < vol < 25.0


# ---------------------------------------------------------------------------
# get_random_packed_structure — atoms-only (no polyhedra)
# ---------------------------------------------------------------------------

def test_rps_atoms_only_returns_structure():
    struct = get_random_packed_structure("SiO2", target_atoms=30, vol_per_atom_source="mp")
    assert isinstance(struct, Structure)


def test_rps_atoms_only_composition():
    struct = get_random_packed_structure("SiO2", target_atoms=30, vol_per_atom_source="mp")
    comp = struct.composition
    ratio = comp["Si"] / comp["O"]
    assert ratio == pytest.approx(0.5, abs=0.01)


def test_rps_atoms_only_target_atoms():
    struct = get_random_packed_structure("SiO2", target_atoms=30, vol_per_atom_source="mp")
    assert struct.num_sites >= 30


def test_rps_atoms_only_cubic_cell():
    struct = get_random_packed_structure("SiO2", target_atoms=30, vol_per_atom_source="mp")
    assert struct.lattice.a == pytest.approx(struct.lattice.b, abs=1e-6)
    assert struct.lattice.a == pytest.approx(struct.lattice.c, abs=1e-6)
    assert struct.lattice.alpha == pytest.approx(90.0, abs=1e-6)


def test_rps_atoms_only_vol_multiply():
    s1 = get_random_packed_structure("SiO2", target_atoms=30, vol_multiply=1.0, vol_per_atom_source="mp")
    s2 = get_random_packed_structure("SiO2", target_atoms=30, vol_multiply=1.5, vol_per_atom_source="mp")
    assert s2.volume / s1.volume == pytest.approx(1.5, rel=0.01)


def test_rps_atoms_only_seed_reproducible():
    s1 = get_random_packed_structure("SiO2", target_atoms=30, packmol_seed=42, vol_per_atom_source="mp")
    s2 = get_random_packed_structure("SiO2", target_atoms=30, packmol_seed=42, vol_per_atom_source="mp")
    np.testing.assert_allclose(s1.cart_coords, s2.cart_coords, atol=1e-6)


def test_rps_string_composition():
    struct = get_random_packed_structure("MgO", target_atoms=30, vol_per_atom_source="mp")
    assert isinstance(struct, Structure)
    assert {"Mg", "O"} == {str(s) for s in struct.species}


# ---------------------------------------------------------------------------
# get_random_packed_structure — with polyhedra
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sio4_tetrahedron():
    """A SiO4 tetrahedron: Si at center, 4 O at tetrahedral vertices ~1.6 Å."""
    d = 1.6
    coords = [
        [0.0, 0.0, 0.0],
        [d, d, d],
        [d, -d, -d],
        [-d, d, -d],
        [-d, -d, d],
    ]
    return Molecule(["Si", "O", "O", "O", "O"], coords)


def test_rps_with_polyhedra(sio4_tetrahedron):
    struct = get_random_packed_structure(
        "SiO2",
        polyhedras=[sio4_tetrahedron],
        target_atoms=30,
        vol_per_atom_source="mp",
    )
    assert isinstance(struct, Structure)
    comp = struct.composition
    assert comp["Si"] / comp["O"] == pytest.approx(0.5, abs=0.01)


def test_rps_with_polyhedra_has_both_species(sio4_tetrahedron):
    struct = get_random_packed_structure(
        "SiO2",
        polyhedras=[sio4_tetrahedron],
        target_atoms=30,
        vol_per_atom_source="mp",
    )
    species = {str(s) for s in struct.species}
    assert species == {"Si", "O"}


# ---------------------------------------------------------------------------
# get_amorphous_structure (integration — requires MP API + packmol)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_amorphous_sio2():
    struct = get_amorphous_structure("SiO2", temperature=1000.0)
    assert isinstance(struct, Structure)
    comp = struct.composition
    assert comp["Si"] / comp["O"] == pytest.approx(0.5, abs=0.01)
    # Amorphous SiO2 density is ~2.2 g/cm³; predicted may differ but should
    # be in a physically sensible range
    assert 1.5 < struct.density < 5.0


@pytest.mark.integration
def test_amorphous_al2o3():
    struct = get_amorphous_structure("Al2O3", temperature=1500.0)
    assert isinstance(struct, Structure)
    comp = struct.composition
    assert comp["Al"] / comp["O"] == pytest.approx(2 / 3, abs=0.01)
    assert 1.5 < struct.density < 6.0


@pytest.mark.integration
def test_amorphous_species_only_in_composition():
    struct = get_amorphous_structure("SiO2", temperature=1000.0)
    species = {str(s) for s in struct.species}
    assert species == {"Si", "O"}


@pytest.mark.integration
def test_amorphous_cubic_cell():
    struct = get_amorphous_structure("SiO2", temperature=1000.0)
    assert struct.lattice.a == pytest.approx(struct.lattice.b, abs=1e-6)
    assert struct.lattice.a == pytest.approx(struct.lattice.c, abs=1e-6)


@pytest.mark.integration
def test_amorphous_num_sites_near_default_target():
    struct = get_amorphous_structure("SiO2", temperature=1000.0)
    # Default target_atoms=100 → integer multiple of formula units >= 100
    assert struct.num_sites >= 100
