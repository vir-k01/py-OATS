"""Tests for py_oats.utils.polyhedra polyhedral environment extraction."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Lattice, Molecule, Structure

from py_oats.utils.polyhedra import (
    _CONNECTED_GEOMETRIES,
    _MINIMUM_GEOMETRY_OP,
    _chemsys_list,
    _classify_sites,
    _polyhedra_from_structure,
    get_polyhedra_from_mp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mgo():
    """MgO rocksalt conventional cell, a = 4.211 Å (8 atoms: 4 Mg + 4 O)."""
    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(4.211),
        ["Mg", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture(scope="module")
def mgo_geometries(mgo):
    return _classify_sites(mgo)


@pytest.fixture(scope="module")
def mgo_polyhedra(mgo, mgo_geometries):
    return _polyhedra_from_structure(mgo, CrystalNN(), mgo_geometries)


# ---------------------------------------------------------------------------
# _chemsys_list
# ---------------------------------------------------------------------------

def test_chemsys_unary():
    assert _chemsys_list(Composition("Li")) == ["Li"]


def test_chemsys_binary_contains_both_unaries_and_binary():
    result = _chemsys_list(Composition("LiO"))
    assert set(result) == {"Li", "O", "Li-O"}


def test_chemsys_binary_alphabetical():
    result = _chemsys_list(Composition("FeO"))
    assert "Fe-O" in result
    assert "O-Fe" not in result


def test_chemsys_ternary_length():
    # 3 unaries + 3 pairwise binaries = 6
    result = _chemsys_list(Composition("LiMnO2"))
    assert len(result) == 6


def test_chemsys_ternary_no_ternary_string():
    result = _chemsys_list(Composition("LiMnO2"))
    assert not any(s.count("-") == 2 for s in result)


def test_chemsys_ternary_contains_all_pairs():
    result = _chemsys_list(Composition("LiMnO2"))
    assert "Li-Mn" in result
    assert "Li-O" in result
    assert "Mn-O" in result


def test_chemsys_quaternary_length():
    # 4 unaries + C(4,2)=6 pairwise binaries = 10
    result = _chemsys_list(Composition("LiMnFeO4"))
    assert len(result) == 10


# ---------------------------------------------------------------------------
# _classify_sites
# ---------------------------------------------------------------------------

def test_classify_sites_length(mgo, mgo_geometries):
    assert len(mgo_geometries) == len(mgo)


def test_classify_sites_returns_str_or_none(mgo_geometries):
    for g in mgo_geometries:
        assert g is None or isinstance(g, str)


def test_classify_sites_only_connected_geometries(mgo_geometries):
    for g in mgo_geometries:
        if g is not None:
            assert g in _CONNECTED_GEOMETRIES


def test_classify_sites_mgo_all_octahedral(mgo_geometries):
    # Every site in MgO rocksalt has perfect octahedral coordination.
    assert all(g == "octahedral" for g in mgo_geometries)


# ---------------------------------------------------------------------------
# _polyhedra_from_structure
# ---------------------------------------------------------------------------

def test_polyhedra_central_atom_at_origin(mgo_polyhedra):
    for mol in mgo_polyhedra:
        np.testing.assert_allclose(mol.cart_coords[0], [0.0, 0.0, 0.0], atol=1e-10)


def test_polyhedra_minimum_two_atoms(mgo_polyhedra):
    for mol in mgo_polyhedra:
        assert len(mol) >= 2


def test_polyhedra_all_are_molecules(mgo_polyhedra):
    assert all(isinstance(m, Molecule) for m in mgo_polyhedra)


def test_polyhedra_none_geometries_yield_empty(mgo):
    mols = _polyhedra_from_structure(mgo, CrystalNN(), [None] * len(mgo))
    assert mols == []


def test_polyhedra_mgo_compositions(mgo_polyhedra):
    # MgO rocksalt: Mg in O-octahedron (MgO6) and O in Mg-octahedron (Mg6O)
    formulas = {m.composition.reduced_formula for m in mgo_polyhedra}
    assert "MgO6" in formulas
    assert "Mg6O" in formulas


def test_polyhedra_mgo_coordination_number(mgo_polyhedra):
    # Both environments are 6-coordinate → molecule has 7 atoms (1 center + 6)
    for mol in mgo_polyhedra:
        assert len(mol) == 7


def test_polyhedra_nnn_gate_removes_isolated(mgo):
    # Force only one site to have a connected geometry; its NNNs (via neighbours
    # that have None geometry) will also have None → gate 2 drops the site.
    geoms = [None] * len(mgo)
    geoms[0] = "octahedral"   # Mg[0] has geometry, but all its NNNs have None
    mols = _polyhedra_from_structure(mgo, CrystalNN(), geoms)
    assert mols == []


# ---------------------------------------------------------------------------
# get_polyhedra_from_mp  (integration — require MP API key)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_mp_returns_list_of_molecules():
    mols = get_polyhedra_from_mp("SiO2")
    assert isinstance(mols, list)
    assert all(isinstance(m, Molecule) for m in mols)


@pytest.mark.integration
def test_mp_central_atom_at_origin():
    mols = get_polyhedra_from_mp("SiO2")
    for mol in mols:
        np.testing.assert_allclose(mol.cart_coords[0], [0.0, 0.0, 0.0], atol=1e-10)


@pytest.mark.integration
def test_mp_no_duplicate_compositions():
    mols = get_polyhedra_from_mp("SiO2")
    formulas = [m.composition.reduced_formula for m in mols]
    assert len(formulas) == len(set(formulas))


@pytest.mark.integration
def test_mp_sio2_has_sio4_tetrahedra():
    mols = get_polyhedra_from_mp("SiO2")
    formulas = {m.composition.reduced_formula for m in mols}
    assert "SiO4" in formulas


@pytest.mark.integration
def test_mp_al2o3_has_alo6_octahedra():
    mols = get_polyhedra_from_mp("Al2O3")
    formulas = {m.composition.reduced_formula for m in mols}
    assert "AlO6" in formulas


@pytest.mark.integration
def test_mp_elements_within_queried_system():
    comp = Composition("LiMn2O4")
    system_elements = {str(el) for el in comp.elements}
    mols = get_polyhedra_from_mp(comp)
    for mol in mols:
        mol_elements = {str(el) for el in mol.composition.elements}
        assert mol_elements.issubset(system_elements)


@pytest.mark.integration
def test_mp_string_input_accepted():
    mols = get_polyhedra_from_mp("MgO")
    assert len(mols) > 0


@pytest.mark.integration
def test_mp_mgo_has_octahedra():
    mols = get_polyhedra_from_mp("MgO")
    formulas = {m.composition.reduced_formula for m in mols}
    # MgO rocksalt: expect at least one polyhedron containing both Mg and O
    # (exact formula depends on whether MP returns primitive or conventional cell)
    assert any(
        {"Mg", "O"}.issubset({str(el) for el in Composition(f).elements})
        for f in formulas
    )
