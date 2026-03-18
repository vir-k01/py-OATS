"""Pytest fixtures for py_oats_overhaul tests."""

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.cell import Cell
from pymatgen.core import Structure, Lattice

from py_oats.io.trajectory import TrajectoryData


@pytest.fixture
def n_frames():
    return 3


@pytest.fixture
def n_atoms():
    return 4


@pytest.fixture
def positions(n_frames, n_atoms):
    np.random.seed(42)
    return np.random.randn(n_frames, n_atoms, 3).astype(np.float64) * 2.0


@pytest.fixture
def species(n_atoms):
    return np.array(["Li", "Mn", "O", "O"], dtype=object)[:n_atoms]


@pytest.fixture
def lattice_3x3():
    return np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64)


@pytest.fixture
def lattices_variable(n_frames, lattice_3x3):
    L = np.zeros((n_frames, 3, 3), dtype=np.float64)
    for i in range(n_frames):
        L[i] = lattice_3x3 + np.random.randn(3, 3).astype(np.float64) * 0.01
    return L


@pytest.fixture
def properties_valid(n_frames, n_atoms):
    return {
        "magmoms": np.random.randn(n_frames, n_atoms).astype(np.float64),
        "energy": np.linspace(0.0, 1.0, n_frames).astype(np.float64),
    }


@pytest.fixture
def traj_data_minimal(positions, species, lattice_3x3):
    """TrajectoryData with constant (3,3) lattice."""
    return TrajectoryData(
        positions=positions,
        species=species,
        lattices=np.array(lattice_3x3, copy=True),
        properties={},
        metadata={},
        composition=None,
    )


@pytest.fixture
def traj_data_variable_lattice(positions, species, lattices_variable):
    """TrajectoryData with (n_frames, 3, 3) lattice."""
    return TrajectoryData(
        positions=positions,
        species=species,
        lattices=lattices_variable,
        properties={},
        metadata={},
        composition=None,
    )


@pytest.fixture
def ase_atoms_list(n_frames, n_atoms, positions, species, lattice_3x3):
    """List of ASE Atoms, same cell and symbols, different positions."""
    atoms_list = []
    for i in range(n_frames):
        a = Atoms(
            symbols=species.tolist(),
            positions=positions[i],
            cell=Cell(lattice_3x3),
        )
        atoms_list.append(a)
    return atoms_list


@pytest.fixture
def ase_atoms_list_with_magmoms(ase_atoms_list, n_frames, n_atoms):
    """ASE atoms list with magmoms array."""
    magmoms = np.random.randn(n_frames, n_atoms).astype(np.float64)
    for i, a in enumerate(ase_atoms_list):
        a.set_array("magmoms", magmoms[i])
    return ase_atoms_list


@pytest.fixture
def pymatgen_structures_list(n_frames, n_atoms, positions, species, lattice_3x3):
    """List of pymatgen Structures."""
    lattice = Lattice(lattice_3x3)
    structures = []
    for i in range(n_frames):
        s = Structure(
            lattice,
            species.tolist(),
            positions[i],
            coords_are_cartesian=True,
        )
        structures.append(s)
    return structures


@pytest.fixture
def pymatgen_structures_with_site_props(pymatgen_structures_list, n_frames, n_atoms):
    """Structures with magmoms in site_properties."""
    magmoms = np.random.randn(n_frames, n_atoms).astype(np.float64)
    for i, s in enumerate(pymatgen_structures_list):
        s.add_site_property("magmoms", magmoms[i].tolist())
    return pymatgen_structures_list
