"""Tests for ASE I/O: atoms_to_data and trajectory_to_data (return dict; TrajectoryData built in reader)."""

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.cell import Cell

from py_oats.io.trajectory import TrajectoryData
from py_oats.utils.io.ase import atoms_to_data, trajectory_to_data


def _to_trajectory_data(data: dict) -> TrajectoryData:
    return TrajectoryData.from_dict(data)


class TestAtomsToData:
    """atoms_to_data(list[Atoms]) -> dict; build TrajectoryData in reader."""

    def test_minimal(self, ase_atoms_list, n_frames, n_atoms, positions, species):
        data = atoms_to_data(ase_atoms_list, metadata={})
        td = _to_trajectory_data(data)
        assert isinstance(td, TrajectoryData)
        assert td.n_frames == n_frames
        assert td.n_atoms == n_atoms
        np.testing.assert_array_almost_equal(td.positions, positions)
        np.testing.assert_array_equal(td.species, species)
        assert td.lattices.ndim == 2 and td.lattices.shape == (3, 3)

    def test_constant_lattice_stored_once(self, ase_atoms_list):
        data = atoms_to_data(ase_atoms_list, metadata={})
        td = _to_trajectory_data(data)
        assert td.lattices.shape == (3, 3)

    def test_metadata_passthrough(self, ase_atoms_list):
        meta = {"time_step": 1.0, "step_skip": 2}
        data = atoms_to_data(ase_atoms_list, metadata=meta)
        td = _to_trajectory_data(data)
        assert td.metadata["time_step"] == 1.0
        assert td.metadata["step_skip"] == 2

    def test_with_magmoms(
        self, ase_atoms_list_with_magmoms, n_frames, n_atoms
    ):
        data = atoms_to_data(ase_atoms_list_with_magmoms, metadata={})
        td = _to_trajectory_data(data)
        assert "magmoms" in td.properties
        assert td.properties["magmoms"].shape == (n_frames, n_atoms)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="Empty list"):
            atoms_to_data([], metadata={})


class TestTrajectoryToData:
    """trajectory_to_data(list[Atoms] | ASE Trajectory) -> TrajectoryData."""

    def test_list_of_atoms_delegates(self, ase_atoms_list, positions, species):
        data = trajectory_to_data(ase_atoms_list, metadata={})
        td = _to_trajectory_data(data)
        assert td.n_frames == len(ase_atoms_list)
        np.testing.assert_array_almost_equal(td.positions, positions)
        np.testing.assert_array_equal(td.species, species)

    def test_ase_trajectory_delegates(self, ase_atoms_list):
        class MockTrajectory:
            def __len__(self):
                return len(ase_atoms_list)

            def __getitem__(self, i):
                return ase_atoms_list[i]

        data = trajectory_to_data(MockTrajectory(), metadata={})
        td = _to_trajectory_data(data)
        assert td.n_frames == len(ase_atoms_list)
        assert td.n_atoms == len(ase_atoms_list[0])
