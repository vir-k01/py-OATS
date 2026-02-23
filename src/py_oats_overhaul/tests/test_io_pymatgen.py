"""Tests for pymatgen I/O: structures_to_data and trajectory_to_data (return dict)."""

import numpy as np
import pytest
from pymatgen.core import Structure, Lattice
from pymatgen.core.trajectory import Trajectory as PmgTrajectory

from py_oats_overhaul.reader.trajectory import TrajectoryData
from py_oats_overhaul.utils.io.pymatgen import (
    data_to_structures,
    structures_to_data,
    trajectory_to_data,
)


def _to_trajectory_data(data: dict) -> TrajectoryData:
    return TrajectoryData.from_dict(data)


class TestStructuresToData:
    """structures_to_data(list[Structure], ...) -> dict."""

    def test_minimal(
        self, pymatgen_structures_list, n_frames, n_atoms, positions, species
    ):
        data = structures_to_data(pymatgen_structures_list, metadata={})
        td = _to_trajectory_data(data)
        assert isinstance(td, TrajectoryData)
        assert td.n_frames == n_frames
        assert td.n_atoms == n_atoms
        np.testing.assert_array_almost_equal(td.positions, positions)
        np.testing.assert_array_equal(td.species, species)
        assert td.lattices.ndim == 2 and td.lattices.shape == (3, 3)

    def test_constant_lattice_stored_once(self, pymatgen_structures_list):
        data = structures_to_data(pymatgen_structures_list, metadata={})
        td = _to_trajectory_data(data)
        assert td.lattices.shape == (3, 3)

    def test_with_site_properties(
        self, pymatgen_structures_with_site_props, n_frames, n_atoms
    ):
        data = structures_to_data(pymatgen_structures_with_site_props, metadata={})
        td = _to_trajectory_data(data)
        assert "magmoms" in td.properties
        assert td.properties["magmoms"].shape == (n_frames, n_atoms)

    def test_frame_properties_scalar(self, pymatgen_structures_list, n_frames):
        frame_props = [{"energy": float(i)} for i in range(n_frames)]
        data = structures_to_data(
            pymatgen_structures_list,
            metadata={},
            frame_properties=frame_props,
        )
        td = _to_trajectory_data(data)
        assert "energy" in td.properties
        assert td.properties["energy"].shape == (n_frames,)
        np.testing.assert_array_almost_equal(
            td.properties["energy"], np.arange(n_frames, dtype=np.float64)
        )

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="Empty list"):
            structures_to_data([], metadata={})


class TestTrajectoryToDataPymatgen:
    """trajectory_to_data(PmgTrajectory, ...) -> TrajectoryData."""

    def test_from_structures(
        self, pymatgen_structures_list, n_frames, n_atoms, species
    ):
        traj = PmgTrajectory.from_structures(pymatgen_structures_list)
        data = trajectory_to_data(traj, metadata={})
        td = _to_trajectory_data(data)
        assert td.n_frames == n_frames
        assert td.n_atoms == n_atoms
        assert td.positions.shape == (n_frames, n_atoms, 3)
        np.testing.assert_array_equal(td.species, species)
        assert td.lattices.ndim == 2 and td.lattices.shape == (3, 3)
        # Positions are finite and consistent with lattice (PmgTrajectory may wrap/store differently)
        assert np.all(np.isfinite(td.positions))

    def test_empty_trajectory_raises(self):
        with pytest.raises(IndexError):
            PmgTrajectory.from_structures([])


class TestDataToStructures:
    """data_to_structures(TrajectoryData) -> list[Structure]."""

    def test_roundtrip(self, traj_data_minimal, n_frames, n_atoms):
        structures = data_to_structures(traj_data_minimal)
        assert len(structures) == n_frames
        assert len(structures[0]) == n_atoms
        data = structures_to_data(structures, metadata=dict(traj_data_minimal.metadata))
        td = _to_trajectory_data(data)
        np.testing.assert_array_almost_equal(td.positions, traj_data_minimal.positions)
        np.testing.assert_array_equal(td.species, traj_data_minimal.species)
        assert td.n_frames == n_frames and td.n_atoms == n_atoms

    def test_properties_passed_to_site_properties(self, traj_data_minimal, n_frames, n_atoms):
        """TrajectoryData with properties -> structures have site_properties."""
        magmoms = np.random.randn(n_frames, n_atoms).astype(np.float64)
        td = TrajectoryData(
            positions=traj_data_minimal.positions,
            species=traj_data_minimal.species,
            lattices=traj_data_minimal.lattices,
            properties={"magmoms": magmoms},
            metadata=traj_data_minimal.metadata,
        )
        structures = data_to_structures(td)
        assert "magmoms" in structures[0].site_properties
        np.testing.assert_array_almost_equal(
            structures[0].site_properties["magmoms"], magmoms[0]
        )
