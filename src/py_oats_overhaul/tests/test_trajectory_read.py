"""Tests for TrajectoryData.read() dispatch."""

import pytest
from ase.atoms import Atoms
from ase.cell import Cell
from pymatgen.core import Structure, Lattice

from py_oats_overhaul.reader.trajectory import TrajectoryData


class TestTrajectoryDataRead:
    """TrajectoryData.read(path_or_object, ...) dispatches to correct backend."""

    def test_read_list_ase_atoms(self, ase_atoms_list, n_frames, n_atoms):
        td = TrajectoryData.read(ase_atoms_list, metadata={})
        assert td.n_frames == n_frames
        assert td.n_atoms == n_atoms
        assert "time_step" in td.metadata
        assert "step_skip" in td.metadata

    def test_read_list_structures(
        self, pymatgen_structures_list, n_frames, n_atoms
    ):
        td = TrajectoryData.read(pymatgen_structures_list, metadata={})
        assert td.n_frames == n_frames
        assert td.n_atoms == n_atoms

    def test_read_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported path_or_object"):
            TrajectoryData.read(123, metadata={})

    def test_read_empty_list_raises(self):
        with pytest.raises(ValueError, match="Empty list"):
            TrajectoryData.read([], metadata={})

    def test_read_metadata_and_temperature(self, ase_atoms_list):
        td = TrajectoryData.read(
            ase_atoms_list,
            time_step=1.0,
            step_skip=2,
            temperature=300.0,
            metadata={"extra": "value"},
        )
        assert td.metadata["time_step"] == 1.0
        assert td.metadata["step_skip"] == 2
        assert td.metadata["temperature"] == 300.0
        assert td.metadata["extra"] == "value"
