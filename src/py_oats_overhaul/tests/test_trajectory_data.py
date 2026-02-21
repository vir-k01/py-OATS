"""Tests for TrajectoryData dataclass and validation."""

import numpy as np
import pytest

from py_oats_overhaul.reader.trajectory import TrajectoryData


class TestTrajectoryDataConstruction:
    """Valid construction and basic properties."""

    def test_minimal_constant_lattice(
        self, positions, species, lattice_3x3, n_frames, n_atoms
    ):
        td = TrajectoryData(
            positions=positions,
            species=species,
            lattices=np.array(lattice_3x3, copy=True),
            properties={},
            metadata={},
        )
        assert td.n_frames == n_frames
        assert td.n_atoms == n_atoms
        assert td.positions.shape == (n_frames, n_atoms, 3)
        assert td.species.shape == (n_atoms,)
        assert td.lattices.ndim == 2 and td.lattices.shape == (3, 3)

    def test_variable_lattice(
        self, positions, species, lattices_variable, n_frames, n_atoms
    ):
        td = TrajectoryData(
            positions=positions,
            species=species,
            lattices=lattices_variable,
            properties={},
            metadata={},
        )
        assert td.lattices.shape == (n_frames, 3, 3)

    def test_with_properties(
        self, positions, species, lattice_3x3, properties_valid, n_frames, n_atoms
    ):
        td = TrajectoryData(
            positions=positions,
            species=species,
            lattices=np.array(lattice_3x3, copy=True),
            properties=properties_valid,
            metadata={"time_step": 2.0},
        )
        assert "magmoms" in td.properties
        assert td.properties["magmoms"].shape == (n_frames, n_atoms)
        assert td.properties["energy"].shape == (n_frames,)
        assert td.metadata["time_step"] == 2.0


class TestTrajectoryDataValidation:
    """Invalid inputs raise ValueError."""

    def test_species_wrong_length(self, positions, lattice_3x3):
        species_bad = np.array(["Li", "O"], dtype=object)  # 2 != n_atoms
        with pytest.raises(ValueError, match="species must be"):
            TrajectoryData(
                positions=positions,
                species=species_bad,
                lattices=np.array(lattice_3x3, copy=True),
                properties={},
                metadata={},
            )

    def test_lattices_wrong_shape_2d(self, positions, species):
        bad_lattice = np.eye(2, dtype=np.float64)  # (2,2) not (3,3)
        with pytest.raises(ValueError, match="lattices must be"):
            TrajectoryData(
                positions=positions,
                species=species,
                lattices=bad_lattice,
                properties={},
                metadata={},
            )

    def test_lattices_wrong_shape_3d(self, positions, species, n_frames):
        bad_lattices = np.eye(4, dtype=np.float64)[:n_frames]  # (T, 4, 4)
        with pytest.raises(ValueError, match="lattices must be"):
            TrajectoryData(
                positions=positions,
                species=species,
                lattices=bad_lattices,
                properties={},
                metadata={},
            )

    def test_property_wrong_shape(self, positions, species, lattice_3x3, n_frames):
        bad_props = {"x": np.zeros((n_frames + 1, 4), dtype=np.float64)}  # (T+1, N)
        with pytest.raises(ValueError, match="property .* must be"):
            TrajectoryData(
                positions=positions,
                species=species,
                lattices=np.array(lattice_3x3, copy=True),
                properties=bad_props,
                metadata={},
            )


class TestTrajectoryDataMethods:
    """lattice_at_frame and positions_for_species."""

    def test_lattice_at_frame_constant(self, traj_data_minimal, lattice_3x3):
        for i in range(traj_data_minimal.n_frames):
            np.testing.assert_array_almost_equal(
                traj_data_minimal.lattice_at_frame(i), lattice_3x3
            )

    def test_lattice_at_frame_variable(
        self, traj_data_variable_lattice, lattices_variable
    ):
        for i in range(traj_data_variable_lattice.n_frames):
            np.testing.assert_array_almost_equal(
                traj_data_variable_lattice.lattice_at_frame(i), lattices_variable[i]
            )

    def test_positions_for_species(
        self, traj_data_minimal, species, positions, n_frames
    ):
        ind, pos = traj_data_minimal.positions_for_species("O")
        assert ind.shape == (2,)  # two O in species
        assert pos.shape == (n_frames, 2, 3)
        np.testing.assert_array_almost_equal(
            pos, traj_data_minimal.positions[:, ind, :]
        )

    def test_positions_for_species_missing(self, traj_data_minimal):
        ind, pos = traj_data_minimal.positions_for_species("X")
        assert len(ind) == 0
        assert pos.shape == (traj_data_minimal.n_frames, 0, 3)


class TestTrajectoryDataFromDictAsDict:
    """from_dict and as_dict round-trip; utils return from_dict-compatible dict."""

    def test_as_dict_has_required_keys(self, traj_data_minimal):
        d = traj_data_minimal.as_dict()
        assert set(d.keys()) == {"positions", "species", "lattices", "properties", "metadata"}

    def test_from_dict_round_trip(self, traj_data_minimal):
        d = traj_data_minimal.as_dict()
        td = TrajectoryData.from_dict(d)
        assert td.n_frames == traj_data_minimal.n_frames
        assert td.n_atoms == traj_data_minimal.n_atoms
        np.testing.assert_array_almost_equal(td.positions, traj_data_minimal.positions)
        np.testing.assert_array_equal(td.species, traj_data_minimal.species)
        np.testing.assert_array_almost_equal(td.lattices, traj_data_minimal.lattices)
        assert list(td.properties.keys()) == list(traj_data_minimal.properties.keys())
        assert td.metadata == traj_data_minimal.metadata

    def test_from_dict_with_missing_metadata_defaults_to_empty(self, positions, species, lattice_3x3):
        d = {
            "positions": positions,
            "species": species,
            "lattices": np.array(lattice_3x3, copy=True),
            "properties": {},
        }
        td = TrajectoryData.from_dict(d)
        assert td.metadata == {}
