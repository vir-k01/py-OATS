"""I/O trajectory data for downstream modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List
from pathlib import Path

import numpy as np
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from ase.atoms import Atoms as AseAtoms
from pymatgen.core.structure import Structure
from ase.trajectory import Trajectory as AseTrajectory
import ase.io
from .utils.io.ase import trajectory_to_data as ase_trajectory_to_data
from .utils.io.pymatgen import trajectory_to_data as pmg_trajectory_to_data
from .utils.io.ase import atoms_to_data
from .utils.io.pymatgen import structures_to_data

@dataclass
class TrajectoryData:
    """
    Lightweight, standardized in-memory trajectory.

    - positions: np.ndarray (n_frames, n_atoms, 3)
        Cartesian coordinates (Å)
    - species: list[str]
        Species strings (e.g. "Li", "Mn3+", "O").
    - lattices: np.ndarray (n_frames, 3, 3) or (3, 3)
        Lattice matrices; (3, 3) when constant across frames.
    - properties: dict[str, np.ndarray]
        Properties (e.g. magmoms, charges). 
        Arrays can be (n_frames, n_atoms), (n_atoms,) or (n_frames,) depending on the property.
    - metadata: dict[str, Any]
        Metadata (time_step, step_skip, temperature, source_path, etc.).
    """
    positions: np.ndarray
    species: np.ndarray[str]
    lattices: np.ndarray
    properties: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T, N, _ = self.positions.shape
        if self.species.shape != (N,):
            raise ValueError(
                f"species must be (n_atoms,), got {self.species.shape}"
            )
        if self.lattices.ndim == 2 and self.lattices.shape != (3, 3):
            raise ValueError(f"lattices must be (n_frames, 3, 3) or (3, 3), got {self.lattices.shape}")
        if self.lattices.ndim == 3 and self.lattices.shape != (T, 3, 3):
            raise ValueError(f"lattices must be (n_frames, 3, 3) or (3, 3), got {self.lattices.shape}")
        for name, array in self.properties.items():
            if array.shape != (T, N) and array.shape != (N,) and array.shape != (T,):
                raise ValueError(f"property {name} must be \
                    (n_frames, n_atoms), (n_atoms,) or (n_frames), got {array.shape}")

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[1]

    def lattice_at_frame(self, i: int) -> np.ndarray:
        """Lattice matrix for frame i (3, 3)."""
        if self.lattices.ndim == 2:
            return self.lattices
        return self.lattices[i]

    def positions_for_species(self, species_label: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Indices and positions for atoms matching species_label (e.g. "Li", "Mn3+").

        Returns:
            ind: np.ndarray (n_match,)
                Position indices for the given species.
            pos: np.ndarray (n_frames, n_match, 3)
                Positions for the given species.
        """
        ind = np.where(self.species == species_label)[0]
        pos = self.positions[:, ind, :]
        return ind, pos

    @classmethod
    def read(
        cls,
        path_or_object: Path | str | PmgTrajectory | AseTrajectory | List[AseAtoms] | List[Structure],
        time_step: float = 2.0,
        step_skip: int = 1,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryData:
        """
        Read a trajectory from a path or in-memory object into TrajectoryData.

        Args:
            path_or_object: File path to read, or trajectory object or list of structures/atoms objects (ASE/pymatgen).
            time_step: Time step in fs.
            step_skip: Step skip used when writing the trajectory to file.
            temperature: Temperature in K.
            metadata: Additional metadata to store in the TrajectoryData object (velocity, forces, charges,etc.).

        Returns:
            TrajectoryData instance.
        """
        meta: dict[str, Any] = dict(metadata) if metadata else {}
        meta.setdefault("time_step", time_step)
        meta.setdefault("step_skip", step_skip)
        if temperature is not None:
            meta["temperature"] = temperature
        if isinstance(path_or_object, (Path, str)):
            meta.setdefault("source_path", str(path_or_object))
            path_or_object = ase.io.read(str(path_or_object), index=":")

        if isinstance(path_or_object, PmgTrajectory):
            return pmg_trajectory_to_data(path_or_object, metadata=meta)
        elif isinstance(path_or_object, AseTrajectory):
            return ase_trajectory_to_data(path_or_object, metadata=meta)
        elif isinstance(path_or_object, List):
            if path_or_object and isinstance(path_or_object[0], AseAtoms):
                return atoms_to_data(path_or_object, metadata=meta)
            elif path_or_object and isinstance(path_or_object[0], Structure):
                return structures_to_data(path_or_object, metadata=meta)
            else:
                raise ValueError(f"Unsupported list of objects: {type(path_or_object[0])}")
        else:
            raise ValueError(f"Unsupported path_or_object type: {type(path_or_object)}")
