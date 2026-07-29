"""I/O trajectory data for downstream modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List
from pathlib import Path
from collections import Counter

import numpy as np
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from ase.atoms import Atoms as AseAtoms
from pymatgen.core.structure import Structure, Composition
import ase.io

from ..utils.io import ase as _ase_io
from ..utils.io import pymatgen as _pmg_io
from ..utils.io.unwrap import unwrap_positions


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
    composition: str | None = None
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

    @property
    def unique_species(self) -> list[str]:
        """Unique species labels that have at least one atom, in first-occurrence order."""
        seen: set[str] = set()
        out: list[str] = []
        for s in self.species:
            s = str(s)
            if s in seen:
                continue
            seen.add(s)
            _, pos = self.positions_for_species(s)
            if pos.shape[1] > 0:
                out.append(s)
        return out

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

    def as_dict(self) -> dict[str, Any]:
        """Return a dict with keys positions, species, lattices, properties, metadata, composition."""
        return {
            "positions": self.positions,
            "species": self.species,
            "lattices": self.lattices,
            "properties": self.properties,
            "metadata": self.metadata,
            "composition": self.composition,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrajectoryData:
        """Build TrajectoryData from a dict produced by `as_dict` or IO helpers."""
        return cls(
            positions=d["positions"],
            species=d["species"],
            lattices=d["lattices"],
            properties=d["properties"],
            metadata=d.get("metadata", {}),
            composition=d.get("composition", None),
        )

    @classmethod
    def read(
        cls,
        path_or_object: Path | str | PmgTrajectory | List[AseAtoms] | List[Structure] | Any,
        time_step: float = 2.0,
        step_skip: int = 1,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
        unwrap: bool = True,
    ) -> TrajectoryData:
        """
        Read a trajectory from a path or in-memory object into TrajectoryData.

        Args:
            path_or_object: File path (read via ase.io.read), list of ASE Atoms, list of Structures, or pymatgen Trajectory.
            time_step: Time step in fs.
            step_skip: Step skip used when writing the trajectory to file.
            temperature: Temperature in K.
            metadata: Additional metadata to store in the TrajectoryData object (velocity, forces, charges, etc.).
            unwrap: If True (default), unwrap positions across PBC so paths are continuous (for MSD/transport).

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

        data: dict[str, Any] | None = None
        if isinstance(path_or_object, PmgTrajectory):
            data = _pmg_io.trajectory_to_data(path_or_object, metadata=meta)
        elif isinstance(path_or_object, List):
            if not path_or_object:
                raise ValueError("Empty list of structures or atoms")
            if isinstance(path_or_object[0], AseAtoms):
                data = _ase_io.atoms_to_data(path_or_object, metadata=meta)
            elif isinstance(path_or_object[0], Structure):
                data = _pmg_io.structures_to_data(path_or_object, metadata=meta)
            else:
                raise ValueError(f"Unsupported list of objects: {type(path_or_object[0])}")
        if data is None:
            raise ValueError(f"Unsupported path_or_object type: {type(path_or_object)}")

        if data.get("composition") is None:
            # Derive a reduced formula from species by simple counting.
            counts = Counter(map(str, data["species"]))
            try:
                comp = Composition(counts)
                data["composition"] = str(comp.reduced_formula)
            except Exception:
                data["composition"] = None

        if unwrap:
            data["positions"] = unwrap_positions(data["positions"], data["lattices"])
            data.setdefault("metadata", {})["positions_unwrapped"] = True

        return cls.from_dict(data)

    def atoms_at_frame(self, i: int) -> AseAtoms:
        """Return ASE Atoms object for frame i."""
        atoms = AseAtoms(symbols=self.species[i], positions=self.positions[i], cell=self.lattice_at_frame(i))
        for key, arr in self.properties.items():
            if arr.shape == (self.n_atoms,):
                atoms.set_array(key, arr)
            elif arr.shape == (self.n_frames,):
                atoms.set_array(key, arr[i])
        return atoms
    
    def structure_at_frame(self, i: int) -> Structure:
        """Return pymatgen Structure for frame i."""
        struct = Structure(lattice=self.lattice_at_frame(i), species=[str(s) for s in self.species], coords=self.positions[i], coords_are_cartesian=True)
        for key, arr in self.properties.items():
            if arr.shape == (self.n_atoms,):
                struct.add_site_property(key, arr)
            elif arr.shape == (self.n_frames,):
                struct.add_site_property(key, arr[i])
        return struct