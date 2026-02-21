"""ASE-based trajectory reading. Returns dict compatible with TrajectoryData.from_dict()."""

from __future__ import annotations

import numpy as np
from typing import Any

from ase.atoms import Atoms

# Return dict keys: positions, species, lattices, properties, metadata (TrajectoryData.from_dict)


def _arrays_from_atoms_list(
    atoms_list: list[Atoms],
    n_frames: int,
    n_atoms: int,
) -> dict[str, np.ndarray]:
    """Collect ASE atoms.arrays (magmoms, charges, etc.) into (n_frames, n_atoms) arrays."""
    out: dict[str, list] = {}
    for i in range(n_frames):
        a = atoms_list[i]
        for key in a.arrays:
            if key in ("positions", "numbers"):
                continue
            vals = a.arrays[key]
            if key not in out:
                out[key] = []
            arr = np.asarray(vals, dtype=np.float64)
            if arr.shape == (n_atoms,):
                out[key].append(arr)
            elif arr.ndim == 2 and arr.shape == (n_atoms, 3):
                out[key].append(arr)
    result: dict[str, np.ndarray] = {}
    for key, lists in out.items():
        if len(lists) != n_frames:
            continue
        stacked = np.array(lists)
        if stacked.shape == (n_frames, n_atoms):
            result[key] = stacked
        elif stacked.shape == (n_frames, n_atoms, 3):
            for j, ax in enumerate("xyz"):
                result[f"{key}_{ax}"] = stacked[:, :, j]
    return result


def trajectory_to_data(
    trajectory: Any,  # ASE Trajectory or list of Atoms
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return TrajectoryData.from_dict-compatible dict from ASE trajectory or list of Atoms."""
    if isinstance(trajectory, list):
        return atoms_to_data(trajectory, metadata=metadata or {})
    atoms_list = [trajectory[i] for i in range(len(trajectory))]
    return atoms_to_data(atoms_list, metadata=metadata or {})


def atoms_to_data(
    atoms: list[Atoms],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return TrajectoryData.from_dict-compatible dict from list of ASE Atoms."""
    n_frames = len(atoms)
    if n_frames == 0:
        raise ValueError("Empty list of Atoms")

    a0 = atoms[0]
    n_atoms = len(a0)

    species = np.array(a0.get_chemical_symbols(), dtype=object)
    positions = np.array([a.get_positions() for a in atoms], dtype=np.float64)

    cell0 = np.asarray(a0.get_cell(), dtype=np.float64)
    if all(np.allclose(np.asarray(a.get_cell(), dtype=np.float64), cell0) for a in atoms):
        lattices = np.array(cell0, dtype=np.float64, copy=True)
    else:
        lattices = np.array([np.asarray(a.get_cell(), dtype=np.float64) for a in atoms], dtype=np.float64)

    meta = dict(metadata or {})
    properties = _arrays_from_atoms_list(atoms, n_frames, n_atoms)

    return {
        "positions": positions,
        "species": species,
        "lattices": lattices,
        "properties": properties,
        "metadata": meta,
    }
