"""Pymatgen-based trajectory reading. Returns dict compatible with TrajectoryData.from_dict()."""

from __future__ import annotations

import numpy as np
from typing import Any

from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from pymatgen.core.structure import Structure

# Return dict keys: positions, species, lattices, properties, metadata (TrajectoryData.from_dict)


def _site_properties_to_arrays(
    structures: list[Structure],
    n_frames: int,
    n_atoms: int,
) -> dict[str, np.ndarray]:
    """Collect structure.site_properties (charges, magmoms, etc.) into (n_frames, n_atoms) arrays."""
    out: dict[str, list] = {}
    for i in range(n_frames):
        site_props = structures[i].site_properties or {}
        for key, vals in site_props.items():
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


def _frame_properties_to_arrays(
    frame_props: Any, n_frames: int, n_atoms: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Split trajectory.frame_properties into:
    - properties: scalar per frame (n_frames,) or per-frame (n_atoms,3) -> _x,_y,_z (n_frames, n_atoms)
    - metadata: raw dict for anything we don't convert
    """
    properties: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    if frame_props is None or (hasattr(frame_props, "__len__") and len(frame_props) != n_frames):
        if frame_props is not None:
            meta["frame_properties"] = frame_props
        return properties, meta

    def get_frame(i: int) -> dict:
        try:
            d = frame_props[i]
        except (KeyError, IndexError, TypeError):
            return {}
        return d or {}

    for key in set().union(*(get_frame(i).keys() for i in range(n_frames))):
        vals = [get_frame(i).get(key) for i in range(n_frames)]
        if all(v is None for v in vals):
            continue
        try:
            arr = np.asarray(vals)
        except (ValueError, TypeError):
            meta[key] = vals
            continue
        if arr.ndim == 1 and arr.size == n_frames:
            properties[key] = arr.astype(np.float64, copy=False)
        elif arr.ndim == 3 and arr.shape == (n_frames, n_atoms, 3):
            for j, ax in enumerate("xyz"):
                properties[f"{key}_{ax}"] = arr[:, :, j]
        elif arr.shape == (n_frames, n_atoms):
            properties[key] = arr.astype(np.float64, copy=False)
        elif arr.shape == (n_frames, 3, 3):
            # Stress tensor: store as stress_xx, stress_xy, ... each (n_frames,)
            for i, a in enumerate("xyz"):
                for j, b in enumerate("xyz"):
                    properties[f"{key}_{a}{b}"] = arr[:, i, j].astype(np.float64, copy=False)
        else:
            meta[key] = vals

    return properties, meta


def trajectory_to_data(
    trajectory: PmgTrajectory,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return TrajectoryData.from_dict-compatible dict from pymatgen Trajectory."""
    n_frames = len(trajectory)
    if n_frames == 0:
        raise ValueError("Empty pymatgen trajectory")

    structures = [trajectory.get_structure(i) for i in range(n_frames)]
    struct0 = structures[0]
    n_atoms = len(struct0)

    species = np.array([str(site.specie) for site in struct0], dtype=object)
    positions = np.array([s.cart_coords for s in structures], dtype=np.float64)
    lattice0 = struct0.lattice.matrix
    if all(np.allclose(s.lattice.matrix, lattice0) for s in structures):
        lattices = np.array(lattice0, dtype=np.float64, copy=True)
    else:
        lattices = np.array([s.lattice.matrix for s in structures], dtype=np.float64)

    meta = dict(metadata)
    if getattr(trajectory, "time_step", None) is not None:
        meta.setdefault("time_step", trajectory.time_step)
    if getattr(trajectory, "step_skip", None) is not None:
        meta.setdefault("step_skip", trajectory.step_skip)

    properties = _site_properties_to_arrays(structures, n_frames, n_atoms)
    frame_props = getattr(trajectory, "frame_properties", None)
    fp_arrays, fp_meta = _frame_properties_to_arrays(frame_props, n_frames, n_atoms)
    properties.update(fp_arrays)
    meta.update(fp_meta)

    return {
        "positions": positions,
        "species": species,
        "lattices": lattices,
        "properties": properties,
        "metadata": meta,
    }


def structures_to_data(
    structures: list[Structure],
    *,
    metadata: dict[str, Any] | None = None,
    frame_properties: Any = None,
) -> dict[str, Any]:
    """Return TrajectoryData.from_dict-compatible dict from list of pymatgen Structures."""
    n_frames = len(structures)
    if n_frames == 0:
        raise ValueError("Empty list of structures")

    struct0 = structures[0]
    n_atoms = len(struct0)

    species = np.array([str(site.specie) for site in struct0], dtype=object)
    positions = np.array([s.cart_coords for s in structures], dtype=np.float64)
    lattice0 = struct0.lattice.matrix
    if all(np.allclose(s.lattice.matrix, lattice0) for s in structures):
        lattices = np.array(lattice0, dtype=np.float64, copy=True)
    else:
        lattices = np.array([s.lattice.matrix for s in structures], dtype=np.float64)

    meta = dict(metadata or {})
    properties = _site_properties_to_arrays(structures, n_frames, n_atoms)
    fp = frame_properties if frame_properties is not None else meta.pop("frame_properties", None)
    fp_arrays, fp_meta = _frame_properties_to_arrays(fp, n_frames, n_atoms)
    properties.update(fp_arrays)
    meta.update(fp_meta)

    return {
        "positions": positions,
        "species": species,
        "lattices": lattices,
        "properties": properties,
        "metadata": meta,
    }
