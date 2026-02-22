"""
PBC unwrapping of trajectory positions for continuous paths (e.g. for MSD/transport).
"""

from __future__ import annotations

import numpy as np


def unwrap_positions(
    positions: np.ndarray,
    lattices: np.ndarray,
) -> np.ndarray:
    """
    Unwrap positions across periodic boundaries so displacements are continuous.

    Converts to fractional coordinates, unwraps frame-to-frame differences, then
    converts back to cartesian using the lattice at each frame. Assumes lattice
    rows are basis vectors (cartesian = fractional @ lattice).

    Args:
        positions: (n_frames, n_atoms, 3) cartesian coordinates in Å.
        lattices: (3, 3) or (n_frames, 3, 3) lattice matrices.

    Returns:
        (n_frames, n_atoms, 3) unwrapped cartesian positions in Å.
    """
    positions = np.asarray(positions, dtype=np.float64)
    lattices = np.asarray(lattices, dtype=np.float64)
    T, N, _ = positions.shape

    if lattices.ndim == 2:
        lattices = np.broadcast_to(lattices, (T, 3, 3))
    elif lattices.ndim != 3 or lattices.shape != (T, 3, 3):
        raise ValueError(f"lattices must be (3, 3) or (n_frames, 3, 3), got {lattices.shape}")

    # Cartesian -> fractional: frac = cart @ inv(L)  (cart = frac @ L)
    frac = np.empty_like(positions)
    for t in range(T):
        frac[t] = positions[t] @ np.linalg.inv(lattices[t])

    # Unwrap: delta in fractional coords, correct jumps by ±1, then cumsum
    if T > 1:
        dfrac = frac[1:] - frac[:-1]  # (T-1, N, 3)
        dfrac = np.where(dfrac > 0.5, dfrac - 1.0, dfrac)
        dfrac = np.where(dfrac < -0.5, dfrac + 1.0, dfrac)
        # Unwrapped fractional path: start at frac[0], then add cumulative deltas
        frac_unwrap = np.empty_like(frac)
        frac_unwrap[0] = frac[0]
        np.cumsum(dfrac, axis=0, out=frac_unwrap[1:])
        frac_unwrap[1:] += frac[0]  # add initial position to each step
        frac = frac_unwrap

    # Fractional -> cartesian: cart = frac @ L (pymatgen: rows of L = a, b, c)
    out = np.empty_like(positions)
    for t in range(T):
        out[t] = frac[t] @ lattices[t]

    return out
