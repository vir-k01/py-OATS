"""Tests for PBC unwrapping."""

import numpy as np
import pytest

from py_oats_overhaul.utils.io.unwrap import unwrap_positions


def test_unwrap_positions_constant_lattice():
    """Unwrap with (3, 3) lattice: minimum-image displacement between frames."""
    L = np.eye(3) * 10.0  # 10 Å cube
    # Two frames: atom at frac 0.1 then 0.9. Delta 0.8 -> minimum image -0.2, so unwrapped frac 0.1, -0.1
    frac0 = np.array([0.1, 0.0, 0.0])
    frac1 = np.array([0.9, 0.0, 0.0])
    positions = np.array([
        [frac0 @ L.T],
        [frac1 @ L.T],
    ])  # (2, 1, 3) -> cart [1,0,0], [9,0,0]
    unwrapped = unwrap_positions(positions, L)
    np.testing.assert_allclose(unwrapped[0], positions[0])
    # Unwrapped: second frame frac -0.1 -> cart -1
    np.testing.assert_allclose(unwrapped[1], [[-1.0, 0.0, 0.0]])


def test_unwrap_positions_wrap_back():
    """Atom goes 0.9 -> 0.1 in frac: delta -0.8 -> correct to +0.2."""
    L = np.eye(3) * 10.0
    frac0 = np.array([0.9, 0.0, 0.0])
    frac1 = np.array([0.1, 0.0, 0.0])
    positions = np.array([
        [frac0 @ L.T],
        [frac1 @ L.T],
    ])
    unwrapped = unwrap_positions(positions, L)
    np.testing.assert_allclose(unwrapped[0], [[9.0, 0.0, 0.0]])
    # Unwrapped frac: 0.9, 0.9+0.2 = 1.1 -> cart 11
    np.testing.assert_allclose(unwrapped[1], [[11.0, 0.0, 0.0]])


def test_unwrap_positions_single_frame():
    """Single frame returns same positions."""
    positions = np.random.randn(1, 5, 3).astype(np.float64)
    lattices = np.eye(3, dtype=np.float64)
    unwrapped = unwrap_positions(positions, lattices)
    np.testing.assert_allclose(unwrapped, positions)


def test_unwrap_positions_shape():
    """Output shape matches input."""
    T, N = 10, 4
    positions = np.random.randn(T, N, 3).astype(np.float64)
    lattices = np.eye(3, dtype=np.float64)
    unwrapped = unwrap_positions(positions, lattices)
    assert unwrapped.shape == (T, N, 3)
