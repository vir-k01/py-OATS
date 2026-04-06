"""Instantaneous Normal Modes (INM) analyzer via finite-difference Hessians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator

from py_oats.analyzers.base import BaseAnalyzer
from py_oats.io.trajectory import TrajectoryData
from py_oats.utils.analyzers.charge_state.charge import _element_from_species_string

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


_EV_PER_A2_PER_AMU_TO_S2 = (1.602176634e-19) / (1.66053906660e-27) / (1e-20)
_SQRT_EV_PER_A2_PER_AMU_TO_PS_INV = float(np.sqrt(_EV_PER_A2_PER_AMU_TO_S2) * 1e-12)


def _mass_weighted_dynamical_matrix(hessian: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    """
    Build D = H / sqrt(m_i m_j) where m repeats for x/y/z.

    H units: eV/Å^2, masses in amu => D in eV/(Å^2 amu).
    """
    m3 = np.repeat(masses_amu.astype(np.float64, copy=False), 3)
    mw = np.sqrt(m3)
    return hessian / np.outer(mw, mw)


def _omega_signed_from_lambda(lam: np.ndarray) -> np.ndarray:
    return np.sign(lam) * np.sqrt(np.abs(lam))


def _omega_ps_inv_from_lambda(lam: np.ndarray) -> np.ndarray:
    return _omega_signed_from_lambda(lam) * _SQRT_EV_PER_A2_PER_AMU_TO_PS_INV


def _element_list_from_species(species: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in species:
        el = _element_from_species_string(str(s))
        if el not in seen:
            seen.add(el)
            out.append(el)
    return out


@dataclass
class INMFrameResult:
    frame_index: int
    time_fs: float
    eigenvalues: np.ndarray  # (3N,)
    eigenvectors: np.ndarray  # (3N, 3N)
    omega: np.ndarray  # signed sqrt(|lambda|) in sqrt(eV/(Å^2 amu))
    omega_ps: np.ndarray  # signed, in ps^-1 (ASE unit conversion)
    unstable_mask: np.ndarray  # (3N,)
    elements: list[str]
    element_participation: np.ndarray  # (3N, n_elements)


class INMAnalyzer(BaseAnalyzer):
    """
    Compute instantaneous normal modes at sampled frames by:
      1) finite-difference Hessian from ASE forces (central differences),
      2) mass-weighting (dynamical matrix),
      3) diagonalization to obtain eigenvalues/eigenvectors,
      4) element participation weights per mode.
    """

    def __init__(self, trajectory: TrajectoryData, calculator: Calculator) -> None:
        super().__init__(trajectory, name="inm_analyzer")
        self.calculator = calculator

        self.frame_indices: list[int] = []
        self.times_fs: list[float] = []
        self.frames: list[INMFrameResult] = []

        self.finite_difference_step: float | None = None
        self.compute_frequency: int | None = None

    def analyze(self, compute_frequency: int, finite_difference_step: float = 0.01) -> None:
        if compute_frequency == 0:
            raise ValueError("compute_frequency must be non-zero")
        self.compute_frequency = int(compute_frequency)
        self.finite_difference_step = float(finite_difference_step)

        traj = self.trajectory
        time_step = float(traj.metadata.get("time_step", 2.0))
        step_skip = int(traj.metadata.get("step_skip", 1))

        frame_indices = (
            [0]
            if compute_frequency < 0
            else list(range(0, traj.n_frames, compute_frequency))
        )

        self.frame_indices = frame_indices
        self.times_fs = [time_step * step_skip * i for i in frame_indices]
        self.frames = []

        iterator = zip(frame_indices, self.times_fs)
        if _tqdm is not None:
            iterator = _tqdm(
                iterator,
                total=len(frame_indices),
                desc="INM frames",
            )

        for idx, t_fs in iterator:
            atoms = traj.atoms_at_frame(idx)
            atoms.calc = self.calculator
            h = self._hessian(atoms, step=self.finite_difference_step)
            # enforce symmetry
            h = 0.5 * (h + h.T)

            d = _mass_weighted_dynamical_matrix(h, atoms.get_masses())
            d = 0.5 * (d + d.T) # in case of numerical noise with inverting mass matrix

            lam, vecs = np.linalg.eigh(d)
            unstable = lam < 0.0
            omega = _omega_signed_from_lambda(lam)
            omega_ps = _omega_ps_inv_from_lambda(lam)

            elements = _element_list_from_species([str(s) for s in traj.species])
            element_part = self._element_participation(vecs, traj.species, elements)

            self.frames.append(
                INMFrameResult(
                    frame_index=int(idx),
                    time_fs=float(t_fs),
                    eigenvalues=lam,
                    eigenvectors=vecs,
                    omega=omega,
                    omega_ps=omega_ps,
                    unstable_mask=unstable,
                    elements=elements,
                    element_participation=element_part,
                )
            )

    def _hessian(self, atoms: Atoms, step: float) -> np.ndarray:
        """
        Prefer calculator-native Hessian if available, else finite-difference via forces.

        The intended fast path is:
            calc.get_hessian(atoms=atoms)
        and on any error we fall back to finite differences.
        """
        calc = atoms.calc
        if calc is not None:
            try:
                h = calc.get_hessian(atoms=atoms)
                return np.asarray(h, dtype=np.float64)
            except Exception:
                pass
        return self._finite_difference_hessian(atoms, step=step)

    def _finite_difference_hessian(self, atoms: Atoms, step: float) -> np.ndarray:
        """
        Central difference Hessian via forces:
          H_{iα,jβ} = - dF_{iα}/d r_{jβ}
        """
        step = float(step)
        if step <= 0:
            raise ValueError("finite_difference_step must be > 0")

        n = len(atoms)
        dof = 3 * n
        h = np.zeros((dof, dof), dtype=np.float64)

        pos0 = atoms.get_positions().copy()

        # Flatten index mapping: q = 3*j + beta
        for q in range(dof):
            j = q // 3
            beta = q % 3

            pos = pos0.copy()
            pos[j, beta] += step
            atoms.set_positions(pos)
            f_plus = atoms.get_forces().reshape(-1)

            pos[j, beta] -= 2.0 * step
            atoms.set_positions(pos)
            f_minus = atoms.get_forces().reshape(-1)

            # restore for next loop iteration
            atoms.set_positions(pos0)

            dF = (f_plus - f_minus) / (2.0 * step)
            h[:, q] = -dF

        return h

    def _element_participation(
        self,
        eigenvectors: np.ndarray,
        species: Sequence[str],
        elements: Sequence[str],
    ) -> np.ndarray:
        """
        For each mode k, compute w_E(k) = sum_{i in E} sum_{α} |e_{iα,k}|^2.

        Returns:
            (3N, n_elements) array where each row k sums to ~1.
        """
        n_atoms = len(species)
        dof = 3 * n_atoms
        if eigenvectors.shape != (dof, dof):
            raise ValueError("eigenvectors must be (3N, 3N)")

        # per-atom participation per mode: p_i(k) = sum_{α} |e_{iα,k}|^2
        # eigenvectors columns are modes from eigh
        vec = eigenvectors.reshape(n_atoms, 3, dof)
        p_atom = np.sum(vec**2, axis=1)  # (n_atoms, dof)

        out = np.zeros((dof, len(elements)), dtype=np.float64)
        atom_elements = np.array([_element_from_species_string(str(s)) for s in species], dtype=object)

        for e_idx, el in enumerate(elements):
            mask = atom_elements == el
            if np.any(mask):
                out[:, e_idx] = np.sum(p_atom[mask, :], axis=0)
        return out

