"""INM analysis helpers: Hessian, dynamical matrix, participation, parallel maps."""

from __future__ import annotations

import copy
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Sequence, TypeVar

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator

from py_oats.utils.analyzers.charge_state.charge import _element_from_species_string

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None

T = TypeVar("T")
R = TypeVar("R")

# eV / (Å² amu) → 1/s², then √(…) → 1/s; ×1e-12 → ps⁻¹
_EV_PER_A2_PER_AMU_TO_S2 = (1.602176634e-19) / (1.66053906660e-27) / (1e-20)
SQRT_EV_PER_A2_PER_AMU_TO_PS_INV = float(np.sqrt(_EV_PER_A2_PER_AMU_TO_S2) * 1e-12)


def _cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


# Below this atom count, parallel FD Hessian columns use a single thread (overhead dominates).
INM_COLUMN_SINGLE_WORKER_BELOW_ATOMS = 100
# From 100 atoms to this count, column workers ramp linearly up to the available budget.
INM_COLUMN_LINEAR_FULL_ATOMS = 500


def _max_column_budget(num_frames: int, parallel_frames: bool) -> int:
    """Upper bound on column-pool size before applying system-size scaling."""
    n = _cpu_count()
    if not parallel_frames or num_frames <= 0:
        return max(1, n)
    frame_reservation = min(num_frames, 8, n)
    return max(1, n - frame_reservation)


def _linear_column_workers(n_atoms: int, max_col: int) -> int:
    """
    Column FD workers: 1 if ``n_atoms < INM_COLUMN_SINGLE_WORKER_BELOW_ATOMS``; otherwise
    linear from 1 (at 100 atoms) up to ``max_col`` (reached at ``INM_COLUMN_LINEAR_FULL_ATOMS``).
    """
    if n_atoms < INM_COLUMN_SINGLE_WORKER_BELOW_ATOMS:
        return 1
    if max_col <= 1:
        return 1
    n0 = INM_COLUMN_SINGLE_WORKER_BELOW_ATOMS
    n1 = INM_COLUMN_LINEAR_FULL_ATOMS
    if n_atoms >= n1:
        return max_col
    t = (n_atoms - n0) / float(n1 - n0)
    return 1 + int(t * (max_col - 1))


def default_inm_column_pool_workers(
    num_frames: int, parallel_frames: bool, n_atoms: int
) -> int:
    """
    Thread budget for parallel finite-difference Hessian columns.

    Scales with structure size: one column worker below :data:`INM_COLUMN_SINGLE_WORKER_BELOW_ATOMS`
    atoms, then ramps linearly up to the column budget (remainder of CPUs after the frame cap).
    """
    max_col = _max_column_budget(num_frames, parallel_frames)
    return min(max_col, _linear_column_workers(n_atoms, max_col))


def default_inm_frame_pool_workers(num_frames: int, column_workers: int) -> int:
    """
    Thread budget for parallel over frames after reserving ``column_workers`` for the
    column FD pool: ``min(num_frames, 8, max(1, cpu_count - column_workers))``.
    """
    if num_frames <= 0:
        return 0
    n = _cpu_count()
    cw = max(0, int(column_workers))
    return min(num_frames, 8, max(1, n - cw))


def map_thread_pool(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int | None,
    desc: str | None = None,
) -> list[R]:
    """
    Run ``fn(x)`` for each ``x`` in parallel (completion order; not necessarily ``items`` order).

    If ``max_workers <= 1`` or a single item, runs sequentially (no thread pool).

    ``max_workers`` should usually be set explicitly. If ``None``, uses all CPUs
    (``max(1, cpu_count)``).
    """
    if not items:
        return []
    if max_workers is not None and max_workers <= 1:
        return [fn(x) for x in items]
    if len(items) == 1:
        return [fn(items[0])]

    mw = max(1, _cpu_count()) if max_workers is None else max(1, int(max_workers))
    out: list[R] = []
    with ThreadPoolExecutor(max_workers=mw) as ex:
        futures = [ex.submit(fn, x) for x in items]
        ac = as_completed(futures)
        if _tqdm is not None and desc is not None:
            ac = _tqdm(ac, total=len(futures), desc=desc)
        for fut in ac:
            out.append(fut.result())
    return out


def copy_calculator(calc: Calculator) -> Calculator:
    """
    Best-effort independent calculator for concurrent frame/column work.

    Some calculators (e.g. pyace GRACE with ``GRACEFSBasisSet``) cannot be ``deepcopy``'d
    because internal objects are not pickleable. We try ``calc.copy()``, then shallow
    ``copy.copy``, then ``deepcopy``. If all fail, raise with guidance to pass
    ``calculator_factory`` to :meth:`~py_oats.analyzers.inm.INMAnalyzer.analyze` or run
    with ``parallel_frames=False`` and ``parallel=False``.
    """
    if hasattr(calc, "copy"):
        try:
            c = calc.copy()
            if c is not None:
                return c
        except Exception:
            pass
    try:
        return copy.deepcopy(calc)
    except Exception:
        pass
    try:
        c = copy.copy(calc)
        if c is not calc:
            warnings.warn(
                "Calculator could not be deep-copied; using a shallow copy for parallel INM. "
                "If results are unstable, pass calculator_factory=... to build a fresh "
                "calculator per task, or disable parallelism.",
                UserWarning,
                stacklevel=2,
            )
            return c
    except Exception:
        pass
    raise RuntimeError(
        "Could not duplicate the ASE calculator for parallel INM (copy/deepcopy failed; "
        "common with pyace GRACE / GRACEFSBasisSet). Pass "
        "`calculator_factory=lambda: <construct a new calculator>` to "
        "`INMAnalyzer.analyze()`, or use `parallel_frames=False` and `parallel=False` "
        "to run sequentially with one calculator."
    )


def mass_weighted_dynamical_matrix(hessian: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    """
    D = H / sqrt(m_i m_j) with masses repeated for x/y/z.

    H: eV/Å², masses amu → D: eV/(Å² amu).
    """
    m3 = np.repeat(masses_amu.astype(np.float64, copy=False), 3)
    mw = np.sqrt(m3)
    return hessian / np.outer(mw, mw)


def omega_signed_from_lambda(lam: np.ndarray) -> np.ndarray:
    return np.sign(lam) * np.sqrt(np.abs(lam))


def omega_ps_inv_from_lambda(lam: np.ndarray) -> np.ndarray:
    return omega_signed_from_lambda(lam) * SQRT_EV_PER_A2_PER_AMU_TO_PS_INV


def element_list_from_species(species: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in species:
        el = _element_from_species_string(str(s))
        if el not in seen:
            seen.add(el)
            out.append(el)
    return out


def element_participation(
    eigenvectors: np.ndarray,
    species: Sequence[str],
    elements: Sequence[str],
) -> np.ndarray:
    """
    Per mode k: w_E(k) = sum_{i in E} sum_α |e_{iα,k}|². Shape (3N, n_elements).
    """
    n_atoms = len(species)
    dof = 3 * n_atoms
    if eigenvectors.shape != (dof, dof):
        raise ValueError("eigenvectors must be (3N, 3N)")

    vec = eigenvectors.reshape(n_atoms, 3, dof)
    p_atom = np.sum(vec**2, axis=1)

    out = np.zeros((dof, len(elements)), dtype=np.float64)
    atom_elements = np.array([_element_from_species_string(str(s)) for s in species], dtype=object)

    for e_idx, el in enumerate(elements):
        mask = atom_elements == el
        if np.any(mask):
            out[:, e_idx] = np.sum(p_atom[mask, :], axis=0)
    return out


def fd_hessian_one_column(
    q: int,
    atoms_template: Atoms,
    step: float,
    calculator_factory: Callable[[], Calculator] | None,
) -> tuple[int, np.ndarray]:
    """
    One Hessian column: H[:, q] = -dF/dr_q (central difference on coordinate q only).
    """
    atoms = atoms_template.copy()
    if calculator_factory is not None:
        atoms.calc = calculator_factory()
    else:
        atoms.calc = atoms_template.calc

    step = float(step)
    j = q // 3
    beta = q % 3

    pos0 = atoms.get_positions().copy()
    pos = pos0.copy()
    pos[j, beta] += step
    atoms.set_positions(pos)
    f_plus = atoms.get_forces().reshape(-1)

    pos[j, beta] -= 2.0 * step
    atoms.set_positions(pos)
    f_minus = atoms.get_forces().reshape(-1)

    atoms.set_positions(pos0)

    dF = (f_plus - f_minus) / (2.0 * step)
    col = -dF
    return q, col.astype(np.float64, copy=False)


def finite_difference_hessian(
    atoms: Atoms,
    step: float,
    *,
    parallel: bool,
    max_workers: int | None,
    calculator_factory: Callable[[], Calculator] | None,
) -> np.ndarray:
    """Central-difference Hessian via forces; optional parallel columns via ``map_thread_pool``."""
    step = float(step)
    if step <= 0:
        raise ValueError("finite_difference_step must be > 0")

    n = len(atoms)
    dof = 3 * n
    h = np.zeros((dof, dof), dtype=np.float64)

    if not parallel or dof == 0:
        pos0 = atoms.get_positions().copy()
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

            atoms.set_positions(pos0)

            dF = (f_plus - f_minus) / (2.0 * step)
            h[:, q] = -dF
        return h

    template = atoms.copy()
    template.calc = atoms.calc

    def _one(q: int) -> tuple[int, np.ndarray]:
        return fd_hessian_one_column(q, template, step, calculator_factory)

    if max_workers is None:
        col_workers = _linear_column_workers(len(atoms), max(1, _cpu_count()))
    else:
        col_workers = max(1, int(max_workers))
    results = map_thread_pool(
        _one,
        list(range(dof)),
        max_workers=col_workers,
        desc="INM Hessian columns",
    )
    for q, col in results:
        h[:, q] = col
    return h


def hessian_native_or_finite_difference(
    atoms: Atoms,
    step: float,
    *,
    parallel: bool,
    max_workers: int | None,
    calculator_factory: Callable[[], Calculator] | None,
) -> np.ndarray:
    """Try ``calc.get_hessian(atoms=...)``; on failure use finite-difference forces."""
    calc = atoms.calc
    if calc is not None:
        try:
            h = calc.get_hessian(atoms=atoms)
            return np.asarray(h, dtype=np.float64)
        except Exception:
            pass
    return finite_difference_hessian(
        atoms,
        step,
        parallel=parallel,
        max_workers=max_workers,
        calculator_factory=calculator_factory,
    )
