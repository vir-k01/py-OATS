"""Instantaneous Normal Modes (INM) analyzer via Hessian / finite-difference forces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from ase.calculators.calculator import Calculator

from py_oats.analyzers.base import BaseAnalyzer
from py_oats.io.trajectory import TrajectoryData
from py_oats.utils.analyzers import inm as inm_util

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


@dataclass
class INMFrameResult:
    frame_index: int
    time_fs: float
    eigenvalues: np.ndarray  # (3N,)
    eigenvectors: np.ndarray  # (3N, 3N)
    omega: np.ndarray  # signed sqrt(|lambda|) in sqrt(eV/(Å^2 amu))
    omega_ps: np.ndarray  # signed, in ps^-1 (ASE unit convention)
    unstable_mask: np.ndarray  # (3N,)
    elements: list[str]
    element_participation: np.ndarray  # (3N, n_elements)


class INMAnalyzer(BaseAnalyzer):
    """
    Compute instantaneous normal modes at sampled frames by:
      1) Hessian (``get_hessian`` if available) or finite-difference forces,
      2) mass-weighting (dynamical matrix),
      3) diagonalization to obtain eigenvalues/eigenvectors,
      4) element participation weights per mode.

    **Frame parallelism is on by default**: each sampled frame runs as an independent task
    (thread pool) with a **fresh calculator copy** so native Hessian and forces stay isolated.

    **Column vs frame tradeoff** (defaults): column FD workers are chosen first from system
    size—**one** column thread below 100 atoms, then a **linear** ramp up to the remaining
    CPU budget by ~500 atoms. Frame workers are then
    ``min(frames, 8, cpus − column_workers)``, so when columns need few threads, more
    cores can go to **frames**; when columns ramp up, the frame pool shrinks accordingly.
    Overrides: ``frame_max_workers``, ``max_workers``, ``parallel_frames=False``.

    Calculators that are not copyable or pickleable (common with **pyace GRACE** potentials)
    should use ``calculator_factory=lambda: ...`` so each parallel task gets a new
    calculator instance; otherwise pass ``parallel_frames=False`` and ``parallel=False``.

    **CPU usage:** Finite-difference columns only run in parallel when ``parallel=True``
    (default). You still often see **one core** if: (1) the structure has **fewer than
    100 atoms** (defaults to **one** column worker—override with ``max_workers``), (2) only **one**
    frame is sampled (no frame pool), (3) the calculator exposes **``get_hessian``** so FD
    columns are skipped, or (4) the calculator is **Python-bound** (GIL). For small cells,
    e.g. ``max_workers=min(16, (os.cpu_count() or 8) - 8)`` together with ``parallel=True``.
    """

    def __init__(self, trajectory: TrajectoryData, calculator: Calculator) -> None:
        super().__init__(trajectory, name="inm_analyzer")
        self.calculator = calculator

        self.frame_indices: list[int] = []
        self.times_fs: list[float] = []
        self.frames: list[INMFrameResult] = []

        self.finite_difference_step: float | None = None
        self.compute_frequency: int | None = None

    def analyze(
        self,
        compute_frequency: int,
        finite_difference_step: float = 0.01,
        *,
        parallel: bool = False,
        max_workers: int | None = None,
        calculator_factory: Callable[[], Calculator] | None = None,
        parallel_frames: bool = True,
        frame_max_workers: int | None = None,
    ) -> None:
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

        num_frames = len(frame_indices)
        n_atoms = int(traj.n_atoms)

        if max_workers is not None:
            cw = max(1, int(max_workers))
        else:
            cw = inm_util.default_inm_column_pool_workers(
                num_frames, parallel_frames=parallel_frames, n_atoms=n_atoms
            )

        if frame_max_workers is not None:
            n_cpu = inm_util._cpu_count()
            fw = (
                min(
                    max(1, int(frame_max_workers)),
                    num_frames,
                    8,
                    max(1, n_cpu - cw),
                )
                if num_frames
                else 0
            )
        else:
            fw = inm_util.default_inm_frame_pool_workers(num_frames, cw)

        if parallel_frames:
            pairs = list(zip(frame_indices, self.times_fs))

            def _one_frame(pair: tuple[int, float]) -> INMFrameResult:
                idx, t_fs = pair
                return self._analyze_single_frame(
                    int(idx),
                    float(t_fs),
                    parallel=parallel,
                    max_workers=cw,
                    calculator_factory=calculator_factory,
                )

            results = inm_util.map_thread_pool(
                _one_frame,
                pairs,
                max_workers=fw,
                desc="INM frames",
            )
            self.frames = sorted(results, key=lambda r: r.frame_index)
        else:
            iterator = zip(frame_indices, self.times_fs)
            if _tqdm is not None:
                iterator = _tqdm(
                    iterator,
                    total=len(frame_indices),
                    desc="INM frames",
                )
            for idx, t_fs in iterator:
                self.frames.append(
                    self._analyze_single_frame(
                        int(idx),
                        float(t_fs),
                        parallel=parallel,
                        max_workers=cw,
                        calculator_factory=calculator_factory,
                    )
                )

    def _analyze_single_frame(
        self,
        idx: int,
        t_fs: float,
        *,
        parallel: bool,
        max_workers: int | None,
        calculator_factory: Callable[[], Calculator] | None,
    ) -> INMFrameResult:
        """Hessian → dynamical matrix → eigenpairs and participation for one trajectory frame."""
        traj = self.trajectory
        atoms = traj.atoms_at_frame(idx)
        atoms.calc = (
            calculator_factory()
            if calculator_factory is not None
            else inm_util.copy_calculator(self.calculator)
        )

        h = inm_util.hessian_native_or_finite_difference(
            atoms,
            float(self.finite_difference_step or 0.01),
            parallel=parallel,
            max_workers=max_workers,
            calculator_factory=calculator_factory,
        )
        h = 0.5 * (h + h.T)

        d = inm_util.mass_weighted_dynamical_matrix(h, atoms.get_masses())
        d = 0.5 * (d + d.T)

        lam, vecs = np.linalg.eigh(d)
        unstable = lam < 0.0
        omega = inm_util.omega_signed_from_lambda(lam)
        omega_ps = inm_util.omega_ps_inv_from_lambda(lam)

        elements = inm_util.element_list_from_species([str(s) for s in traj.species])
        element_part = inm_util.element_participation(vecs, traj.species, elements)

        return INMFrameResult(
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
