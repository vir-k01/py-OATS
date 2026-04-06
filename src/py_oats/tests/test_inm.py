from unittest.mock import patch

import numpy as np
import pytest

from ase.calculators.calculator import Calculator, all_changes

from py_oats.analyzers.inm import INMAnalyzer
from py_oats.io.trajectory import TrajectoryData
from py_oats.schemas.inm import INMDoc
from py_oats.utils.analyzers import inm as inm_mod


class HarmonicWell(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.k = float(k)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        r = atoms.get_positions()
        e = 0.5 * self.k * float(np.sum(r**2))
        f = -self.k * r
        self.results["energy"] = e
        self.results["forces"] = f


def test_inm_harmonic_well_eigenvalues_and_metrics():
    # Li2O2 over 3 frames (4 atoms, 12 DOF).
    # For the harmonic well used here, the Hessian is k * I (no coupling),
    # so eigenvalues are simply k / m_i repeated for x/y/z per atom.
    positions = np.array(
        [
            # frame 0
            [
                [0.10, 0.00, 0.00],  # Li
                [0.00, 0.10, 0.00],  # Li
                [0.00, 0.00, 0.10],  # O
                [-0.10, 0.00, 0.00],  # O
            ],
            # frame 1
            [
                [0.12, 0.00, 0.00],
                [0.00, 0.12, 0.00],
                [0.00, 0.00, 0.12],
                [-0.12, 0.00, 0.00],
            ],
            # frame 2
            [
                [0.14, 0.00, 0.00],
                [0.00, 0.14, 0.00],
                [0.00, 0.00, 0.14],
                [-0.14, 0.00, 0.00],
            ],
        ],
        dtype=float,
    )  # (T=3, N=4, 3)
    species = np.array(["Li", "Li", "O", "O"], dtype=object)
    lattices = np.eye(3)
    traj = TrajectoryData(
        positions=positions,
        species=species,
        lattices=lattices,
        properties={},
        metadata={"time_step": 2.0, "step_skip": 1},
        composition="Li2O2",
    )

    k = 3.0
    analyzer = INMAnalyzer(traj, calculator=HarmonicWell(k=k))
    analyzer.analyze(compute_frequency=1, finite_difference_step=0.01)

    assert len(analyzer.frames) == 3

    for fr in analyzer.frames:
        assert fr.eigenvalues.shape == (12,)
        assert fr.eigenvectors.shape == (12, 12)
        assert fr.unstable_mask.shape == (12,)
        assert not np.any(fr.unstable_mask)

        masses = traj.atoms_at_frame(fr.frame_index).get_masses()
        # masses: [Li, Li, O, O]
        expected = np.sort(
            np.concatenate(
                [
                    np.full(6, k / masses[0]),  # 2 Li atoms * 3 DOF
                    np.full(6, k / masses[2]),  # 2 O atoms * 3 DOF
                ]
            )
        )
        got = np.sort(fr.eigenvalues)
        assert np.allclose(got, expected, rtol=5e-3, atol=5e-3)

        # Element participation weights per mode sum to 1.
        assert fr.element_participation.shape == (12, len(fr.elements))
        assert np.allclose(np.sum(fr.element_participation, axis=1), 1.0, atol=1e-8)

    doc = INMDoc.from_analyzer(analyzer, dos_bins=50)
    assert doc.num_atoms == 4
    assert doc.f_unstable_mean == 0.0
    assert all(v == 0.0 for v in doc.f_unstable_by_element_mean.values())

    # Element-resolved DOS should sum back to total DOS (weights per mode sum to 1).
    # Check all per-frame DOS entries (exclude the final trajectory-average entry).
    for per_frame_dos in doc.dos[:-1]:
        summed = None
        for arr in per_frame_dos.element_rho_total.values():
            summed = arr if summed is None else summed + arr
        assert summed is not None
        assert np.allclose(summed, per_frame_dos.rho_total, atol=1e-10)


# --- INM thread-pool defaults (frame vs column split) ---


@pytest.mark.parametrize(
    "cpu,n_frames,column_workers,expected",
    [
        (32, 100, 24, 8),
        (32, 3, 1, 3),
        (4, 10, 1, 3),
        (4, 2, 1, 2),
        (1, 100, 1, 1),
    ],
)
def test_default_inm_frame_pool_workers(cpu, n_frames, column_workers, expected):
    with patch.object(inm_mod.os, "cpu_count", return_value=cpu):
        assert inm_mod.default_inm_frame_pool_workers(n_frames, column_workers) == expected


def test_default_inm_frame_pool_workers_zero_frames():
    with patch.object(inm_mod.os, "cpu_count", return_value=32):
        assert inm_mod.default_inm_frame_pool_workers(0, 1) == 0


def test_default_inm_frame_pool_workers_cpu_none_falls_back_to_one():
    with patch.object(inm_mod.os, "cpu_count", return_value=None):
        assert inm_mod.default_inm_frame_pool_workers(5, 1) == 1


@pytest.mark.parametrize(
    "cpu,n_frames,parallel_frames,n_atoms,expected_cw",
    [
        (32, 100, True, 500, 24),
        (32, 3, True, 500, 29),
        (8, 8, True, 50, 1),
        (8, 0, True, 500, 8),
        (8, 3, False, 500, 8),
        (4, 2, True, 500, 2),
        (32, 100, True, 50, 1),
    ],
)
def test_default_inm_column_pool_workers(cpu, n_frames, parallel_frames, n_atoms, expected_cw):
    with patch.object(inm_mod.os, "cpu_count", return_value=cpu):
        assert (
            inm_mod.default_inm_column_pool_workers(n_frames, parallel_frames, n_atoms)
            == expected_cw
        )


def test_linear_column_workers_mid_range():
    # cpu=32, 100 frames -> max_col=24; at 300 atoms, halfway between 100 and 500 -> ~half of 23 + 1
    with patch.object(inm_mod.os, "cpu_count", return_value=32):
        assert inm_mod.default_inm_column_pool_workers(100, True, 300) == 12


def _li2o2_three_frame_traj() -> TrajectoryData:
    positions = np.array(
        [
            [
                [0.10, 0.00, 0.00],
                [0.00, 0.10, 0.00],
                [0.00, 0.00, 0.10],
                [-0.10, 0.00, 0.00],
            ],
            [
                [0.12, 0.00, 0.00],
                [0.00, 0.12, 0.00],
                [0.00, 0.00, 0.12],
                [-0.12, 0.00, 0.00],
            ],
            [
                [0.14, 0.00, 0.00],
                [0.00, 0.14, 0.00],
                [0.00, 0.00, 0.14],
                [-0.14, 0.00, 0.00],
            ],
        ],
        dtype=float,
    )
    species = np.array(["Li", "Li", "O", "O"], dtype=object)
    return TrajectoryData(
        positions=positions,
        species=species,
        lattices=np.eye(3),
        properties={},
        metadata={"time_step": 2.0, "step_skip": 1},
        composition="Li2O2",
    )


def test_analyze_resolves_frame_and_column_workers(monkeypatch):
    traj = _li2o2_three_frame_traj()
    analyzer = INMAnalyzer(traj, calculator=HarmonicWell(k=3.0))

    fw_seen = []
    cw_seen = []

    real_map = inm_mod.map_thread_pool

    def map_wrapped(fn, items, max_workers=None, desc=None):
        if desc == "INM frames":
            fw_seen.append(max_workers)
        return real_map(fn, items, max_workers=max_workers, desc=desc)

    real_hess = inm_mod.hessian_native_or_finite_difference

    def hess_wrapped(atoms, step, *, parallel, max_workers, calculator_factory):
        cw_seen.append(max_workers)
        return real_hess(
            atoms,
            step,
            parallel=parallel,
            max_workers=max_workers,
            calculator_factory=calculator_factory,
        )

    monkeypatch.setattr(inm_mod, "map_thread_pool", map_wrapped)
    monkeypatch.setattr(inm_mod, "hessian_native_or_finite_difference", hess_wrapped)

    with patch.object(inm_mod.os, "cpu_count", return_value=32):
        analyzer.analyze(compute_frequency=1, parallel=True, parallel_frames=True)

    assert fw_seen == [3]
    assert cw_seen == [1, 1, 1]


def test_analyze_parallel_frames_false_small_system_one_column_worker(monkeypatch):
    traj = _li2o2_three_frame_traj()
    analyzer = INMAnalyzer(traj, calculator=HarmonicWell(k=3.0))

    cw_seen = []

    real_hess = inm_mod.hessian_native_or_finite_difference

    def hess_wrapped(atoms, step, *, parallel, max_workers, calculator_factory):
        cw_seen.append(max_workers)
        return real_hess(
            atoms,
            step,
            parallel=parallel,
            max_workers=max_workers,
            calculator_factory=calculator_factory,
        )

    monkeypatch.setattr(inm_mod, "hessian_native_or_finite_difference", hess_wrapped)

    with patch.object(inm_mod.os, "cpu_count", return_value=8):
        analyzer.analyze(
            compute_frequency=1,
            parallel=True,
            parallel_frames=False,
        )

    assert cw_seen == [1, 1, 1]


def test_analyze_explicit_frame_max_workers_capped_by_num_frames(monkeypatch):
    traj = _li2o2_three_frame_traj()
    analyzer = INMAnalyzer(traj, calculator=HarmonicWell(k=3.0))

    fw_seen = []

    real_map = inm_mod.map_thread_pool

    def map_wrapped(fn, items, max_workers=None, desc=None):
        if desc == "INM frames":
            fw_seen.append(max_workers)
        return real_map(fn, items, max_workers=max_workers, desc=desc)

    monkeypatch.setattr(inm_mod, "map_thread_pool", map_wrapped)

    with patch.object(inm_mod.os, "cpu_count", return_value=32):
        analyzer.analyze(
            compute_frequency=1,
            parallel=True,
            parallel_frames=True,
            frame_max_workers=16,
        )

    assert fw_seen == [3]


def test_analyze_explicit_max_workers_overrides_column_default(monkeypatch):
    traj = _li2o2_three_frame_traj()
    analyzer = INMAnalyzer(traj, calculator=HarmonicWell(k=3.0))

    cw_seen = []

    real_hess = inm_mod.hessian_native_or_finite_difference

    def hess_wrapped(atoms, step, *, parallel, max_workers, calculator_factory):
        cw_seen.append(max_workers)
        return real_hess(
            atoms,
            step,
            parallel=parallel,
            max_workers=max_workers,
            calculator_factory=calculator_factory,
        )

    monkeypatch.setattr(inm_mod, "hessian_native_or_finite_difference", hess_wrapped)

    with patch.object(inm_mod.os, "cpu_count", return_value=32):
        analyzer.analyze(
            compute_frequency=1,
            parallel=True,
            parallel_frames=True,
            max_workers=4,
        )

    assert cw_seen == [4, 4, 4]


def test_map_thread_pool_none_uses_cpu_count():
    captured = []
    real_ex = inm_mod.ThreadPoolExecutor

    def capture_executor(*args, **kwargs):
        captured.append(kwargs.get("max_workers"))
        return real_ex(*args, **kwargs)

    with patch.object(inm_mod.os, "cpu_count", return_value=7):
        with patch.object(inm_mod, "ThreadPoolExecutor", side_effect=capture_executor):
            inm_mod.map_thread_pool(lambda x: x + 1, [1, 2], max_workers=None)

    assert captured == [7]


def test_finite_difference_hessian_none_max_workers_small_system_one_worker():
    from ase import Atoms

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], calculator=HarmonicWell(k=2.0))
    captured = []

    real_map = inm_mod.map_thread_pool

    def map_wrapped(fn, items, max_workers=None, desc=None):
        captured.append(max_workers)
        return real_map(fn, items, max_workers=max_workers, desc=desc)

    with patch.object(inm_mod, "map_thread_pool", map_wrapped):
        with patch.object(inm_mod.os, "cpu_count", return_value=11):
            inm_mod.finite_difference_hessian(
                atoms,
                0.01,
                parallel=True,
                max_workers=None,
                calculator_factory=None,
            )

    assert captured == [1]

