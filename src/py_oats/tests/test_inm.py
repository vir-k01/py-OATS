import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from py_oats.analyzers.inm import INMAnalyzer
from py_oats.io.trajectory import TrajectoryData
from py_oats.schemas.inm import INMDoc


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

