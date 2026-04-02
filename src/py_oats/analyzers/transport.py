"""
Onsager transport coefficient analyzer from trajectory MSDs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..io.trajectory import TrajectoryData
from ..utils.analyzers.transport.correlate import calc_Lii, calc_Lii_self, calc_Lij
from ..utils.analyzers.transport.fitting import fit_data
from .base import BaseAnalyzer


def correlation_pair_key(s1: str, s2: str) -> str:
    """Key for ``correlation_functions`` (JSON-friendly string, not a tuple)."""
    return f"{s1}-{s2}"


class TransportAnalyzer(BaseAnalyzer):
    """
    Compute the Onsager transport coefficients (L_ij) from TrajectoryData.

    All run parameters (temperature, time_step, step_skip) are read from
    trajectory.metadata. Species are the unique labels in trajectory.species that
    have at least one atom. L_tensor ordering matches self.species;
    correlation_functions uses ``correlation_pair_key(species_i, species_j)`` strings;
    fit_dicts are indexed by (i, j) species index.
    """

    def __init__(self, trajectory: TrajectoryData) -> None:
        """
        Args:
            trajectory: TrajectoryData. metadata must contain
                "temperature" to perform the fitting. Optionally "time_step", "step_skip".
        """
        super().__init__(trajectory, name="transport_analyzer")
        meta = self.trajectory.metadata

        if "temperature" not in meta:
            raise ValueError("trajectory.metadata must contain 'temperature'")
        self.temperature = float(meta["temperature"])
        self.time_step = float(meta.get("time_step", 2.0))
        self.step_skip = int(meta.get("step_skip", 1))

        self.traj_length = self.trajectory.n_frames
        self.species = self.trajectory.unique_species
        self.mapping = {s: i for i, s in enumerate(self.species)}
        self.inv_mapping = {i: s for i, s in enumerate(self.species)}

        self.times = self.time_step * np.linspace(
            0, self.traj_length * self.step_skip, self.traj_length, dtype=np.float64
        )
        self.kbT = 8.617333262e-5 * self.temperature * 10.0  # eV/atom, A^2/fs -> cm^2/s
        vol_ang3 = np.mean(
            [np.abs(np.linalg.det(self.trajectory.lattice_at_frame(i))) for i in range(self.trajectory.n_frames)]
        )
        self.volume = float(vol_ang3 * 1e-24)  # Å³ → cm³

        self.scaling_factor = 1.0 / (6.0 * self.kbT * self.volume)
        self._L_tensor = np.zeros((len(self.species), len(self.species)))
        self._L_tensor_self = np.zeros((len(self.species), len(self.species)))
        self.fit_dicts = np.zeros((len(self.species), len(self.species)), dtype=object)
        self.fit_dicts_self = np.zeros((len(self.species), len(self.species)), dtype=object)
        self.correlation_functions: dict[str, dict[str, np.ndarray]] = {}

    def analyze(
        self,
        start_step: int | None = None,
        end_step: int | None = None,
        smoothing: str | None = "best_fit",
    ) -> None:
        """
        Compute L_ij from FFT MSD/cross-MSD and linear fit. Fills L_tensor,
        L_tensor_self, L_tensor_dis; fit_dicts indexed by (i, j) species index.

        Args:
            start_step: int, start time index for fitting
            end_step: int, end time index for fitting
            smoothing: str, type of smoothing to apply when fitting the data.
                "best_fit" (default) brute-forces the best fit interval.
                "blockavg" uses block averaging.
                None does not smooth the data.
        """
        T = self.traj_length
        if start_step is None:
            start_step = max(1, T // 10)
        if end_step is None:
            end_step = (9 * T) // 10

        for s1 in self.species:
            _, pos1 = self.trajectory.positions_for_species(s1)  # (n_frames, n_atoms_s, 3)
            total = calc_Lii(pos1)
            self_ = calc_Lii_self(pos1)
            self.correlation_functions.setdefault(correlation_pair_key(s1, s1), {}).update({"total": total, "self": self_})
            self._L_tensor[self.mapping[s1], self.mapping[s1]], self.fit_dicts[self.mapping[s1], self.mapping[s1]] = fit_data(
                total, self.times, start_step, end_step, smoothing
            )
            self._L_tensor_self[self.mapping[s1], self.mapping[s1]], self.fit_dicts_self[self.mapping[s1], self.mapping[s1]] = fit_data(
                self_, self.times, start_step, end_step, smoothing
            )

            for s2 in self.species:
                if s2 == s1 or self.mapping[s2] < self.mapping[s1]:
                    continue
                _, pos2 = self.trajectory.positions_for_species(s2)  # (n_frames, n_atoms_s2, 3)
                distinct = calc_Lij(pos1, pos2)
                self.correlation_functions.setdefault(correlation_pair_key(s1, s2), {}).update({"distinct": distinct})
                Lij, fd = fit_data(
                    distinct, self.times, start_step, end_step, smoothing
                )
                self._L_tensor[self.mapping[s1], self.mapping[s2]] = self._L_tensor[self.mapping[s2], self.mapping[s1]] = Lij
                self.fit_dicts[self.mapping[s1], self.mapping[s2]] = self.fit_dicts[self.mapping[s2], self.mapping[s1]] = fd

    @property
    def L_tensor(self) -> np.ndarray:
        """Onsager transport coefficients L_ij in 1/cm/s/eV"""
        return self._L_tensor.copy() * self.scaling_factor

    @property
    def L_tensor_self(self) -> np.ndarray:
        """Onsager self-transport coefficients L_ii in 1/cm/s/eV"""
        return self._L_tensor_self.copy() * self.scaling_factor

    @property
    def L_tensor_dis(self) -> np.ndarray:
        """Onsager distinct transport coefficients L_ij in 1/cm/s/eV"""
        return self.L_tensor - self.L_tensor_self

    def get_diffusivity(self, specie: str | int) -> float:
        """Get the Einstein self-diffusion coefficient for a given species in cm^2/s"""
        if isinstance(specie, str):
            specie = self.mapping[specie]
        
        specie_amount = len(self.trajectory.species[self.trajectory.species == self.inv_mapping[specie]])
        return self.L_tensor_self[specie, specie].copy() / specie_amount