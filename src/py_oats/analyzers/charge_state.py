"""
Analyzer for charge state of the system.
Decorates trajectory species with oxidation states (e.g. Li+1, Mn+3) using
user-provided magmoms/charges or a charge decorator. Works case-by-case without
converting the full trajectory to pymatgen unless the chosen path requires it.
"""
from __future__ import annotations

from .base import BaseAnalyzer
from ..reader.trajectory import TrajectoryData
from ..utils.analyzers.charge_state.charge import (
    _element_from_species_string,
    _initial_boundaries,
    species_array_from_elements_and_oxi_states,
)
from ..utils.analyzers.charge_state.decorators import (
    ChargeDecorator,
    NaiveChargeDecorator,
    _trajectory_chunk,
)
from ..utils.analyzers.charge_state.oxidation import assign_oxidation_states


def _unique_elements_from_trajectory(trajectory: TrajectoryData) -> list[str]:
    """Unique element symbols in trajectory.species (strip oxidation state)."""
    seen: set[str] = set()
    out: list[str] = []
    for s in trajectory.species:
        el = _element_from_species_string(str(s))
        if el not in seen:
            seen.add(el)
            out.append(el)
    return out


def _has_user_values(trajectory: TrajectoryData) -> bool:
    """True if trajectory has magmoms, final_magmom, or charge for oxidation assignment."""
    return (
        "magmoms" in trajectory.properties
        or "final_magmom" in trajectory.properties
        or "charge" in trajectory.properties
    )


def _values_at_frame(
    trajectory: TrajectoryData, i: int
) -> tuple[list[float], bool]:
    """
    Per-site values for frame i and whether they are charges (higher => higher oxidation).
    Prefer "charge", else "final_magmom", else "magmoms".
    """
    T, N = trajectory.n_frames, trajectory.n_atoms
    props = trajectory.properties
    if "charge" in props:
        key, higher_means_higher = "charge", True
    elif "final_magmom" in props:
        key, higher_means_higher = "final_magmom", False
    else:
        key, higher_means_higher = "magmoms", False
    arr = props[key]
    if arr.shape == (T, N):
        return arr[i].tolist(), higher_means_higher
    return arr.tolist(), higher_means_higher


class ChargeStateAnalyzer(BaseAnalyzer):
    """
    Analyzer for charge state of the system.
        If TrajectoryData has magmoms, final_magmom, or charge, uses them to assign oxidation
    states at decorate_freq and returns a list of TrajectoryData (length 1 or T//decorate_freq).
    Otherwise uses the charge decorator (Naive: no Structure; BV/CHGNet: convert
    only at sampled frames inside the decorator).
    """

    def __init__(
        self,
        trajectory: TrajectoryData,
        oxidation_states: dict[str, list[int]] | None = None,
        charge_decorator: ChargeDecorator | None = None,
    ) -> None:
        super().__init__(trajectory, name="charge_state_analyzer")
        self.oxidation_states = oxidation_states
        self._charge_decorator = charge_decorator

    def analyze(self, decorate_freq: int = -1) -> list[TrajectoryData]:
        """
        Decorate with oxidation states.

        - If trajectory has magmoms/final_magmom/charge: decorate at decorate_freq
          using assign_oxidation_states; return list of TrajectoryData
          (length 1 if decorate_freq 0 or <0, else T//decorate_freq chunks).
          Metadata tag: decorated_with="user_magmoms" or "user_charge", decorate_freq=...
        - Else: use charge_decorator.decorate(trajectory, decorate_freq).
          NaiveChargeDecorator does not convert to Structure; BV/CHGNet/Other ASE calculators convert
          only at sampled frames inside the decorator.

        Returns:
            list[TrajectoryData]: Decorated trajectory/chunks with species set to charged labels.
        """
        if self.oxidation_states is None:
            elements = _unique_elements_from_trajectory(self.trajectory)
            self.oxidation_states = {e: [0] for e in elements}

        T = self.trajectory.n_frames

        if _has_user_values(self.trajectory):
            return self._analyze_with_user_values(decorate_freq)
        return self._analyze_with_decorator(decorate_freq)

    def _analyze_with_user_values(self, decorate_freq: int) -> list[TrajectoryData]:
        """Use magmoms or charge in trajectory at sampled frame(s); no Structure conversion."""
        T = self.trajectory.n_frames
        _, use_charge = _values_at_frame(self.trajectory, 0)
        decorated_with = "user_charge" if use_charge else "user_magmoms"
        meta_extra = {
            "decorated_with": decorated_with,
            "decorate_freq": decorate_freq,
        }

        def _one_chunk(frame_idx: int):
            values, higher = _values_at_frame(self.trajectory, frame_idx)
            oxi_states = assign_oxidation_states(
                self.trajectory.species,
                self.oxidation_states,
                values,
                adaptive_boundaries=True,
                bounds=_initial_boundaries,
                higher_value_means_higher_oxidation=higher,
            )
            return species_array_from_elements_and_oxi_states(
                self.trajectory.species, oxi_states
            )

        if decorate_freq == 0:
            species = _one_chunk(0)
            return [
                _trajectory_chunk(
                    self.trajectory, 0, T, species, metadata_extra=meta_extra
                )
            ]
        if decorate_freq < 0:
            species = _one_chunk(T - 1)
            return [
                _trajectory_chunk(
                    self.trajectory, 0, T, species, metadata_extra=meta_extra
                )
            ]
        # decorate_freq > 0: chunk by decorate_freq
        result: list[TrajectoryData] = []
        for start in range(0, T, decorate_freq):
            end = min(start + decorate_freq, T)
            species = _one_chunk(start)
            result.append(
                _trajectory_chunk(
                    self.trajectory, start, end, species, metadata_extra=meta_extra
                )
            )
        return result

    def _analyze_with_decorator(self, decorate_freq: int) -> list[TrajectoryData]:
        """Use charge decorator (Naive: no Structure; BV/CHGNet: convert inside decorator)."""
        decorator = self._charge_decorator or NaiveChargeDecorator(
            oxidation_states=self.oxidation_states
        )
        decorator.oxidation_states = self.oxidation_states
        return decorator.decorate(self.trajectory, decorate_freq)
