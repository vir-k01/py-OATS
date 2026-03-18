"""Charge decorators for assigning oxidation states to structures."""
from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
from pymatgen.core import Structure

from .charge import (
    _element_from_species_string,
    _initial_boundaries,
    species_array_from_elements_and_oxi_states,
)
from .oxidation import assign_oxidation_states
from ase.calculators.calculator import Calculator

if TYPE_CHECKING:
    from ....reader.trajectory import TrajectoryData


def _trajectory_chunk(
    trajectory: "TrajectoryData",
    start: int,
    end: int,
    species_labels: np.ndarray,
    metadata_extra: dict[str, Any] | None = None,
) -> "TrajectoryData":
    """Build TrajectoryData for frames [start:end] with given species (length N)."""
    from ....reader.trajectory import TrajectoryData

    T, N = trajectory.n_frames, trajectory.n_atoms
    positions = trajectory.positions[start:end]
    species = np.asarray(species_labels, dtype=object)
    if trajectory.lattices.ndim == 2:
        lattices = trajectory.lattices
    else:
        lattices = trajectory.lattices[start:end]
    properties = {}
    for key, arr in trajectory.properties.items():
        if arr.shape == (T, N):
            properties[key] = arr[start:end]
        elif arr.shape == (N,):
            properties[key] = arr.copy()
        elif arr.shape == (T,):
            properties[key] = arr[start:end].copy()
    meta = dict(trajectory.metadata)
    if metadata_extra:
        meta.update(metadata_extra)
    return TrajectoryData(
        positions=positions,
        species=species,
        lattices=lattices,
        properties=properties,
        metadata=meta,
    )


class ChargeDecorator:
    """Base class for decorators that assign oxidation states to structures."""

    def __init__(self, oxidation_states: Dict[str, List[int]]):
        """
        Args:
            oxidation_states: Dictionary mapping element symbols to lists of oxidation states.
                e.g., {"Li": [1], "Ti": [2, 4], "O": [-2]}
        """
        self.oxidation_states = oxidation_states

    def decorate(
        self, trajectory: "TrajectoryData", decorate_freq: int
    ) -> list["TrajectoryData"]:
        """
        Decorate the trajectory with oxidation states.

        Args:
            trajectory: TrajectoryData to decorate.
            decorate_freq: How often to sample (e.g. every N frames); decorator may ignore.
        Returns:
            list[TrajectoryData]: List of decorated trajectory chunks.
        """
        raise NotImplementedError


class NaiveChargeDecorator(ChargeDecorator):
    """
    Naive charge decorator: replace trajectory.species with element+oxidation
    using mean oxidation per element. No Structure conversion.
    """

    def decorate(
        self, trajectory: "TrajectoryData", decorate_freq: int
    ) -> list["TrajectoryData"]:
        """Single chunk; decorate_freq ignored."""
        oxi_states = []
        for s in trajectory.species:
            oxi_states.append(np.mean(self.oxidation_states.get(_element_from_species_string(str(s)), [0])))
        new_species = species_array_from_elements_and_oxi_states(trajectory.species, oxi_states)
        return [_trajectory_chunk(trajectory, 0, trajectory.n_frames, new_species, metadata_extra={"decorated_with": "NaiveChargeDecorator"})]


class BVChargeDecorator(ChargeDecorator):
    """
    Decorates a pymatgen Structure with oxidation states using bond valence analysis.
    Requires pymatgen.analysis.bond_valence.BVAnalyzer.
    """

    def __init__(
        self,
        oxidation_states: Dict[str, List[int]],
        bv_kwargs: Dict[str, Any] | None = None,
    ):
        super().__init__(oxidation_states)
        from pymatgen.analysis.bond_valence import BVAnalyzer
        self.bv_analyzer = BVAnalyzer(**(bv_kwargs or {}))

    def decorate(self, trajectory: "TrajectoryData", decorate_freq: int) -> list["TrajectoryData"]:
        decorated_trajectory = []
        for i in range(0, trajectory.n_frames, decorate_freq):
            meta_extra = {"decorated_with": "BVChargeDecorator", "decorate_freq": decorate_freq, "start_frame": i, "end_frame": min(i + decorate_freq, trajectory.n_frames)}
            structure = trajectory.structure_at_frame(i)
            bv_structure = self.bv_analyzer.get_oxi_state_decorated_structure(structure)
            oxi_states = [site.specie.oxi_state for site in bv_structure]
            new_species = species_array_from_elements_and_oxi_states(trajectory.species, oxi_states)
            decorated_trajectory.append(
                _trajectory_chunk(trajectory, i, min(i + decorate_freq, trajectory.n_frames), new_species, metadata_extra=meta_extra)
            )
        return decorated_trajectory


class CHGNetChargeDecorator(ChargeDecorator):
    """
    Decorates a TrajectoryData with oxidation states using CHGNet-predicted
    magnetic moments and adaptive boundaries. Requires chgnet from the matgl package.
    """

    def __init__(
        self,
        model_path: str | None = None,
        oxidation_states: Dict[str, List[int]] | None = None,
        adaptive_boundaries: bool = True,
    ):
        super().__init__(oxidation_states or {})
        self.model_path = model_path
        self.adaptive_boundaries = adaptive_boundaries
        if model_path:
            from chgnet.model import CHGNetCalculator
            self.model = CHGNetCalculator.from_file(self.model_path)
        else:
            from chgnet.model import CHGNet, CHGNetCalculator
            model = CHGNet(model_name="r2scan")
            self.model = CHGNetCalculator(model=model)

    def decorate(self, trajectory: "TrajectoryData", decorate_freq: int) -> list["TrajectoryData"]:
        decorated_trajectory = []
        for i in range(0, trajectory.n_frames, decorate_freq):
            meta_extra = {"decorated_with": "CHGNetChargeDecorator", "decorate_freq": decorate_freq, "start_frame": i, "end_frame": min(i + decorate_freq, trajectory.n_frames)}
            atoms = trajectory.atoms_at_frame(i)
            atoms.calc = self.model
            magmoms = list(atoms.get_magnetic_moments())
            oxi_states = assign_oxidation_states(
                trajectory.species,
                self.oxidation_states,
                magmoms_or_charges=magmoms,
                adaptive_boundaries=self.adaptive_boundaries,
                bounds=_initial_boundaries,
            )
            new_species = species_array_from_elements_and_oxi_states(trajectory.species, oxi_states)
            decorated_trajectory.append(
                _trajectory_chunk(trajectory, i, min(i + decorate_freq, trajectory.n_frames), new_species, metadata_extra=meta_extra)
            )
        return decorated_trajectory

class AseCalcChargeDecorator(ChargeDecorator):
    """
    Decorates a TrajectoryData with charges using a user-provided ASE calculator.
    Requires a ase.calculators.base.Calculator to be provided that can be used to calculate charges.
    """

    def __init__(
        self,
        calculator: Calculator,
        oxidation_states: Dict[str, List[int]] | None = None,
        adaptive_boundaries: bool = True,
    ):
        super().__init__(oxidation_states or {})
        self.calculator = calculator
        self.adaptive_boundaries = adaptive_boundaries

    def decorate(self, trajectory: "TrajectoryData", decorate_freq: int) -> list["TrajectoryData"]:
        decorated_trajectory = []
        for i in range(0, trajectory.n_frames, decorate_freq):
            meta_extra = {"decorated_with": "AseCalcChargeDecorator", "decorate_freq": decorate_freq, "start_frame": i, "end_frame": min(i + decorate_freq, trajectory.n_frames)}
            atoms = trajectory.atoms_at_frame(i)
            atoms.calc = self.calculator
            charges = list(atoms.get_charges())
            oxi_states = assign_oxidation_states(
                trajectory.species,
                self.oxidation_states,
                magmoms_or_charges=charges,
                adaptive_boundaries=self.adaptive_boundaries,
                bounds=_initial_boundaries,
                higher_value_means_higher_oxidation=True,
            )
            new_species = species_array_from_elements_and_oxi_states(trajectory.species, oxi_states)
            decorated_trajectory.append(
                _trajectory_chunk(trajectory, i, min(i + decorate_freq, trajectory.n_frames), new_species, metadata_extra=meta_extra)
            )
        return decorated_trajectory