"""Schema to store transport analysis results from `TransportAnalyzer`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Composition

from py_oats.analyzers.transport import TransportAnalyzer


class TransportDoc(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    """
    Serializable schema for Onsager transport coefficients and related metadata.

    Args:
        composition: Composition of the system.
        reduced_formula: Reduced chemical formula string.
        species: Species labels in the system (ordering matches tensors).
        L_tensor: Onsager transport coefficients \(L_{ij}\) in 1/cm/s/eV.
        L_tensor_self: Self-transport coefficients \(L_{ii}^\text{self}\) in 1/cm/s/eV.
        L_tensor_dis: Distinct transport coefficients \(L_{ij}^{\\text{dis}}\) in 1/cm/s/eV.
        diffusivity: Per-species Einstein self-diffusion coefficients in cm²/s.
        temperature: Temperature of the system in K.
        volume: Volume of the simulation cell in Å³.
        num_atoms: Number of atoms in the system.
        mapping: Mapping of species labels to indices in stored tensors.
        correlation_functions: Correlation functions keyed by
            \((i, j)\) index pairs.
        times: Time grid corresponding to correlation functions (default in fs).
        time_step: Time step of the simulation in fs.
        step_skip: Step skip of the simulation.
    """

    composition: Optional[Composition | str] = Field(None, description="composition")
    reduced_formula: Optional[str] = Field(None, description="reduced_formula")
    species: Optional[List[str]] = Field(None, description="species")
    L_tensor: Optional[np.ndarray] = Field(None, description="L_tensor")
    L_tensor_self: Optional[np.ndarray] = Field(None, description="L_tensor_self")
    L_tensor_dis: Optional[np.ndarray] = Field(None, description="L_tensor_dis")
    diffusivity: Optional[Dict[str, float]] = Field(
        None, description="per-species self-diffusion coefficients (cm^2/s)"
    )
    temperature: Optional[float] = Field(None, description="temperature (K)")
    volume: Optional[float] = Field(None, description="volume (Å^3)")
    num_atoms: Optional[int] = Field(None, description="number of atoms")
    mapping: Optional[Dict[str, int]] = Field(
        None, description="mapping of species labels to tensor indices"
    )
    correlation_functions: Optional[Any] = Field(
        None, description="correlation functions keyed by (species_i, species_j) indices"
    )
    fit_dicts: Optional[Any] = Field(
        None, description="fit dictionaries (often 2D object arrays)"
    )
    fit_dicts_self: Optional[Any] = Field(
        None, description="self fit dictionaries (often 2D object arrays)"
    )
    times: Optional[List[float]] = Field(None, description="time grid (fs)")
    time_step: Optional[float] = Field(None, description="time_step (fs)")
    step_skip: Optional[int] = Field(None, description="step_skip")

    def get_transport_coefficient(
        self, species: List[str] | str, self_transport: bool = False
    ) -> float:
        """
        Get a transport coefficient for a given species or species pair.

        Args:
            species: Single species (str) or a pair [species_i, species_j].
            self_transport: If True, use `L_tensor_self` for diagonal terms.
        """
        if self.mapping is None or self.L_tensor is None:
            raise ValueError("Transport tensors and mapping must be set.")

        if isinstance(species, list) and len(species) == 2:
            if not all(isinstance(s, str) for s in species):
                raise ValueError("species must be a list of str")
            return self.L_tensor[self.mapping[species[0]], self.mapping[species[1]]]

        if isinstance(species, str):
            if self.mapping is None or species not in self.mapping:
                raise KeyError(f"Species {species} not in mapping")
            return self.L_tensor_self[self.mapping[species], self.mapping[species]] if self_transport else self.L_tensor[self.mapping[species], self.mapping[species]]

        raise ValueError("species must be a list of two items, or a single str")

    def as_dict(self) -> dict:
        """Return a dictionary representation (for JSON/YAML storage)."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, d: dict) -> "TransportDoc":
        """Construct `TransportDoc` from a dictionary."""
        return cls.model_validate(d)

    @classmethod
    def from_analyzer(cls, analyzer: TransportAnalyzer) -> "TransportDoc":
        """Build a `TransportDoc` from a `TransportAnalyzer`."""
        td = cls()
        td.temperature = analyzer.temperature
        td.species = list(analyzer.species)
        td.mapping = dict(analyzer.mapping)
        td.L_tensor = analyzer.L_tensor
        td.L_tensor_self = analyzer.L_tensor_self
        td.L_tensor_dis = analyzer.L_tensor_dis
        td.volume = analyzer.volume
        td.correlation_functions = dict(analyzer.correlation_functions)
        td.fit_dicts = analyzer.fit_dicts
        td.fit_dicts_self = analyzer.fit_dicts_self
        td.times = analyzer.times.tolist()
        td.time_step = analyzer.time_step
        td.step_skip = analyzer.step_skip
        # Derive composition and reduced formula from trajectory species if possible.
        td.composition = analyzer.trajectory.composition

        td.num_atoms = analyzer.trajectory.n_atoms

        # Diffusivity per species using self-transport coefficients.
        diffusivity: Dict[str, float] = {}
        for label in analyzer.species:
            try:
                diffusivity[label] = analyzer.get_diffusivity(label)
            except Exception:
                continue
        td.diffusivity = diffusivity or None
        return td

