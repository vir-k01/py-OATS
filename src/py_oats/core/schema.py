"""Useful schema to store OATS analysis results"""

from pydantic import Field
from monty.json import MSONable
from typing import Optional, Any, List, Dict, Tuple
from pymatgen.core import Composition
from py_oats.core.analyzer import OnsagerTransportAnalyzer
        
class TransportDoc(MSONable):
    """
    Schema to store transport coefficients in a serializable format
    
    Args:
        composition: Composition, composition of the system
        species: List[str], species in the system
        L_tensor: np.ndarray, transport coefficients in cm^2/s/eV
        L_tensor_self: np.ndarray, self transport coefficients in cm^2/s/eV
        L_tensor_dis: np.ndarray, distinct transport coefficients in cm^2/s/eV
        diffusivity: Dict[str, float], diffusivity of the species in cm^2/s
        temperature: float, temperature of the system in K
        volume: float, volume of the system in A^3
        num_atoms: int, number of atoms in the system
        mapping: Dict[str, int], mapping of species to indices in the stored transport tensors.
        msds: Dict[(int, int), list[float]], mean square displacement (generally the correlation function of the displacement)
            of the species in cm^2/s
        times: list[float], times of the correlation function (default in fs)
        time_step: float, time step of the simulation in fs (default 2 fs)
        step_skip: int, step skip of the simulation (default 1)
    
    Returns:
        TransportDoc, schema to store transport coefficients in a serializable format
    """
    
    composition: Optional[Composition] = Field(None, description='composition')
    reduced_formula: Optional[str] = Field(None, description='reduced_formula')
    species: Optional[List[str]] = Field(None, description='species')
    L_tensor: Optional[Any] = Field(None, description='L_tensor')
    L_tensor_self: Optional[Any] = Field(None, description='L_tensor_self')
    L_tensor_dis: Optional[Any] = Field(None, description='L_tensor_dis')
    diffusivity: Optional[Any] = Field(None, description='diffusivity')
    temperature: Optional[float] = Field(None, description='temperature')
    volume: Optional[float] = Field(None, description='volume')
    num_atoms: Optional[int] = Field(None, description='num_atoms')
    mapping: Optional[Dict[str, int]] = Field(None, description='mapping')
    msds: Optional[Dict[Tuple[int, int], list[float]]] = Field(None, description='msds')
    times: Optional[list[float]] = Field(None, description='times')
    time_step: Optional[float] = Field(None, description='time_step')
    step_skip: Optional[int] = Field(None, description='step_skip')
    
    def get_transport_coefficient(self, species : list[str|int] | int | str, self_transport: bool = False) -> float:
        """
        Get the transport coefficient for a given set of species. 
        :param species: list[str|int] | int | str, species for which to get the transport coefficient
        :param self_transport: bool, whether to get the self transport coefficient
        :return: float, transport coefficient
        """
        if isinstance(species, list) and len(species) == 2:
            if not all(isinstance(s, (int, str)) for s in species):
                raise ValueError("species must be a list of int, str")
            species = [self.mapping[s] for s in species if isinstance(s, str)]
            return self.L_tensor[species[0], species[1]]
        elif isinstance(species, (int, str)):
            species = self.mapping[species] if isinstance(species, str) else species
            return self.L_tensor_self[species, species] if self_transport else self.L_tensor[species, species]
        else:
            raise ValueError("species must be a list of int, str")
    
    @classmethod
    def from_analyzer(cls, analyzer: OnsagerTransportAnalyzer):
        td = cls()
        td.temperature = analyzer.temperature
        td.reduced_formula = analyzer.composition
        td.composition = analyzer.first_structure.composition
        td.species = analyzer.species
        td.L_tensor = analyzer.L_tensor
        td.L_tensor_self = analyzer.L_tensor_self
        td.L_tensor_dis = analyzer.L_tensor_dis
        td.diffusivity = analyzer.diffusivity
        td.mapping = analyzer.mapping
        td.volume = analyzer.volume
        td.num_atoms = analyzer.first_structure.num_sites
        td.msds = analyzer.msds
        td.times = analyzer.times
        td.time_step = analyzer.time_step
        td.step_skip = analyzer.step_skip
        return td