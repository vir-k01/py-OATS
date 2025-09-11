from pymatgen.core import Structure, Species
from typing import Dict, List, Union
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.util.typing import PathLike
from chgnet.model import CHGNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from .charge import _initial_boundaries, get_specie_with_multiple_ox_states
from .oxidation import assign_oxidation_states_by_magmoms

class ChargeDecorator():
    def __init__(self,
                 oxidation_states: Dict[str, List[int]]):
        """
        Args:
            structure: pymatgen Structure object
            oxidation_states: Dictionary mapping element symbols to lists of oxidation states.
                            e.g., {"Li": [1], "Ti": [2, 4], "O": [-2]}
        """
        self.oxidation_states = oxidation_states
    
    def decorate_structure(self, structure: Structure) -> Structure:
        pass

class NaiveChargeDecorator(ChargeDecorator):
    """
    Naive charge decorator that assigns oxidation states to each site in the structure.
    """
    def decorate_structure(self, structure: Structure) -> Structure:
        single_oxidation_states = {k: np.mean(v) for k, v in self.oxidation_states.items()}
        return structure.add_oxidation_state_by_element(single_oxidation_states)

class BVChargeDecorator(ChargeDecorator):
    """
    Decorates a pymatgen Structure object with oxidation states using bond valence analysis.
    """
    def decorate_structure(self, structure: Structure) -> Structure:
        """
        Decorates a pymatgen Structure object with oxidation states.
        
        Returns:
            Structure: New structure with Species objects containing oxidation states
        """
        bv_analyzer = BVAnalyzer()
        bv_structure = bv_analyzer.get_oxi_state_decorated_structure(structure)
        return bv_structure


class CHGNetChargeDecorator(ChargeDecorator):
    """
    Decorates a pymatgen Structure object with oxidation states using CHGNet. 
    If no model path is provided, pretrained CHGnet will be used.
    """
    
    def __init__(self, model_path: str | PathLike | None = None, 
                 oxidation_states: Dict[str, List[int]] = None,
                 adaptive_boundaries: bool = False):
        super().__init__(oxidation_states)
        self.model_path = model_path
        self.adaptive_boundaries = adaptive_boundaries
        if model_path:
            self.model = CHGNetCalculator.from_file(self.model_path)
        else:
            self.model = CHGNetCalculator()
    
    def decorate_structure(self, structure: Structure) -> Structure:
        # Check if structure already has magnetic moments
        if structure.site_properties.get('magmoms', None) is not None:
            magmoms = structure.site_properties['magmoms']
            oxi_states = assign_oxidation_states_by_magmoms(
                structure=structure,
                oxidation_states=self.oxidation_states,
                magmoms=magmoms,
                adaptive_boundaries=self.adaptive_boundaries,
                bounds=_initial_boundaries,
            )
            structure.add_oxidation_state_by_site(oxi_states)
        else:
            # Use CHGNet to get magnetic moments
            ase_atoms = AseAtomsAdaptor().get_atoms(structure)
            ase_atoms.calc = self.model
            magmoms = ase_atoms.get_magnetic_moments()
            oxi_states = assign_oxidation_states_by_magmoms(
                structure=structure,
                oxidation_states=self.oxidation_states,
                magmoms=magmoms,
                adaptive_boundaries=self.adaptive_boundaries,
                bounds=_initial_boundaries,
            )
            structure.add_oxidation_state_by_site(oxi_states)
        return structure

