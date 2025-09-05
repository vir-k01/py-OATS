from typing import Dict, List
import numpy as np

_initial_boundaries = {"Mn" : {4 : 3.08,
                                   3 : 4.2,
                                   2 : 6}}


def get_unique_charged_species(oxidation_states: Dict[str, List[int]]) -> List[str]:
    """
    Get a list of unique charged species from oxidation states dictionary.
    
    Args:
        oxidation_states: Dictionary mapping element symbols to lists of oxidation states
    
    Returns:
        List[str]: List of unique charged species (e.g., ["Li+1", "Ti+2", "Ti+4", "O-2"])
    """
    unique_species = []
    for element, ox_states in oxidation_states.items():
        for ox_state in ox_states:
            if ox_state > 0:
                unique_species.append(f"{element}+{ox_state}")
            else:
                unique_species.append(f"{element}{ox_state}")
    return unique_species 


def get_specie_with_multiple_ox_states(oxidation_states: Dict[str, List[int]]) -> List[str]:
    """
    Get a list of unique charged species from oxidation states dictionary.
    More than one oxidation state will be considered as having a multiple oxidation states.
    Args:
        oxidation_states: Dictionary mapping element symbols to lists of oxidation states
    
    Returns:
        List[str]: List of species with multiple oxidation states
    """
    specie_with_multiple_ox_states = []
    for element, ox_states in oxidation_states.items():
        if len(ox_states) > 1:
            specie_with_multiple_ox_states.append(element)
    return specie_with_multiple_ox_states

def get_oxi_state_from_magmom(magmom: float, element: str, bounds: dict) -> int:
    """
    Get the oxidation state from the magnetic moment.
    Higher oxidation states have lower magnetic moments.
    """
    
    bounds = list(bounds[element].items())
    # Sort by magnetic moment value (ascending)
    bounds.sort(key=lambda x: x[1])
    
    # For magnetic moments, higher oxidation states have lower moments
    # So we check from highest to lowest oxidation state
    for oxi_state, bound in (bounds):
        if magmom <= bound:
            return oxi_state
    
    # If no match found, return the lowest oxidation state
    return bounds[0][0]