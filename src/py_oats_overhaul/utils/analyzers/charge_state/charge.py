"""Charge/oxidation state utilities for trajectory decoration."""
from typing import Dict, List

import numpy as np
from pymatgen.core import Structure


_initial_boundaries = {
    "Mn": {4: 3.08, 3: 4.2, 2: 6},
}


def _element_from_species_string(s: str) -> str:
    """Extract element symbol from species string (e.g. 'Li+1' -> 'Li', 'O-2' -> 'O')."""
    s = str(s)
    if "+" in s:
        return s.split("+")[0]
    if "-" in s and s[0] != "-":
        return s.split("-")[0]
    return s


def species_string_from_element_and_oxidation(element: str, oxidation: float) -> str:
    """
    Format a single species label from element and oxidation state.
    Examples: Li+1, Mn+3.5, O-2, Li0. Reused everywhere to avoid Structure round-trip.
    """
    el = _element_from_species_string(element)
    ox = float(oxidation)
    if ox > 0:
        return f"{el}+{int(ox) if ox == int(ox) else ox}"
    return f"{el}{int(ox) if ox == int(ox) else ox}"


def species_array_from_elements_and_oxi_states(
    species_array: np.ndarray,
    oxi_states: List[int],
) -> np.ndarray:
    """
    Build (n_atoms,) species strings from species_array and oxi_states.
    """
    n = len(oxi_states)
    if len(species_array) != n:
        raise ValueError("species_array and oxi_states must have same length")
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = species_string_from_element_and_oxidation(species_array[i], oxi_states[i])
    return out


def decorate_species_naive(
    species_array: np.ndarray,
    oxidation_states: Dict[str, List[int]],
) -> np.ndarray:
    """Assign mean oxidation per element to each site; return decorated species strings."""
    n = len(species_array)
    oxi_list = []
    for i in range(n):
        el = _element_from_species_string(str(species_array[i]))
        states = oxidation_states.get(el, [0])
        oxi_list.append(float(np.mean(states)))
    return species_array_from_elements_and_oxi_states(species_array, oxi_list)


def unique_species_from_oxidation_states(oxidation_states: Dict[str, List[int]]) -> List[str]:
    """List of unique species strings from oxidation states dictionary."""
    unique_species = []
    for element, ox_states in oxidation_states.items():
        for ox_state in ox_states:
            if ox_state > 0:
                unique_species.append(f"{element}+{ox_state}")
            else:
                unique_species.append(f"{element}{ox_state}")
    return unique_species


def get_specie_with_multiple_ox_states(
    oxidation_states: Dict[str, List[int]],
) -> List[str]:
    """List of elements that have more than one oxidation state."""
    return [e for e, states in oxidation_states.items() if len(states) > 1]