from typing import Dict, List
import numpy as np
from pymatgen.core import Structure


def get_oxi_state_from_magmom(magmom: float, element: str, bounds: dict) -> int:
    """
    Get the oxidation state from the magnetic moment.
    Assumes monotonic relation: higher oxidation state -> lower magnetic moment.

    Supports two modes:
    - Single-threshold mode (recommended): bounds[element] provides a single threshold under the
      higher oxidation state key. If magmom <= threshold, assign higher state; else lower state.
    - Legacy multi-threshold mode: bounds[element] maps ox_state -> threshold. The function will
      iterate thresholds in ascending order and return the first ox_state whose threshold is met.
    """
    element_bounds = bounds.get(element, {})
    if not element_bounds:
        raise ValueError(f"No bounds provided for element {element}.")

    # If only one threshold is provided, interpret it as the split between high and low states
    if len(element_bounds) == 1:
        (ox_state_key, threshold_value), = element_bounds.items()
        return ox_state_key if magmom <= threshold_value else ox_state_key - 1

    # Multi-threshold legacy path
    items = list(element_bounds.items())
    items.sort(key=lambda x: x[1])
    for oxi_state, threshold in items:
        if magmom <= threshold:
            return oxi_state
    ox_states = [ox for ox, _ in items]
    return min(ox_states)


def _solve_boundary_for_charge_neutrality(
    structure: Structure,
    element: str,
    magmoms_species: List[float],
    oxidation_states: Dict[str, List[int]]
) -> float:
    if element not in oxidation_states or len(oxidation_states[element]) < 2:
        raise ValueError("Target element must have at least two oxidation states to solve a boundary.")

    element_states = sorted(oxidation_states[element], reverse=True)
    ox_high = element_states[0]
    ox_low = element_states[-1]

    composition = structure.composition.get_el_amt_dict()
    base_charge = 0
    for other_elem, count in composition.items():
        if other_elem == element:
            continue
        states = oxidation_states.get(other_elem, [])
        if len(states) == 0:
            continue
        fixed_state = states[0]
        base_charge += count * fixed_state

    magmoms_sorted = sorted(magmoms_species)
    num_sites_element = len(magmoms_sorted)

    denom = (ox_high - ox_low)
    if denom == 0:
        return float(np.median(magmoms_sorted)) if num_sites_element > 0 else 0.0
    rhs = -base_charge - num_sites_element * ox_low
    n_high_required = int(np.ceil(rhs / denom))

    n_high_required = max(0, min(num_sites_element, n_high_required))

    if num_sites_element == 0:
        return 0.0
    if n_high_required == 0:
        return magmoms_sorted[0] - 1e-6
    if n_high_required == num_sites_element:
        return magmoms_sorted[-1] + 1e-6

    lower = magmoms_sorted[n_high_required - 1]
    upper = magmoms_sorted[n_high_required]
    return 0.5 * (lower + upper)


def _adapt_boundaries(
    structure: Structure,
    bounds: dict,
    element: str,
    magmoms_species: List[float],
    oxidation_states: Dict[str, List[int]]
) -> dict:
    new_boundary = _solve_boundary_for_charge_neutrality(
        structure=structure,
        element=element,
        magmoms_species=magmoms_species,
        oxidation_states=oxidation_states,
    )

    if element not in bounds:
        bounds[element] = {}

    element_states = sorted(oxidation_states[element], reverse=True)
    bounds[element][element_states[0]] = new_boundary
    return bounds


def _solve_boundaries_for_multiple_elements(
    structure: Structure,
    elements: List[str],
    magmoms_by_element: Dict[str, List[float]],
    oxidation_states: Dict[str, List[int]],
) -> Dict[str, float]:
    composition = structure.composition.get_el_amt_dict()

    base_charge = 0
    for elem, count in composition.items():
        if elem in elements:
            continue
        states = oxidation_states.get(elem, [])
        if len(states) == 0:
            continue
        fixed_state = states[0]
        base_charge += count * fixed_state

    per_elem_params = {}
    total_low_charge = 0
    for elem in elements:
        states_sorted = sorted(oxidation_states[elem], reverse=True)
        ox_high = states_sorted[0]
        ox_low = states_sorted[-1]
        mag_sorted = sorted(magmoms_by_element.get(elem, []))
        N = len(mag_sorted)
        per_elem_params[elem] = {
            "ox_high": ox_high,
            "ox_low": ox_low,
            "delta": ox_high - ox_low,
            "mag_sorted": mag_sorted,
            "N": N,
        }
        total_low_charge += N * ox_low

    target = - base_charge - total_low_charge

    dp = {0: {elem: 0 for elem in elements}}

    for elem in elements:
        params = per_elem_params[elem]
        delta = params["delta"]
        N = params["N"]
        if delta == 0 or N == 0:
            continue
        new_dp = dict(dp)
        for current_sum, counts in dp.items():
            for k in range(1, N + 1):
                s = current_sum + k * delta
                if s not in new_dp:
                    new_counts = dict(counts)
                    new_counts[elem] = k
                    new_dp[s] = new_counts
        dp = new_dp

    if target in dp:
        chosen_counts = dp[target]
    else:
        if len(dp) == 0:
            chosen_counts = {elem: 0 for elem in elements}
        else:
            best_sum = min(dp.keys(), key=lambda s: abs(s - target))
            chosen_counts = dp[best_sum]

    thresholds: Dict[str, float] = {}
    for elem in elements:
        params = per_elem_params[elem]
        mag_sorted = params["mag_sorted"]
        N = params["N"]
        n_high = int(chosen_counts.get(elem, 0))
        if N == 0:
            thresholds[elem] = 0.0
            continue
        if n_high <= 0:
            thresholds[elem] = mag_sorted[0] - 1e-6
        elif n_high >= N:
            thresholds[elem] = mag_sorted[-1] + 1e-6
        else:
            thresholds[elem] = 0.5 * (mag_sorted[n_high - 1] + mag_sorted[n_high])

    return thresholds


def assign_oxidation_states_by_magmoms(
    structure: Structure,
    oxidation_states: Dict[str, List[int]],
    magmoms: List[float],
    adaptive_boundaries: bool = True,
    bounds: Dict[str, Dict[int, float]] | None = None,
) -> List[int]:
    default_state_by_elem = {elem: states[0] for elem, states in oxidation_states.items()}
    oxi_states: List[int] = [default_state_by_elem[site.specie.symbol] for site in structure]

    elements_in_structure = set([s.symbol for s in structure.species])
    multi_state_elements = [
        elem for elem, states in oxidation_states.items()
        if len(states) > 1 and elem in elements_in_structure
    ]
    if len(multi_state_elements) == 0:
        return oxi_states

    if adaptive_boundaries:
        magmoms_by_elem: Dict[str, List[float]] = {}
        indices_by_elem: Dict[str, List[int]] = {}
        for elem in multi_state_elements:
            mask = [s.symbol == elem for s in structure.species]
            magmoms_by_elem[elem] = [magmoms[i] for i, is_elem in enumerate(mask) if is_elem]
            indices_by_elem[elem] = [i for i, is_elem in enumerate(mask) if is_elem]

        thresholds = _solve_boundaries_for_multiple_elements(
            structure=structure,
            elements=multi_state_elements,
            magmoms_by_element=magmoms_by_elem,
            oxidation_states=oxidation_states,
        )

        for elem in multi_state_elements:
            states_sorted = sorted(oxidation_states[elem], reverse=True)
            ox_high = states_sorted[0]
            ox_low = states_sorted[-1]
            threshold = thresholds.get(elem, None)
            if threshold is None:
                continue
            for idx in indices_by_elem[elem]:
                oxi_states[idx] = ox_high if magmoms[idx] <= threshold else ox_low
        return oxi_states
    else:
        if bounds is None:
            raise ValueError("bounds must be provided when adaptive_boundaries is False")
        for elem in multi_state_elements:
            mask = [s.symbol == elem for s in structure.species]
            indices = [i for i, is_elem in enumerate(mask) if is_elem]
            for idx in indices:
                oxi_states[idx] = get_oxi_state_from_magmom(magmoms[idx], elem, bounds)
        return oxi_states


