"""Oxidation state assignment from magnetic moments. Uses species arrays only (no Structure)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _element_from_species_string(s: str) -> str:
    """Extract element symbol (e.g. 'Li+1' -> 'Li', 'O-2' -> 'O')."""
    s = str(s)
    if "+" in s:
        return s.split("+")[0]
    if "-" in s and s[0] != "-":
        return s.split("-")[0]
    return s


def composition_from_species_array(species_array: np.ndarray) -> Dict[str, int]:
    """Element -> count from (n_sites,) species array. Strips oxidation from labels."""
    comp: Dict[str, int] = {}
    for s in species_array:
        el = _element_from_species_string(str(s))
        comp[el] = comp.get(el, 0) + 1
    return comp


def get_oxi_state_from_value(value: float, element: str, bounds: dict) -> int:
    """
    Get the oxidation state from the value.
    Assumes monotonic relation: higher oxidation state -> lower value for magnetic moments, higher value for charges.
    """
    element_bounds = bounds.get(element, {})
    if not element_bounds:
        raise ValueError(f"No bounds provided for element {element}.")
    if len(element_bounds) == 1:
        (ox_state_key, threshold_value), = element_bounds.items()
        return ox_state_key if value <= threshold_value else ox_state_key - 1
    items = list(element_bounds.items())
    items.sort(key=lambda x: x[1])
    for oxi_state, threshold in items:
        if value <= threshold:
            return oxi_state
    return min(ox for ox, _ in items)


def _solve_boundary_for_charge_neutrality(
    composition: Dict[str, int],
    element: str,
    values: List[float],
    oxidation_states: Dict[str, List[int]],
) -> float:
    if element not in oxidation_states or len(oxidation_states[element]) < 2:
        raise ValueError(
            "Target element must have at least two oxidation states to solve a boundary."
        )
    element_states = sorted(oxidation_states[element], reverse=True)
    ox_high, ox_low = element_states[0], element_states[-1]
    base_charge = 0
    for other_elem, count in composition.items():
        if other_elem == element:
            continue
        states = oxidation_states.get(other_elem, [])
        if len(states) == 0:
            continue
        base_charge += count * states[0]
    values_sorted = sorted(values)
    num_sites_element = len(values_sorted)
    denom = ox_high - ox_low
    if denom == 0:
        return float(np.median(values_sorted)) if num_sites_element > 0 else 0.0
    rhs = -base_charge - num_sites_element * ox_low
    n_high_required = max(0, min(num_sites_element, int(np.ceil(rhs / denom))))
    if num_sites_element == 0:
        return 0.0
    if n_high_required == 0:
        return values_sorted[0] - 1e-6
    if n_high_required == num_sites_element:
        return values_sorted[-1] + 1e-6
    return 0.5 * (values_sorted[n_high_required - 1] + values_sorted[n_high_required])


def _solve_boundaries_for_multiple_elements(
    composition: Dict[str, int],
    elements: List[str],
    values_by_element: Dict[str, List[float]],
    oxidation_states: Dict[str, List[int]],
) -> Dict[str, float]:
    base_charge = 0
    for elem, count in composition.items():
        if elem in elements:
            continue
        states = oxidation_states.get(elem, [])
        if len(states) > 0:
            base_charge += count * states[0]
    per_elem_params = {}
    total_low_charge = 0
    for elem in elements:
        states_sorted = sorted(oxidation_states[elem], reverse=True)
        ox_high, ox_low = states_sorted[0], states_sorted[-1]
        values_sorted = sorted(values_by_element.get(elem, []))
        N = len(values_sorted)
        per_elem_params[elem] = {
            "ox_high": ox_high,
            "ox_low": ox_low,
            "delta": ox_high - ox_low,
            "values_sorted": values_sorted,
            "N": N,
        }
        total_low_charge += N * ox_low
    target = -base_charge - total_low_charge
    dp = {0: {elem: 0 for elem in elements}}
    for elem in elements:
        params = per_elem_params[elem]
        delta, N = params["delta"], params["N"]
        if delta == 0 or N == 0:
            continue
        new_dp = dict(dp)
        for current_sum, counts in dp.items():
            for k in range(1, N + 1):
                s = current_sum + k * delta
                if s not in new_dp:
                    new_dp[s] = {**counts, elem: k}
        dp = new_dp
    if target in dp:
        chosen_counts = dp[target]
    else:
        chosen_counts = (
            dp[min(dp.keys(), key=lambda s: abs(s - target))]
            if dp
            else {e: 0 for e in elements}
        )
    thresholds: Dict[str, float] = {}
    for elem in elements:
        values_sorted = per_elem_params[elem]["values_sorted"]
        N = per_elem_params[elem]["N"]
        n_high = int(chosen_counts.get(elem, 0))
        if N == 0:
            thresholds[elem] = 0.0
        elif n_high <= 0:
            thresholds[elem] = values_sorted[0] - 1e-6
        elif n_high >= N:
            thresholds[elem] = values_sorted[-1] + 1e-6
        else:
            thresholds[elem] = 0.5 * (values_sorted[n_high - 1] + values_sorted[n_high])
    return thresholds


def assign_oxidation_states(
    species_array: np.ndarray,
    oxidation_states: Dict[str, List[int]],
    magmoms_or_charges: List[float],
    adaptive_boundaries: bool = True,
    bounds: Dict[str, Dict[int, float]] | None = None,
    *,
    higher_value_means_higher_oxidation: bool = False,
) -> List[int]:
    """
    Assign oxidation state per site from values (magnetic moments or charges).

    species_array: (n_sites,) element symbols or species strings.
    magmoms_or_charges: (n_sites,) magnetic moments or charges.
    higher_value_means_higher_oxidation: If True, treat as charges (higher => higher oxidation).
    """
    n = len(species_array)
    if len(magmoms_or_charges) != n:
        raise ValueError("species_array and magmoms must have same length")
    values = [-float(v) for v in magmoms_or_charges] if higher_value_means_higher_oxidation else [float(v) for v in magmoms_or_charges]
    elements_sites = [_element_from_species_string(str(s)) for s in species_array]
    default_state_by_elem = {
        elem: states[0] for elem, states in oxidation_states.items()
    }
    oxi_states = [default_state_by_elem.get(el, 0) for el in elements_sites]
    elements_in_structure = set(elements_sites)
    multi_state_elements = [
        elem
        for elem, states in oxidation_states.items()
        if len(states) > 1 and elem in elements_in_structure
    ]
    if len(multi_state_elements) == 0:
        return oxi_states
    if adaptive_boundaries:
        values_by_elem: Dict[str, List[float]] = {}
        indices_by_elem: Dict[str, List[int]] = {}
        for elem in multi_state_elements:
            mask = [el == elem for el in elements_sites]
            values_by_elem[elem] = [values[i] for i, m in enumerate(mask) if m]
            indices_by_elem[elem] = [i for i, m in enumerate(mask) if m]
        composition = composition_from_species_array(species_array)
        thresholds = _solve_boundaries_for_multiple_elements(
            composition,
            multi_state_elements,
            values_by_elem,
            oxidation_states,
        )
        for elem in multi_state_elements:
            states_sorted = sorted(oxidation_states[elem], reverse=True)
            ox_high, ox_low = states_sorted[0], states_sorted[-1]
            th = thresholds.get(elem)
            if th is None:
                continue
            for idx in indices_by_elem[elem]:
                oxi_states[idx] = ox_high if values[idx] <= th else ox_low
        return oxi_states
    if bounds is None:
        raise ValueError(
            "bounds must be provided when adaptive_boundaries is False"
        )
    for elem in multi_state_elements:
        for i, el in enumerate(elements_sites):
            if el == elem:
                oxi_states[i] = get_oxi_state_from_value(values[i], elem, bounds)
    return oxi_states
