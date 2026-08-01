"""
Amorphous density descriptor.

Model
-----
    V_amorph/atom = sum_i  frac_i * alpha[i] * beta_i(T) * v_el[i]

    beta_i(T) = 1 + c[i] * (T - T0)

    density = mass_per_atom [amu] * AMU_TO_G / (V_amorph [Å³/atom] * ANG3_TO_CM3)

where:
    frac_i   — atomic fraction of element i
    alpha[i] — per-element amorphization scaling factor (dimensionless, fitted)
    c[i]     — per-element thermal-expansion coefficient (K⁻¹, fitted)
    v_el[i]  — crystalline vol/atom for element i from Materials Project (Å³/atom)
    T0       — reference temperature = 5000 K (melt baseline; beta = 1 there)

Coefficients are stored in ./descriptor_coefficients.json.
Elements absent from the tables default to alpha=1, c=0 (bare crystalline vol/atom).

Quick usage
-----------

    coeffs = load_coefficients()
    rho = predict_density("Al2O3", T=1000, coeffs=coeffs)
    rho = predict_density("NaTi2(PO4)3", T=3000, coeffs=coeffs)
"""

from __future__ import annotations

from monty.serialization import loadfn
from pathlib import Path

from pymatgen.core import Composition

_AMU_TO_G    = 1.66053906660e-24   # grams per atomic mass unit
_ANG3_TO_CM3 = 1.0e-24            # cm³ per Å³

_DEFAULT_COEFFS = loadfn(Path(__file__).parent / "density_coefficients.json")

def predict_density(formula: str, T: float, coeffs: dict = _DEFAULT_COEFFS) -> float:
    """Predict amorphous density (g/cm³) for a given formula and temperature.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. "Al2O3", "NaTi2(PO4)3", "Li3CuS2".
    T : float
        Temperature in Kelvin.
    coeffs : dict
        Loaded from load_coefficients().

    Returns
    -------
    float
        Predicted density in g/cm³.

    Raises
    ------
    KeyError
        If any element in the formula has no entry in the v_el table.
    """
    comp  = Composition(formula)
    T0    = coeffs["base_T"]
    alpha = coeffs["alpha"]
    c     = coeffs["c"]
    v_el  = coeffs["v_el"]

    n_atoms = comp.num_atoms
    fracs   = {str(el): amt / n_atoms for el, amt in comp.items()}

    # predicted volume per atom (Å³/atom)
    v_pred = 0.0
    for el, frac in fracs.items():
        if el not in v_el:
            raise KeyError(f"Element '{el}' not in v_el table.")
        beta    = 1.0 + c.get(el, 0.0) * (T - T0)
        v_pred += frac * alpha.get(el, 1.0) * beta * v_el[el]

    # mass per atom (amu/atom)
    mass_per_atom = sum(float(el.atomic_mass) * amt for el, amt in comp.items()) / n_atoms

    return (mass_per_atom * _AMU_TO_G) / (v_pred * _ANG3_TO_CM3)


def predict_density_batch(
    entries: list[tuple[str, float]],
    coeffs: dict,
) -> list[float | None]:
    """Predict density for a list of (formula, T) pairs.

    Returns None for any formula with an unknown element.
    """
    results = []
    for formula, T in entries:
        try:
            results.append(predict_density(formula, T, coeffs))
        except KeyError:
            results.append(None)
    return results
