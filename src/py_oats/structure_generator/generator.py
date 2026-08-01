"""Random packed structure generation for amorphous/glassy materials.

Adapted from atomate2/common/jobs/mpmorph.py (kinziefarnell/atomate2).
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Molecule, Structure
from pymatgen.io.packmol import PackmolBoxGen

from typing import Literal

from py_oats.structure_generator.density_descriptor import predict_density
from py_oats.utils.polyhedra import get_polyhedra_from_mp

_DEFAULT_AVG_VOL_FILE = Path("~/.cache/py_oats").expanduser() / "db_avg_vols.json.gz"
if not _DEFAULT_AVG_VOL_FILE.parents[0].exists():
    os.makedirs(_DEFAULT_AVG_VOL_FILE.parents[0], exist_ok=True)
_DEFAULT_AVG_VOL_URL = "https://figshare.com/ndownloader/files/49704288"


def _get_average_volumes_file(
    chunk_size: int = 2048, timeout: float = 60
) -> pd.DataFrame:
    """Retrieve stored average volume data from figshare if not cached locally."""
    if not _DEFAULT_AVG_VOL_FILE.exists():
        import requests

        stream_data = requests.get(_DEFAULT_AVG_VOL_URL, stream=True, timeout=timeout)
        with open(str(_DEFAULT_AVG_VOL_FILE), "wb") as file:
            for chunk in stream_data.iter_content(chunk_size=chunk_size):
                file.write(chunk)

    return pd.read_json(_DEFAULT_AVG_VOL_FILE, orient="split")


def _get_chem_env_key_from_composition(
    composition: Composition, ignore_oxi_states: bool = True
) -> str:
    """Return a dunder-separated chemical environment string, e.g. 'Ag+__Cu2+__O2-'."""
    comp = composition
    if ignore_oxi_states:
        comp = comp.remove_charges()
    chem_env = "__".join(sorted(set(comp.as_dict())))
    for char in ["+", "-"]:
        chem_env = chem_env.replace(f"0{char}", "")
    return chem_env


def get_average_volume_from_database(
    composition: Composition,
    avg_vols: pd.DataFrame,
    ignore_oxi_states: bool = True,
) -> float:
    """
    Get average volume per atom for a composition from a cached database DataFrame.

    Parameters
    ----------
    composition : Composition
        Target composition.
    avg_vols : pd.DataFrame
        DataFrame with columns "chem_env", "avg_vol", "count", "with_oxi".
    ignore_oxi_states : bool
        Whether to ignore oxidation states in the lookup.

    Returns
    -------
    float
        Average volume per atom in Angstrom^3.
    """

    def get_entry_from_dict(chem_env: str) -> dict | None:
        data = avg_vols[avg_vols["chem_env"] == chem_env]
        data = data[
            (
                data["with_oxi"]
                if (not ignore_oxi_states and len(data[data["with_oxi"]]) > 0)
                else ~data["with_oxi"]
            )
        ]
        if len(data) > 0:
            return {k: data[k].squeeze() for k in ("avg_vol", "count")}
        return None

    chem_env_key = _get_chem_env_key_from_composition(composition, ignore_oxi_states)
    if (avg_vol := get_entry_from_dict(chem_env_key)) is not None:
        return avg_vol["avg_vol"]

    vols = []
    counts = 0
    for ielt in range(2, len(composition)):
        for combo in combinations(composition, ielt):
            chem_env_key = _get_chem_env_key_from_composition(
                Composition({spec: 1 for spec in combo}),
                ignore_oxi_states=ignore_oxi_states,
            )
            if (avg_vol := get_entry_from_dict(chem_env_key)) is not None:
                vols.append(avg_vol["avg_vol"] * avg_vol["count"])
                counts += avg_vol["count"]

    return sum(vols) / counts


def get_average_volume_from_db_cached(
    composition: Composition,
    db_name: str,
    cache_file: pd.DataFrame | None = None,
    ignore_oxi_states: bool = True,
) -> float:
    """
    Get average volume per atom using locally cached database data.

    Parameters
    ----------
    composition : Composition
        Target composition.
    db_name : str
        Database name to filter by (e.g. "mp" or "icsd").
    cache_file : pd.DataFrame | None
        Pre-loaded DataFrame; downloads from figshare if None.
    ignore_oxi_states : bool
        Whether to ignore oxidation states.

    Returns
    -------
    float
        Average volume per atom in Angstrom^3.
    """
    avg_vols = cache_file if cache_file is not None else _get_average_volumes_file()
    avg_vols = avg_vols[avg_vols["source"] == db_name]
    return get_average_volume_from_database(
        composition,
        avg_vols=avg_vols,
        ignore_oxi_states=ignore_oxi_states,
    )


def get_average_volume_from_mp_api(
    composition: Composition, mp_api_key: str | None = None
) -> float:
    """
    Get average volume per atom from the Materials Project API (live query).

    Parameters
    ----------
    composition : Composition
        Target composition.
    mp_api_key : str | None
        MP API key; falls back to the MP_API_KEY environment variable if None.

    Returns
    -------
    float
        Average volume per atom in Angstrom^3.
    """
    from mp_api.client import MPRester

    with MPRester(api_key=mp_api_key) as mpr:
        comp_entries = mpr.get_entries(composition.reduced_formula, inc_structure=True)

    if len(comp_entries) > 0:
        vols = [
            entry.structure.volume / entry.structure.num_sites for entry in comp_entries
        ]
    else:
        with MPRester(api_key=mp_api_key) as mpr:
            _entries = mpr.get_entries_in_chemsys(
                [str(el) for el in composition.elements], inc_structure=True
            )
        entries = [
            entry
            for entry in _entries
            if len(set(composition).intersection(set(entry.structure.composition))) > 1
        ]
        vols = [entry.structure.volume / entry.structure.num_sites for entry in entries]

    return float(np.mean(vols))


def get_average_volume_from_mp(
    composition: Composition, use_cached: bool = True, **kwargs
) -> float:
    """
    Get average volume per atom from MP data (cached or live API).

    Parameters
    ----------
    composition : Composition
        Target composition.
    use_cached : bool
        Use locally cached MP data (True) or query the MP API (False).
    **kwargs
        Forwarded to `get_average_volume_from_db_cached` or
        `get_average_volume_from_mp_api`.

    Returns
    -------
    float
        Average volume per atom in Angstrom^3.
    """
    if use_cached:
        return get_average_volume_from_db_cached(composition, db_name="mp", **kwargs)
    return get_average_volume_from_mp_api(composition, **kwargs)


def get_random_packed_structure(
    composition: Composition | str,
    polyhedras: list[Molecule] | None = None,
    target_atoms: int = 100,
    vol_multiply: float = 1.5,
    tol: float = 2.0,
    vol_per_atom_source: Literal["mp", "icsd"] = "mp",
    pbc: bool = False,
    db_kwargs: dict | None = None,
    packmol_seed: int = 1,
    packmol_output_dir: str | Path | None = None,
) -> Structure:
    """
    Generate a random packed structure with a target number of atoms.

    Polyhedra packing follows Zachariasen's random network theory.
    Designed for amorphous/glassy structures. Defaults to cached MP data.

    Parameters
    ----------
    composition : Composition | str
        Target composition.
    polyhedras : list[Molecule] | None
        Polyhedra molecules to pack into the cell; remaining atoms fill stoichiometry.
    target_atoms : int
        Approximate target number of atoms in the cell.
    vol_multiply : float
        Scale factor applied to the estimated volume.
    tol : float
        Buffer (Å) subtracted from each box face to avoid overlaps at boundaries.
        Set to 0 when pbc=True.
    vol_per_atom_source : Literal["mp", "icsd", "descriptor"]
        Volume per atom source: "mp" (cached), "icsd" (cached), or "descriptor" (fit to existing data).
    pbc : bool
        Apply periodic boundary conditions in packmol (requires packmol >= 20.15.0).
    db_kwargs : dict | None
        Extra kwargs forwarded to the volume-lookup function.
    packmol_seed : int
        Random seed for packmol.
    packmol_output_dir : str | Path | None
        Directory for packmol output files; uses a temp dir if None.

    Returns
    -------
    Structure
        Randomly packed pymatgen Structure.
    """
    if isinstance(composition, (str, dict)):
        composition = Composition(composition)

    struct_db = (
        vol_per_atom_source.lower() if isinstance(vol_per_atom_source, str) else None
    )
    db_kwargs = db_kwargs or ({"use_cached": True} if struct_db == "mp" else {})

    if isinstance(vol_per_atom_source, (float, int)):
        vol_per_atom = vol_per_atom_source
    elif struct_db == "mp":
        vol_per_atom = get_average_volume_from_mp(composition, **db_kwargs)
    elif struct_db == "icsd":
        vol_per_atom = get_average_volume_from_db_cached(
            composition, db_name="icsd", **db_kwargs
        )
    else:
        raise ValueError(f"Unknown volume per atom source: {vol_per_atom_source!r}.")

    formula, _ = composition.get_integer_formula_and_factor()
    integer_composition = Composition(formula)
    full_cell_composition = integer_composition * np.ceil(
        target_atoms / integer_composition.num_atoms
    )

    if polyhedras:
        polyhedra_total_comp = sum(
            (poly.composition for poly in polyhedras), start=Composition()
        )
        num_polyhedra_sites = int(
            min(
                full_cell_composition.as_dict()[el] / polyhedra_total_comp.as_dict()[el]
                for el in full_cell_composition.as_dict()
                if el in polyhedra_total_comp.as_dict()
            )
        )
        atomic_site_composition = full_cell_composition - (
            polyhedra_total_comp * num_polyhedra_sites
        )
    else:
        atomic_site_composition = full_cell_composition

    supercell_composition = {
        str(el): int(atomic_site_composition.element_composition.get(el))
        for el in full_cell_composition
        if int(atomic_site_composition.element_composition.get(el)) != 0
    }

    with TemporaryDirectory() as tmpdir:
        molecules = []
        used_names: set[str] = set()

        def _unique_name(base: str) -> str:
            name = base
            i = 1
            while name in used_names:
                name = f"{base}_{i}"
                i += 1
            used_names.add(name)
            return name

        if polyhedras:
            for poly in polyhedras:
                name = _unique_name(poly.composition.reduced_formula)
                xyz_file = f"{tmpdir}/{name}.xyz"
                poly.to(xyz_file)
                molecules.append(
                    {
                        "name": name,
                        "number": num_polyhedra_sites,
                        "coords": xyz_file,
                    }
                )

        for element, num_sites in supercell_composition.items():
            name = _unique_name(element)
            xyz_file = f"{tmpdir}/{name}.xyz"
            with open(xyz_file, "w") as f:
                f.write(f"1\ncomment\n{element} 0.0 0.0 0.0\n")
            molecules.append({"name": name, "number": num_sites, "coords": xyz_file})

        box_scale = (vol_per_atom * full_cell_composition.num_atoms * vol_multiply) ** (
            1 / 3
        )
        box_lower_bound = tol / 2
        box_upper_bound = box_scale - tol / 2
        box_size = 3 * [box_lower_bound] + 3 * [box_upper_bound]

        packmol_additional_params = (
            {"pbc": [" ".join(map(str, box_size)) + "\n"]} if pbc else {}
        )

        packmol_set = PackmolBoxGen(
            seed=packmol_seed,
            control_params=packmol_additional_params,
        ).get_input_set(molecules=molecules, box=box_size)
        packmol_output_dir = str(packmol_output_dir or tmpdir)
        packmol_set.write_input(directory=packmol_output_dir)
        packmol_set.run(path=packmol_output_dir)

        mol = Molecule.from_file(f"{packmol_output_dir}/packmol_out.xyz")

    return Structure(
        [[box_scale if i == j else 0.0 for j in range(3)] for i in range(3)],
        species=mol.species,
        coords=mol.cart_coords,
        coords_are_cartesian=True,
    )


def get_amorphous_structure(
    composition: Composition | str,
    temperature: float = 5000.0,
    generator_kwargs: dict | None = None,
) -> Structure:
    """
    Generate a random packed structure with a target number of atoms.
    All poylhedra corresponding to the given composition are obtained 
    from the corresponding MP structures and packed into a random structure. 
    The remaining atoms are filled in to satisfy the stoichiometry.
    The density of the box is at temperature T is predicted using a 
    density descriptor model which is a linear regression model trained on
    MP's ab-initio diffusivity data.
    
    NOTE: The generated structure is not guaranteed to be physically realistic, 
    and should be relaxed using DFT or other methods before use in simulations. 
    It is only a reasonable starting point to the actual amorphous structure 
    (which depends on temperature, pressure, cooling rate and equilibration time!).

    Parameters
    ----------
    composition : Composition | str
        Target composition.
    temperature : float
        Temperature in Kelvin for density prediction (used if vol_per_atom_source="descriptor").
    seed : int | None
        Random seed for packmol; if None, uses a random seed.
    generator_kwargs : dict | None
        Additional keyword arguments forwarded to `get_random_packed_structure`.
    """
    
    if isinstance(composition, (str, dict)):
        composition = Composition(composition)

    generator_kwargs = generator_kwargs or {}

    density = predict_density(str(composition), temperature)

    # density (g/cm³) → vol_per_atom (ų/atom)
    mass_per_atom_amu = (
        sum(float(el.atomic_mass) * amt for el, amt in composition.items())
        / composition.num_atoms
    )
    _AMU_TO_G = 1.66053906660e-24
    vol_per_atom = (mass_per_atom_amu * _AMU_TO_G) / (density * 1e-24)  # Å³/atom

    expected_polyhedra = get_polyhedra_from_mp(composition)
    guessed_structure = get_random_packed_structure(
        composition, polyhedras=expected_polyhedra, **generator_kwargs
    )

    target_volume = vol_per_atom * guessed_structure.num_sites
    scaled_structure = guessed_structure.copy()
    scaled_structure.scale_lattice(target_volume)

    return scaled_structure