from __future__ import annotations

import os
from pathlib import Path

from pymatgen.core import Structure

from py_oats.analyzers.base import BaseAnalyzer
from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.schemas.transport import TransportDoc


TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "templates"

_AMORPHOUS_STATE_DEFAULTS: dict = {
    "atom_style": "atomic",
    "potential_path": str(Path.home() / ".cache/grace/GRACE-FS-OAM"),
    "temperature": 300.0,
    "pressure": 0.0,
    "time_step": 0.001,
    "t_damp": 0.1,
    "p_damp": 1.0,
    "t_npt": 20.0,
    "t_nvt": 20.0,
    "t_ramp": 20.0,
    "ramp_temperature": 5000.0,
    "thermo_interval": 1000,
    "dump_interval": 1000,
    "density_threshold": 0.30,
    "n_frames": 10,
    "e_tol": 1.0e-10,
    "f_tol": 1.0e-10,
    "max_iter": 20000,
    "max_eval": 200000,
}

_PRODUCTION_MD_DEFAULTS: dict = {
    "atom_style": "atomic",
    "pair_style": "grace",
    "potential_path": str(Path.home() / ".cache/grace/GRACE-FS-OAM"),
    "temperature": 300.0,
    "time_step": 0.001,
    "log_interval": 100,
    "dump_interval": 1000,
    "eq_steps": 50000,
    "prod_steps": 10000000,
    "seed": 12345,
}

_ANALYZER_TO_SCHEMA: dict[type[BaseAnalyzer], type] = {
    TransportAnalyzer: TransportDoc,
}


def _species_string(structure: Structure) -> str:
    """Return a space-separated species string preserving element order."""
    seen: set[str] = set()
    species: list[str] = []
    for site in structure:
        sym = site.specie.symbol
        if sym not in seen:
            seen.add(sym)
            species.append(sym)
    return " ".join(species)


def _find_trajectory_file(run_dir: str) -> str:
    """Locate production.dump or production.dump.gz in a run directory."""
    for name in ("production.dump", "production.dump.gz"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No production dump file found in {run_dir}"
    )
