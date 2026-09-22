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
    "pair_style": "grace padding 0.05",
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
    """Return a space-separated species string in pymatgen Element order.

    Must match the type ordering that ``LammpsData.from_structure`` uses:
    ``sorted(Element(el) for el in symbols)``, which sorts by
    electronegativity (the default ``Element`` comparison).
    """
    from pymatgen.core import Element
    elements = sorted({Element(site.specie.symbol) for site in structure})
    return " ".join(e.symbol for e in elements)


def _build_species_settings(structure: Structure) -> dict:
    """Build species_string and vel_dump_cmd from a resolved Structure."""
    species = _species_string(structure)
    elements = species.split()
    fmt_parts = ["$(step)"]
    for i in range(1, len(elements) + 1):
        fmt_parts += [f"$(c_vcm[{i}][{d}])" for d in (1, 2, 3)]
    title_parts = ["step"]
    for e in elements:
        title_parts += [f"{e}_vx", f"{e}_vy", f"{e}_vz"]
    vel_dump_cmd = (
        f'"{" ".join(fmt_parts)}" '
        f'file species_vcm.dat screen no '
        f'title "# {" ".join(title_parts)}"'
    )
    return {"species_string": species, "vel_dump_cmd": vel_dump_cmd}


#: launcher/wrapper executables that may prefix the LAMMPS binary in a
#  LAMMPS_CMD on HPC systems, e.g. "srun -N1 -n1 --gpus=1 /path/to/lmp".
_LAUNCHERS = frozenset(
    {"srun", "mpirun", "mpiexec", "aprun", "jsrun", "ibrun",
     "numactl", "taskset", "nice", "env", "stdbuf", "time"}
)


def _resolve_lammps_binary(cmd: str) -> str | None:
    """The LAMMPS executable inside a possibly launcher-wrapped command.

    ``LAMMPS_CMD`` is a *command*, not a path: on a cluster it is typically
    "srun <flags> /path/to/lmp". Split it and return the first token that
    resolves to an executable and is not a known launcher; launcher flags
    ("-n1", "--cpu-bind=cores") and their values never resolve, so they are
    skipped naturally.
    """
    import shlex
    import shutil

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    for tok in tokens:
        if tok.startswith("-") or os.path.basename(tok) in _LAUNCHERS:
            continue
        path = shutil.which(tok)
        if path is None and os.path.isfile(tok) and os.access(tok, os.X_OK):
            path = tok
        if path:
            return path
    return None


def check_lammps_grace(
    lammps_cmd: str | None = None,
    pair_style: str = "grace",
) -> str:
    """Verify a LAMMPS binary exists and supports the requested GRACE pair style.

    Searches (in order): ``lammps_cmd`` argument, ``LAMMPS_CMD`` from
    atomate2 settings, ``$lmp`` env var, then ``lmp`` on PATH. Each candidate
    may be a bare binary or a full launch command such as
    ``srun -N1 -n1 --gpus=1 /path/to/lmp``.

    Returns the candidate command string whose binary resolved (verbatim, so
    a launcher prefix is preserved).

    Raises:
        FileNotFoundError: if no LAMMPS binary is found.
        RuntimeError: if the binary lacks the requested pair style.
    """
    import subprocess

    candidates: list[str] = []
    if lammps_cmd:
        candidates.append(lammps_cmd)

    try:
        from atomate2.settings import Atomate2Settings
        settings = Atomate2Settings()
        if settings.LAMMPS_CMD:
            candidates.append(settings.LAMMPS_CMD)
    except Exception:
        pass

    env_lmp = os.environ.get("lmp")
    if env_lmp:
        candidates.append(env_lmp)

    candidates.append("lmp")

    resolved = binary = None
    for cmd in candidates:
        binary = _resolve_lammps_binary(cmd)
        if binary:
            resolved = cmd
            break

    if resolved is None:
        raise FileNotFoundError(
            "No LAMMPS binary found. Searched: "
            + ", ".join(candidates)
            + ". Set LAMMPS_CMD in atomate2.yaml, export $lmp, or add lmp to PATH."
        )

    style_name = pair_style.split()[0]
    try:
        # Probe the bare binary, never the launcher: `srun ... -h` would try
        # to attach to an allocation. MPICH_GPU_SUPPORT_ENABLED=0 keeps a
        # Cray-linked binary from aborting on nodes without the GTL library.
        result = subprocess.run(
            [binary, "-h"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "MPICH_GPU_SUPPORT_ENABLED": "0"},
        )
        if result.returncode == 0:
            help_text = result.stdout + result.stderr
            if style_name not in help_text:
                raise RuntimeError(
                    f"LAMMPS at {binary} does not support pair_style "
                    f"'{style_name}'. Rebuild LAMMPS with the ML-PACE or "
                    f"GRACE plugin."
                )
    except (subprocess.TimeoutExpired, OSError):
        pass

    return resolved


def _find_trajectory_file(run_dir: str) -> str:
    """Locate production.dump or production.dump.gz in a run directory."""
    for name in ("production.dump", "production.dump.gz"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No production dump file found in {run_dir}"
    )
