---
description: Run a production NVT MD job for transport analysis via LAMMPS
---

# Run Production MD Job

Use `ProductionMDMaker` to run a production NVT MD simulation from an equilibrated structure. Includes NPT volume averaging, NVT equilibration, and production NVT with per-type COM velocity tracking.

## Quick usage

```python
from pymatgen.core import Structure
from jobflow import run_locally
from py_oats.workflow.core import ProductionMDMaker

struct = Structure.from_file("relaxed.json")
maker = ProductionMDMaker(settings={"temperature": 1500, "prod_steps": 10000000})
job = maker.make(input_structure=struct)
result = run_locally(job, create_folders=True)
```

## CLI script

```bash
python skills/setup/submit_production_md.py --structure relaxed.json --temperature 1500 --prod-steps 10000000 --run-locally
```

## Default settings

```python
{
    "atom_style": "atomic",
    "pair_style": "grace",
    "potential_path": "~/.cache/grace/GRACE-FS-OAM",
    "temperature": 300.0,
    "time_step": 0.001,       # ps
    "log_interval": 100,
    "dump_interval": 1000,
    "eq_steps": 50000,
    "prod_steps": 10000000,
    "seed": 12345,
}
```

## What the LAMMPS job does

1. Minimization (CG, etol=1e-4, ftol=1e-6)
2. NPT equilibration for `eq_steps`: first half discarded, second half volume-averaged
3. Rescale box to average NPT volume
4. NVT equilibration for `eq_steps` at fixed volume
5. Production NVT for `prod_steps` with:
   - Trajectory dump every `dump_interval` steps (`production.dump`)
   - Per-type COM velocity dump every step (`species_vcm.dat`)
   - Final structure written (`production_final.data`)

## Species handling

`species_string` and `vel_dump_cmd` are auto-computed from the input structure at build time. When chaining from another job (OutputReference), use `run_production_md()` instead — it resolves species at runtime via `Response(replace=...)`.

## Output

- `production.dump` — trajectory with positions, velocities, forces
- `species_vcm.dat` — per-type center-of-mass velocities (for Onsager transport)
- `production_final.data` — final LAMMPS data file
- `LammpsTaskDocument` with structure, thermo log, state

## Files involved

- `src/py_oats/workflow/core.py` — `ProductionMDMaker`, `run_production_md`
- `src/py_oats/utils/workflow/helpers.py` — `_PRODUCTION_MD_DEFAULTS`, `_build_species_settings`
- `src/py_oats/templates/production_md_job.in` — LAMMPS template
