---
description: Run the amorphous state equilibration workflow (NVT ramp + NPT + density check + minimize) via LAMMPS
---

# Run Amorphous State Job

Use `AmorphousStateMaker` to generate an amorphous structure from a composition and equilibrate it with LAMMPS using a GRACE potential.

## Quick usage

```python
from pymatgen.core import Composition
from jobflow import run_locally
from py_oats.workflow.core import AmorphousStateMaker

maker = AmorphousStateMaker(settings={"temperature": 300, "potential_path": "/path/to/GRACE-FS-OAM"})
job = maker.make(composition=Composition("Li2S"))
result = run_locally(job, create_folders=True)
```

## CLI script

```bash
python skills/setup/submit_amorphous_state.py --composition Li2S --temperature 300 --run-locally
```

## Default settings (override any via `settings={}`)

```python
{
    "atom_style": "atomic",
    "pair_style": "grace padding 0.05",
    "potential_path": "~/.cache/grace/GRACE-FS-OAM",
    "temperature": 300.0,
    "pressure": 0.0,
    "time_step": 0.001,       # ps (LAMMPS metal units)
    "t_damp": 0.1,
    "p_damp": 1.0,
    "t_npt": 20.0,            # ps
    "t_nvt": 20.0,            # ps
    "t_ramp": 20.0,           # ps
    "ramp_temperature": 5000.0,
    "thermo_interval": 1000,
    "dump_interval": 1000,
    "density_threshold": 0.30,
    "n_frames": 10,
    "e_tol": 1e-10,
    "f_tol": 1e-10,
    "max_iter": 20000,
    "max_eval": 200000,
}
```

Only pass keys you want to override — defaults fill in automatically via `__post_init__`.

## What the LAMMPS job does

1. NVT ramp at `ramp_temperature` (e.g. 5000K) for `t_ramp` ps
2. NVT cool from `ramp_temperature` to `temperature` for `t_ramp` ps
3. NPT at `temperature` for `t_npt` ps, captures density
4. Density check: if change > `density_threshold`, reverts to pre-NPT structure
5. Otherwise rescales box to average NPT density
6. Production NVT for `t_nvt` ps, saves `n_frames` restart files
7. Minimizes each restart frame

## Output

The job returns a `LammpsTaskDocument` with:
- `state`: "successful" or "error"
- `dir_name`: path to the run directory
- `structure`: last minimized frame (from `minimized_nvt_frame_N.data`)
- `thermo_log`: parsed thermo data from each stage

## Files involved

- `src/py_oats/workflow/core.py` — `AmorphousStateMaker`
- `src/py_oats/utils/workflow/helpers.py` — `_AMORPHOUS_STATE_DEFAULTS`
- `src/py_oats/templates/amorphous_state_job.in` — LAMMPS template
