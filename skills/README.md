# py-OATS Skills

Runnable scripts for setting up and submitting LAMMPS MD workflows.

| Script | Description |
|---|---|
| `setup/generate_structure.py` | Generate an amorphous structure from a composition |
| `setup/submit_amorphous_state.py` | Run the amorphous state equilibration workflow |
| `setup/submit_production_md.py` | Run a production NVT MD job from an existing structure |
| `setup/submit_full_pipeline.py` | End-to-end: structure gen + MD + transport analysis |

## Usage

```bash
python skills/setup/generate_structure.py --composition Li2S --temperature 300 --output structure.json
python skills/setup/submit_production_md.py --structure structure.json --temperature 1500 --prod-steps 5000000 --run-locally
```

## Prerequisites

- `py-OATS` installed (`pip install -e .`)
- `atomate2` and `jobflow` for workflow scripts (`pip install atomate2 jobflow`)
- A GRACE potential (default: `~/.cache/grace/GRACE-FS-OAM`)

Post-processing is handled by `AnalysisMaker` inside the workflows, or by using `TransportAnalyzer`, `CoordinationAnalyzer`, and `ChargeStateAnalyzer` directly in Python.
