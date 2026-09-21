---
description: Run the full end-to-end pipeline — structure generation, LAMMPS MD, and transport analysis
---

# Run Full Pipeline

Use `ProductionOnsagerMaker` to chain structure generation, production MD, and transport analysis into a single jobflow Flow.

## Quick usage

```python
from pymatgen.core import Composition
from jobflow import run_locally
from py_oats.workflow.core import ProductionOnsagerMaker, ProductionMDMaker

maker = ProductionOnsagerMaker(
    production_md_maker=ProductionMDMaker(settings={
        "temperature": 1500,
        "prod_steps": 5000000,
    }),
)
flow = maker.make(structure_or_composition=Composition("YAlO3"))
result = run_locally(flow, create_folders=True)
```

## CLI script

```bash
python skills/setup/submit_full_pipeline.py --composition YAlO3 --temperature 1500 --prod-steps 5000000 --run-locally
```

## Accepts either Composition or Structure

- **Composition**: generates amorphous structure first (3 jobs: amorphous + production + analysis)
- **Structure**: skips generation (2 jobs: production + analysis)

## Flow structure

```
Composition("YAlO3")
  └─ AmorphousStateMaker.make(composition)  →  equilibrated structure
      └─ run_production_md(structure, maker)  →  production MD (Response(replace=...))
          └─ AnalysisMaker.make(run_dir)      →  [TransportDoc, ...]
```

## Configuring sub-makers

All three sub-makers are exposed as fields:

```python
from py_oats.workflow.core import (
    AmorphousStateMaker, ProductionMDMaker, AnalysisMaker, ProductionOnsagerMaker
)
from py_oats.analyzers.coordination import CoordinationAnalyzer
from py_oats.analyzers.transport import TransportAnalyzer

maker = ProductionOnsagerMaker(
    amorphous_state_maker=AmorphousStateMaker(settings={
        "potential_path": "/path/to/GRACE-2L",
        "pair_style": "grace padding 0.05",
    }),
    production_md_maker=ProductionMDMaker(settings={
        "temperature": 1500,
        "prod_steps": 10000000,
        "pair_style": "grace",
        "potential_path": "/path/to/GRACE-FS-OAM",
    }),
    analysis_maker=AnalysisMaker(
        analyzers=[TransportAnalyzer, CoordinationAnalyzer],
    ),
)
```

`AnalysisMaker` automatically syncs `temperature`, `time_step`, and `dump_interval` from the production maker's settings.

## Analysis output

`AnalysisMaker` reads the production dump, runs the specified analyzers, and returns a list of schema docs:
- `TransportAnalyzer` → `TransportDoc` (L_ij, diffusivities, correlation functions)
- `CoordinationAnalyzer` → `CoordinationAnalysisDoc` (RDFs, cage correlations) — requires adding to `_ANALYZER_TO_SCHEMA`

## FireWorks support

```python
from jobflow.managers.fireworks import flow_to_workflow
wf = flow_to_workflow(flow)
# Submit to LaunchPad
```

## Files involved

- `src/py_oats/workflow/core.py` — all makers + `run_production_md`
- `src/py_oats/utils/workflow/helpers.py` — defaults, helpers
- `src/py_oats/templates/` — LAMMPS templates
