---
description: Generate an amorphous structure from a chemical composition using py-OATS
---

# Generate Amorphous Structure

Use `py_oats.structure_generator.generator.get_amorphous_structure` to create a random-packed amorphous structure from a composition.

## Quick usage

```python
from pymatgen.core import Composition
from py_oats.structure_generator.generator import get_amorphous_structure

struct = get_amorphous_structure("Li2S", temperature=5000.0)
struct.to(filename="amorphous_Li2S.json")
```

## CLI script

```bash
python skills/setup/generate_structure.py --composition Li2S --temperature 5000 --target-atoms 200 --output structure.json
```

## Key parameters

- `composition`: Chemical formula string or `pymatgen.core.Composition`
- `temperature`: Temperature for density prediction (K). Higher = lower density. Default: 5000
- `generator_kwargs`: Dict passed to `get_random_packed_structure`:
  - `target_atoms`: Minimum atom count (default: 100)
  - `vol_multiply`: Volume scaling factor (default: 1.0)
  - `packmol_seed`: Reproducibility seed

## How it works

1. Queries Materials Project for stable phases in the relevant chemical systems
2. Extracts polyhedral environments (e.g. SiO4 tetrahedra) via CrystalNN fingerprinting
3. Predicts amorphous density at the given temperature using the linear density descriptor model
4. Packs polyhedra + remaining atoms into a cubic cell using packmol

## Files involved

- `src/py_oats/structure_generator/generator.py` — `get_amorphous_structure`, `get_random_packed_structure`
- `src/py_oats/structure_generator/density_descriptor.py` — `predict_density`
- `src/py_oats/utils/polyhedra.py` — `get_polyhedra_from_mp`
- `src/py_oats/structure_generator/data/db_avg_vols.json.gz` — cached per-element volume database
