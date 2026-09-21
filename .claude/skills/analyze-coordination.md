---
description: Compute RDFs, coordination numbers, and cage correlation functions from an MD trajectory
---

# Analyze Coordination

Use `CoordinationAnalyzer` to compute radial distribution functions g(r), running coordination numbers, and cage correlation functions C(tau) from a trajectory.

## Quick usage

```python
from py_oats.io.trajectory import TrajectoryData
from py_oats.analyzers.coordination import CoordinationAnalyzer

data = TrajectoryData.read("production.dump", temperature=1500.0, time_step=2.0, step_skip=1)
analyzer = CoordinationAnalyzer(data, rmax=8.0, ngrid=201, sigma=0.1)
analyzer.analyze()

# Access RDF results
for pair_key, rdf_result in analyzer.rdfs.items():
    print(f"{pair_key}: first peak at {rdf_result.peaks[0]:.2f} A, CN = {rdf_result.coordination_number[-1]:.1f}")

# Cage correlation (how long does the neighbor shell persist?)
cage = analyzer.get_cage_correlation(
    mobile_species="Li",
    cage_species=["O"],
    cutoff=3.0,  # use first RDF minimum
)
print(f"Cage time: {cage.cage_time:.1f} fs")
```

## CoordinationAnalyzer parameters

- `rmax`: maximum r for g(r) (default: 8.0 A)
- `ngrid`: number of grid points (default: 201)
- `sigma`: Gaussian broadening sigma (default: 0.1 A)
- `frame_sample`: sample every N frames (default: all)

## get_cage_correlation() parameters

- `mobile_species`: species whose cage is tracked (str or list)
- `cage_species`: species forming the cage (str or list)
- `cutoff`: neighbor cutoff in A (use first minimum of the mobile-cage RDF)
- `max_lag`: maximum lag in frames (default: half the trajectory)

## Schema

```python
from py_oats.schemas.coordination import CoordinationAnalysisDoc
doc = CoordinationAnalysisDoc.from_analyzer(analyzer, cage_results={"Li-O": cage})
```

## Plotting

```python
from py_oats.plotter.coordination import plot_rdf, plot_rdf_grid
fig = plot_rdf_grid(analyzer, show_coordination_number=True)
fig = plot_rdf(analyzer, pairs=["Li-O", "Al-O"], show_coordination_number=True)
```

## Files involved

- `src/py_oats/analyzers/coordination.py` — `CoordinationAnalyzer`
- `src/py_oats/schemas/coordination.py` — `CoordinationAnalysisDoc`, `RDFResult`, `CageResult`
- `src/py_oats/plotter/coordination.py` — plotting functions
- `src/py_oats/utils/analyzers/coordination/neighbors.py` — fast periodic neighbor list
