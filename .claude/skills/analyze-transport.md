---
description: Compute Onsager transport coefficients and diffusivities from an MD trajectory
---

# Analyze Transport

Use `TransportAnalyzer` to compute the full Onsager L_ij matrix, self/distinct decomposition, and per-species diffusivities from a trajectory.

## Quick usage

```python
from py_oats.io.trajectory import TrajectoryData
from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.schemas.transport import TransportDoc

data = TrajectoryData.read("production.dump", temperature=1500.0, time_step=2.0, step_skip=1)
analyzer = TransportAnalyzer(data)
analyzer.analyze(smoothing="best_fit")

# Access results
print(analyzer.L_tensor)          # Onsager L_ij (1/cm/s/eV)
print(analyzer.L_tensor_self)     # Self component
print(analyzer.L_tensor_dis)      # Distinct component
print(analyzer.get_diffusivity("Li"))  # D_self in cm^2/s

# Serialize
doc = TransportDoc.from_analyzer(analyzer)
```

## TrajectoryData.read() parameters

- `path_or_object`: file path (LAMMPS dump, VASP XDATCAR, .xyz, .traj) or list of ASE Atoms / pymatgen Structures
- `time_step`: time step in **fs** (not ps). For LAMMPS metal units, convert: 0.001 ps = 1 fs
- `step_skip`: frames were dumped every N steps (e.g. dump_interval=1000 → step_skip=1000)
- `temperature`: temperature in K (required by TransportAnalyzer)
- `unwrap`: True (default) to unwrap PBC for continuous displacement paths

## TransportAnalyzer.analyze() parameters

- `smoothing`: `"best_fit"` (default, brute-force best linear interval), `"blockavg"`, or `None`
- `start_step` / `end_step`: frame indices for fitting window (default: 10%-90% of trajectory)

## Units

- `L_tensor`: Onsager coefficients in 1/(cm·s·eV)
- `get_diffusivity(species)`: Einstein self-diffusion in cm²/s. Formula: `slope/(6·N)·0.1` where slope is in Å²/fs
- `times`: time grid in fs
- `volume`: average cell volume in cm³

## Plotting

```python
from py_oats.plotter.transport import plot_all_correlations, plot_correlation_pairwise
fig = plot_all_correlations(analyzer)  # or pass a TransportDoc
fig = plot_correlation_pairwise(analyzer, log=True)
```

## Files involved

- `src/py_oats/analyzers/transport.py` — `TransportAnalyzer`
- `src/py_oats/schemas/transport.py` — `TransportDoc`
- `src/py_oats/plotter/transport.py` — plotting functions
- `src/py_oats/utils/analyzers/transport/correlate.py` — FFT-based MSD computation
- `src/py_oats/utils/analyzers/transport/fitting.py` — linear fitting strategies
