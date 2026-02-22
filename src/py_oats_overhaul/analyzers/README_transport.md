# Transport analyzer (Onsager) – design and skeleton

**Approach in short:** The analyzer takes **TrajectoryData** only (no file I/O). Species come from `trajectory.species` or a user list. We use the same FFT-based MSD and slope fitting as the current Onsager analyzer, but positions come from `trajectory.positions_for_species(label)` and volume/times from `TrajectoryData`. Optional PBC unwrapping can be added in the analyzer. The skeleton in `transport.py` sets up species, mappings, times, volume, and per-species positions; `compute_L_tensor()` is left as a stub that must be wired to FFT + fitting (e.g. from `py_oats.utils` or copied into overhaul utils).

---

## Current implementation (py_oats.core.analyzer.OnsagerTransportAnalyzer)

- **Input**: List of pymatgen `Structure` (from XDATCAR, LAMMPS dump, pymatgen Trajectory, or ASE trajectory via factory methods).
- **Species**: Either inferred from first structure (element symbols) or from `oxidation_states` (charged species like `"Li+"`, `"Mn3+"`). With oxidation: decorate structures (charge decorator or magmoms), then `get_unique_charged_species`.
- **Positions**: `find_species(symbol, structure)` → site indices; `prep_positions(ind, structures)` → (T, N, 3) in **cartesian**, with **PBC unwrapping** in fractional space (cumulative delta, wrap ±0.5).
- **Core**: `compute_L_tensor()`: for each species pair (i,j), `compute_all_Lij_pairs(pos_i, pos_j, volume)` → MSD-like curves via FFT (`msd_fft`, `msd_fft_cross`, `calc_Lii_self`, `calc_Lii`, `calc_Lij`), then `fit_data(..., smoothing)` → slope = L_ij. Stores `L_tensor`, `L_tensor_self`, `L_tensor_dis`, `msds`, `fit_dicts`.
- **Extras**: `compute_D_from_msd(specie)` wraps pymatgen `DiffusionAnalyzer.from_structures`; properties `get_L_tensor`, `get_available_species`, etc.; factory methods for different file types.

## Proposed approach for overhaul

### 1. Single input: `TrajectoryData`

- **No** `from_xdatcar` / `from_lammps_dump` / `from_ase_trajectory` in the analyzer. User does:
  - `td = TrajectoryData.read(path_or_object, ...)` (already supports path, list of Atoms, list of Structures, pymatgen Trajectory),
  - then `analyzer = TransportAnalyzer(td, temperature=..., ...)`.
- All file-format handling stays in the reader; analyzer only sees `TrajectoryData`.

### 2. Species

- **Option A**: Infer from `trajectory.species` (unique labels, e.g. `["Li", "Mn", "O"]` or `["Li+", "Mn3+", "O2-"]` if already decorated).
- **Option B**: User passes `species: list[str] | None`; if None, use unique from `trajectory.species`.
- **Oxidation / decoration**: For “charge by magmoms” or charge decorator we need per-frame structure-like data. Options:
  - (a) Defer: only support pre-decorated `TrajectoryData` (species strings already include oxidation),
  - (b) Add an optional “decorator” that takes `TrajectoryData` + magmoms/charges and returns a new `TrajectoryData` with updated species strings (then analyzer uses that).
- **Skeleton**: Take `species: list[str] | None = None`; if None, `species = sorted(set(trajectory.species))`. Filter to species that have at least one atom (`positions_for_species(s)[1].shape[1] > 0`). No decoration in the first version; document that for charged species the trajectory should already have labels like `"Mn3+"`.

### 3. Positions and PBC

- **TrajectoryData** has `positions` (n_frames, n_atoms, 3) cartesian and `positions_for_species(label)` → (ind, pos) with pos (n_frames, n_match, 3).
- **PBC unwrapping**: Current code unwraps in fractional space. We have `lattice_at_frame(i)` and positions in cartesian. Two options:
  - (a) **Assume unwrapped**: Document that trajectories should be unwrapped (e.g. LAMMPS dump often is); no unwrap in analyzer.
  - (b) **Unwrap in analyzer**: Add a small util that converts positions to fractional, unwraps (delta in frac > 0.5 → ±1), converts back to cartesian using per-frame lattice. Use (b) in skeleton so we match old behavior when trajectory is in fractional/wrapped form.
- **Volume**: From `TrajectoryData`: e.g. `volume = np.mean([np.linalg.det(trajectory.lattice_at_frame(i)) for i in range(trajectory.n_frames)]) * 1e-24` (cm³), or use a helper.

### 4. MSD and L-tensor (unchanged physics)

- Reuse (or copy) FFT-based MSD and fitting:
  - `msd_fft(r)`, `msd_fft_cross(r, k)` for curves,
  - `calc_Lii_self`, `calc_Lii`, `calc_Lij` same as now (positions → MSD-like arrays, then /6/kbT/volume),
  - `fit_data(f, times, start, end, smoothing)` → slope = L_ij.
- **Dependencies**: Prefer copying/coupling to minimal helpers in `py_oats_overhaul.utils` (e.g. `msd.py`, `fitting.py`) so the overhaul can run without the old `py_oats` package, or add an optional dependency on `py_oats.utils` for FFT/fitting in a first version.

### 5. Outputs and API

- **Attributes**: `species`, `mapping` (species → index), `inv_mapping`, `L_tensor`, `L_tensor_self`, `L_tensor_dis`, `msds`, `fit_dicts`, `times`, `temperature`, `volume`, optionally `diffusivity` per species.
- **Methods**: `compute_L_tensor(start_step, end_step)`, optional `compute_D_from_msd(specie)` (can depend on pymatgen `DiffusionAnalyzer` and structures rebuilt from trajectory for that specie, or defer).
- **Properties**: `get_L_tensor`, `get_available_species` (return list of species that have atoms).

### 6. File layout

- `py_oats_overhaul/analyzers/__init__.py`
- `py_oats_overhaul/analyzers/transport.py` – `TransportAnalyzer` class.
- Optional: `py_oats_overhaul/utils/msd.py` – FFT MSD helpers; `py_oats_overhaul/utils/fitting.py` – `fit_data` / best_fit / blockavg, or import from `py_oats.utils` if we keep that dependency.

### 7. Skeleton class (no implementation yet)

See `transport.py` skeleton below: constructor resolves species and positions from `TrajectoryData`, sets times/volume/kbT, and calls `compute_L_tensor()`; `compute_L_tensor` and MSD helpers are stubs. FFT/fitting can be wired to existing `py_oats.utils` or to new utils in the overhaul.
