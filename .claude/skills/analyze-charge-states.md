---
description: Decorate trajectory species with oxidation states and optionally run transport analysis on charge-resolved species
---

# Analyze Charge States

Use `ChargeStateAnalyzer` to assign oxidation states to trajectory species (e.g. Mn → Mn2+/Mn3+/Mn4+) using magnetic moments, bond valence, or CHGNet, then optionally compute charge-resolved transport.

## Quick usage

```python
from py_oats.io.trajectory import TrajectoryData
from py_oats.analyzers.charge_state import ChargeStateAnalyzer

data = TrajectoryData.read("LiMn2O4.dump", temperature=1250.0, time_step=2.0, step_skip=1)
analyzer = ChargeStateAnalyzer(
    data,
    oxidation_states={"Li": [1], "Mn": [2, 3, 4], "O": [-2]},
)
decorated_trajectories = analyzer.analyze(decorate_freq=-1)

# decorated_trajectories is a list of TrajectoryData with charged species labels
for traj in decorated_trajectories:
    print(traj.unique_species)  # e.g. ['Li+1', 'Mn+2', 'Mn+3', 'O-2']
```

## Charge decorators

```python
from py_oats.utils.analyzers.charge_state.decorators import (
    NaiveChargeDecorator,    # mean oxidation per element (no structure needed)
    BVChargeDecorator,       # bond valence analysis via pymatgen
    CHGNetChargeDecorator,   # CHGNet ML potential for magnetic moments (pip install chgnet)
    AseCalcChargeDecorator,  # any user-provided ASE calculator
)

analyzer = ChargeStateAnalyzer(
    data,
    oxidation_states={"Li": [1], "Mn": [2, 3, 4], "O": [-2]},
    charge_decorator=BVChargeDecorator(),
)
```

## Charge-resolved transport

After decorating, run `TransportAnalyzer` on the decorated trajectory to get oxidation-state-resolved Onsager coefficients (e.g. L(Mn2+, Mn3+)):

```python
from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.schemas.transport import TransportDoc

for traj in decorated_trajectories:
    transport = TransportAnalyzer(traj)
    transport.analyze()
    doc = TransportDoc.from_analyzer(transport)
    print(doc.species, doc.diffusivity)
```

## Parameters

- `oxidation_states`: dict mapping element → list of integer oxidation states
- `charge_decorator`: a `ChargeDecorator` instance (default: `NaiveChargeDecorator`)
- `decorate_freq`: how often to re-assign charges. `-1` = use last frame only (default)

## Files involved

- `src/py_oats/analyzers/charge_state.py` — `ChargeStateAnalyzer`
- `src/py_oats/utils/analyzers/charge_state/decorators.py` — decorator backends
- `src/py_oats/utils/analyzers/charge_state/oxidation.py` — `assign_oxidation_states`
