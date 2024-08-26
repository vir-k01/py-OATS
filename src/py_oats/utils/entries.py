import numpy as np
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

def get_entry(formula : str, entries : list[ComputedEntry, ComputedStructureEntry]):
    return [entry for entry in entries if entry.composition.reduced_formula==formula]

def get_min_entry(formula : str, entries : list[ComputedEntry, ComputedStructureEntry]):
    formula_entries = get_entry(formula, entries)
    return formula_entries[np.argmin([entry.energy_per_atom for entry in formula_entries])]
