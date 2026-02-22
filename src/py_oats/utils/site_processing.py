import numpy as np
from pymatgen.core import Structure, Element, Species


def find_species(symbol: str, structure: Structure):
    """
    Finds the indices of atoms of a given species in the structure.
    :param symbol: str, chemical symbol of species (can include oxidation state like "Li+1", "Ti+2", etc.)
    :return: ind: list[int], indices of atoms of given species
    """
    ind = []
    for i in range(len(structure)):
        site_species = structure.sites[i].species.elements[0]

        # Check if the symbol contains oxidation state (e.g., "Li+1", "Ti+2")
        if "+" in symbol or "-" in symbol:
            # Extract element and oxidation state from symbol
            if "+" in symbol:
                element, ox_state_str = symbol.split("+")
                ox_state = int(ox_state_str)
            else:
                element, ox_state_str = symbol.split("-")
                ox_state = -int(ox_state_str)

            # Check if site species matches both element and oxidation state
            if (isinstance(site_species, Species) and
                site_species.symbol == element and
                site_species.oxi_state == ox_state):
                ind.append(i)
        else:
            # Original behavior for element symbols without oxidation states
            if site_species.symbol == symbol:
                ind.append(i)

    return ind

def prep_positions(ind: list[int], structures: list[Structure], correct_pbc: bool = True, com_frame: bool = False):
    """
    Prepares the positions of a given species for analysis.
    :param ind: list[int], indices of atoms of given species
    :return: positions: array[float,float,float], position of each atom over time.
    Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively
    """
    T = len(structures)
    num_ind = len(ind)

    if num_ind == 0:
        # Return empty array if no atoms of this species found
        return np.zeros((T, 0, 3))

    positions = np.zeros((T, num_ind, 3))
    frac_coords = np.array([[structure.sites[ind[j]].frac_coords for j in range(num_ind)] for structure in structures])

    # Ensure frac_coords has the right shape (T, num_ind, 3)
    if frac_coords.ndim == 2:
        # This happens when num_ind == 1, need to add a dimension
        frac_coords = frac_coords.reshape(T, 1, 3)

    if correct_pbc and T > 1:
        dfrac_coords = frac_coords[1:, :, :] - frac_coords[:-1, :, :]
        dfrac_coords = np.where(dfrac_coords > 0.5, dfrac_coords - 1, dfrac_coords)
        dfrac_coords = np.where(dfrac_coords < -0.5, dfrac_coords + 1, dfrac_coords)

        # Accumulate unwrapped path in fractional coords (displacement from frame 0, then add r_0)
        positions[1:, :, :] = np.cumsum(dfrac_coords, axis=0)
        positions += frac_coords[0]  # positions[t] = unwrapped fractional position at t
        # Convert to cartesian so MSD and L_ij have correct units (length^2, 1/(Omega*cm*s*eV))
        lattice = structures[0].lattice
        final_positions = positions @ lattice.matrix  # fractional -> cartesian (rows of matrix = a, b, c)
    else:
        # No PBC correction or single frame: use coordinates directly and convert to cartesian
        positions = frac_coords * structures[0].lattice.abc
        final_positions = positions

    if com_frame:
        masses = np.array([[site.species.elements[0].atomic_mass for site in structure.sites] for structure in structures])
        coords = np.array([[site.coords for site in structure.sites] for structure in structures])
        com = np.einsum('ijk,ij->ik', coords, masses) / np.sum(masses)
        final_positions = final_positions - com[:, np.newaxis, :] / np.sum(masses)

    return final_positions
