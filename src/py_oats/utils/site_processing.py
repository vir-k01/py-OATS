import numpy as np
from pymatgen.core import Structure, Element


def find_species(symbol: str, structure: Structure):
    """
    Finds the indices of atoms of a given species in the structure.
    :param symbol: str, chemical symbol of species
    :return: ind: list[int], indices of atoms of given species
    """
    ind = [i for i in range(len(structure)) if structure.sites[i].species.elements[0].symbol == symbol]
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
    
    positions = np.zeros((T, num_ind, 3))
    frac_coords = np.array([[structure.sites[ind[j]].frac_coords for j in range(num_ind)] for structure in structures])
    
    if correct_pbc:
        dfrac_coords = frac_coords[1:, :, :] - frac_coords[:-1, :, :]
        dfrac_coords = np.where(dfrac_coords > 0.5, dfrac_coords - 1, dfrac_coords)
        dfrac_coords = np.where(dfrac_coords < -0.5, dfrac_coords + 1, dfrac_coords)

    # Accumulate positions
    positions[1:, :, :] = np.cumsum(dfrac_coords, axis=0)
        
    if com_frame:
        masses = np.array([[site.species.elements[0].atomic_mass for site in structure.sites] for structure in structures])
        coords = np.array([[site.coords for site in structure.sites] for structure in structures])
        com = np.einsum('ijk,ij->ik', coords, masses) / np.sum(masses)
        positions -= com[:, np.newaxis, :] / np.sum(masses)
            
    final_positions = positions * structures[0].lattice.abc

    return final_positions