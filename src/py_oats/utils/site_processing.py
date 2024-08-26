import numpy as np
from pymatgen.core import Structure, Element


def find_species(symbol: str, structures: list[Structure]):
    """
    Finds the indices of atoms of a given species in the structure.
    :param symbol: str, chemical symbol of species
    :return: ind: list[int], indices of atoms of given species
    """
    ind = [i for i in range(len(structures[0])) if structures[0].sites[i].species.elements[0].symbol == symbol]
    return ind

def prep_positions(ind: list[int], structures: list[Structure], correct_pbc: bool = True, com_frame: bool = False):
    """
    Prepares the positions of a given species for analysis.
    :param ind: list[int], indices of atoms of given species
    :return: positions: array[float,float,float], position of each atom over time.
    Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively
    """
    T = len(structures)
    dpositions = np.zeros((T, len(ind), 3))
    positions = np.zeros((T, len(ind), 3))
    com = np.zeros((T, 3))
    mtot = 0
    for i in range(0, T):
        for j in range(len(structures[i])):
            com[i, :] += structures[i].sites[j].coords*structures[i].sites[j].species.elements[0].atomic_mass
            mtot += structures[i].sites[j].species.elements[0].atomic_mass
    
    if correct_pbc:
        for i in range(1, T):
            for j in range(len(ind)):
                dpositions[i, j, :] = structures[i].sites[ind[j]].frac_coords - structures[i-1].sites[ind[j]].frac_coords
                if np.any(dpositions[i, j, :] > 0.5):
                    dims = np.argwhere(dpositions[i, j, :] > 0.5)
                    dpositions[i, j, dims] = dpositions[i, j, dims] - 1
                if np.any(dpositions[i, j, :] < -0.5):
                    dims = np.argwhere(dpositions[i, j, :] < -0.5)
                    dpositions[i, j, dims] = dpositions[i, j, dims] + 1
                positions[i, j, :] = positions[i-1, j, :] + dpositions[i, j, :]
                if com_frame:
                    positions[i, j, :] = positions[i, j, :] - com[i, :]/mtot

    return positions*structures[0].lattice.abc