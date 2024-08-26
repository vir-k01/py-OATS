from mp_api.client import MPRester
from rxn_network.thermo.chempot_diagram import ChemicalPotentialDiagram
from rxn_network.entries.entry_set import GibbsEntrySet
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
import numpy as np
import pandas as pd
from typing import Union
from ..utils.entries import get_entry, get_min_entry


def mu_distance(phases : list[Union[ComputedEntry, ComputedStructureEntry]], chempot : ChemicalPotentialDiagram = None, mode : str ='min'):
    '''
    Compute the distance between the chemical potentials of two phases in a ChemicalPotentialDiagram object.
    Parameters:
        phases : list, list of phases to compare.
        chempot : ChemicalPotentialDiagram, the ChemicalPotentialDiagram object.
        mode : str, 'min', 'max' or 'mean'. The mode to compute the distance.

    Returns:
        distance : np.array[float], the distance between the chemical potentials of the two phases.
    
    Note: 
    
    '''
    if chempot is None:
        species = set()
        for phase in phases:
            species.update(phase.composition.get_el_amount_dict().keys())
        with MPRester() as mpr:
            entries = mpr.get_entries_in_chemsys(species)
        chempot = ChemicalPotentialDiagram.from_entries(GibbsEntrySet.from_computed_entries(entries))
    
    domains = []
    for phase in phases:
        if phase in list(chempot.domains.keys()):
            domains.append(chempot.domains[phase])
        else:
            domains.append(chempot.metastable_domains[phase])

    if mode == 'max':
        max_distance = 0
        ind_1, ind_2 = 0, 0
        for node1 in range(len(domains[0])):
            for node2 in range(len(domains[1])):
                distance = np.linalg.norm(domains[0][node1] - domains[1][node2])
                if distance > max_distance:
                    max_distance = distance
                    ind_1, ind_2 = node1, node2
        return domains[0][ind_1] - domains[1][ind_2] 
    
    if mode == 'min':
        min_distance = 1000
        ind_1, ind_2 = 0, 0
        for node1 in range(len(domains[0])):
            for node2 in range(len(domains[1])):
                distance = np.linalg.norm(domains[0][node1] - domains[1][node2])
                if distance < min_distance:
                    min_distance = distance
                    ind_1, ind_2 = node1, node2
        return domains[0][ind_1] - domains[1][ind_2]
    
    if mode == 'mean':
        return np.mean(domains[0], axis=0) - np.mean(domains[1], axis=0)
    

def get_fluxes_across_interface(interface : list[Union[ComputedStructureEntry, ComputedEntry, str]], L_data : Union[pd.DataFrame, np.array[float], list[np.array[float]]], temperature : int, chempot : ChemicalPotentialDiagram, mode : str ='min'):
    '''
    
    Compute the ionic fluxes across a solid-solid interface, approximating the fluxes as the product of the Onsager coefficients and the chemical potential gradient. 
    Parameters:
        interface : list, list of the three phases forming the interface in the order: [reactant1, product, reactant2]. Can be a list of ComputedEntry objects or strings. If strings, MPRester is used to query the most stable entry of that formula.
        L_data : Union[pd.DataFrame, np.array[float], list[np.array[float]]], the L_ij values for the interface. Can be a pandas DataFrame, a numpy array or a list of numpy arrays. If a DataFrame, the temperature is used to extract the L_ij values. If a numpy array, the L_ij values are directly used. If a list of numpy arrays, the first element is used as the L_ij values and the second element is used as the standard deviation of the L_ij values.
        temperature : int, the temperature at which the fluxes are computed.
        chempot : ChemicalPotentialDiagram, the ChemicalPotentialDiagram object. If none, MPRester is used to query the entries in the chemical system of the interface and a ChemicalPotentialDiagram object is created.
        mode : str, 'min', 'max' or 'mean'. The mode to compute the distance. If 'min', the minimum distance between the two phases is used. If 'max', the maximum distance is used. If 'mean', the mean distance is used.
    
    Returns:
        fluxes : np.array[float], the ionic fluxes across the interface.
        fluxes_std : np.array[float], the standard deviation of the ionic fluxes.
    
    Note: Ensure the order of phases and species in the L_data DataFrame matches the order of phases and species in the interface list.
    '''
    if isinstance(interface[0], str):
        with MPRester() as mpr:
            entries = mpr.get_entries(interface)
        interface = [get_min_entry(interface[0], entries), get_min_entry(interface[2], entries), get_min_entry(interface[2], entries)]
            
            
    mu = np.delete(mu_distance([interface[0], interface[2]], chempot, mode), 2)
    
    if isinstance(L_data, pd.DataFrame):
        temps = L_data['Temperature'].unique()
        diffs = np.abs(np.array(temps) - temperature)
        closest_temp_idx = np.argmin(diffs)
        closest_temp = temps[closest_temp_idx]
        
        formula_data =  L_data[(L_data['Formula'] == interface[1]) & (L_data['Temperature'] == closest_temp)]
    # Extract L_ij values
        
        L_ij_values = [
            [formula_data['L00'], formula_data['L01'], formula_data['L02']],
            [formula_data['L01'], formula_data['L11'], formula_data['L12']],
            [formula_data['L02'], formula_data['L12'], formula_data['L22']]
        ]
        L_ij_values_std = [
            [formula_data['L00_std'], formula_data['L01_std'], formula_data['L02_std']],
            [formula_data['L01_std'], formula_data['L11_std'], formula_data['L12_std']],
            [formula_data['L02_std'], formula_data['L12_std'], formula_data['L22_std']]
        ]
        if L_ij_values[0][0].empty:
            #raise ValueError(f"No data found for {interface[1]} at {closest_temp}")
            L_ij_values = np.eye(3)*1e15
            L_ij_values_std = np.eye(3)*1e15
    
    if isinstance(L_data, np.array):
        L_ij_values = L_data
        L_ij_values_std = np.zeros(L_data.shape)
    
    if isinstance(L_data, list):
        L_ij_values = L_data[0]
        L_ij_values_std = L_data[1]

    L = np.array(L_ij_values).reshape(3, 3)
    L_std = np.array(L_ij_values_std).reshape(3, 3)
    return np.dot(L, mu), np.dot(L_std, mu)
