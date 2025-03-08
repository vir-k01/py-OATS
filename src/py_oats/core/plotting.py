import matplotlib.pyplot as plt
import numpy as np
from .analyzer import OnsagerTransportAnalyzer
from typing import Union
from pymatgen.core import Element

def plot_msds_from_traj(analyzer : OnsagerTransportAnalyzer, size: int = 10, smoothed: bool = False):
    """
    Plot the MSDs for all species in a trajectory. This is a wrapper for the DiffusionAnalyzer.plot() method.
    :param trajectory: Trajectory object
    """
    time = np.arange(len(analyzer.structures))
    plt.figure(figsize=(size, size))
    for specie in set(analyzer.structures[0].species):
        D, msd = analyzer.compute_D_from_msd(specie, smoothed=smoothed)
        analyzer.diffusivity[specie] = D
        plt.plot(time*analyzer.time_step, msd, label=specie)
        print('D for ' + str(specie) + ': '+ str(D) + ' cm2/s')
    plt.xlabel('Time (fs)')
    plt.ylabel('MSD')
    plt.legend()
    plt.show()


def plot_msd(analyzer : OnsagerTransportAnalyzer, msds : list, labels=[1, 2], log=False, size=10):
    plt.figure(figsize=(size, size))
    plt.plot(analyzer.times, msds[0], label=labels[0] + '-' + labels[0] + '_net')
    plt.plot(analyzer.times, msds[1], label=labels[0] + '-' + labels[0] +'_self')
    plt.plot(analyzer.times, msds[2], label=labels[1] + '-' + labels[1] +'_net')
    plt.plot(analyzer.times, msds[3], label=labels[1] + '-' + labels[1] +'_self')
    plt.plot(analyzer.times, msds[4], label=labels[0] + '-' + labels[1])
    plt.xlabel('Time (fs)')
    plt.ylabel('MSD')
    plt.title(analyzer.composition + ' at ' + str(analyzer.temperature) + ' K')
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.legend()
    plt.show()

def plot_correlation_pair(analyzer : OnsagerTransportAnalyzer, i : Union[str, int, Element], j : Union[str, int, Element], log=False, return_msd=False):
    '''
    Plots the time correlation function for a pair of species, indexed by i and j, which can either be strings or Element objects or the index of the species from the analyzer.species object. 
    If return_msd is True, the "MSD" array is returned.
    Note : all numbers are scaled by kB*T*V, as per the Green-Kubo relation.
    '''
    if isinstance(i, str):
        i = analyzer.mapping[i]
    if isinstance(j, str):
        j = analyzer.mapping[j]
    if isinstance(i, Element):
        i = analyzer.mapping[i.symbol]
    if isinstance(j, Element):
        j = analyzer.mapping[j.symbol]
    plot_msd(analyzer.msds[analyzer.msd_map[(i, j)]], labels=[analyzer.species[i], analyzer.species[j]], log=log)
    if return_msd:
        return analyzer.msds[analyzer.msd_map[(i, j)]]

def plot_correlation_pairwise(analyzer: OnsagerTransportAnalyzer, size=10):
    '''
    Plots the time correlation function for all pairs of species in the system. 
    Note : all numbers are scaled by kB*T*V, as per the Green-Kubo relation.
    '''
    for i in range(len(analyzer.species)):
        for j in range(i, len(analyzer.species)):
            if i != j:
                plot_msd(analyzer, analyzer.msds[analyzer.msd_map[(i, j)]], labels=[analyzer.species[i], analyzer.species[j]], size=size)


def plot_all_correlations(analyzer: OnsagerTransportAnalyzer, size=10):
    '''
    Plots all possible correlation functions (auto and cross) for the system, including self effects. 
    Note : all numbers are scaled by kB*T*V, as per the Green-Kubo relation.
    '''
    plt.figure(figsize=(size, size))
    for i in range(len(analyzer.species)):
        for j in range(i, len(analyzer.species)):
            if i != j:
                plt.plot(analyzer.times, analyzer.msds[analyzer.msd_map[(i, j)]][4], label=analyzer.species[i] + '-' + analyzer.species[j])
                plt.plot(analyzer.times, analyzer.msds[analyzer.msd_map[(i, j)]][1], label=analyzer.species[i] + '-' + analyzer.species[i] + '_self')
                plt.plot(analyzer.times, analyzer.msds[analyzer.msd_map[(i, j)]][0], label=analyzer.species[i] + '-' + analyzer.species[i] + '_net')
    plt.xlabel('Time (fs)')
    plt.ylabel('Correlation function')
    plt.title(analyzer.composition + ' at ' + str(analyzer.temperature) + ' K')
    plt.legend()
    plt.show()