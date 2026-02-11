import matplotlib.pyplot as plt
import numpy as np
from .analyzer import OnsagerTransportAnalyzer
from .schema import TransportDoc
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


def plot_msd(times : list, msds : list, labels=[1, 2], log=False, size=10):
    fig = plt.figure(figsize=(size, size))
    plt.plot(times, msds[0], label=labels[0] + '-' + labels[0] + '_net')
    plt.plot(times, msds[1], label=labels[0] + '-' + labels[0] +'_self')
    plt.plot(times, msds[2], label=labels[1] + '-' + labels[1] +'_net')
    plt.plot(times, msds[3], label=labels[1] + '-' + labels[1] +'_self')
    plt.plot(times, msds[4], label=labels[0] + '-' + labels[1])
    plt.xlabel('Time (fs)')
    plt.ylabel('Correlation function in 1/(cm-eV)')
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.legend()
    plt.show()

def plot_correlation_pair(analyzer_or_doc : OnsagerTransportAnalyzer | TransportDoc, i : Union[str, int, Element], j : Union[str, int, Element], log=False, return_msd=False):
    '''
    Plots the time correlation function for a pair of species, indexed by i and j, which can either be strings or Element objects or the index of the species from the analyzer.species object. 
    If return_msd is True, the "MSD" array is returned.
    Note : all numbers are scaled by kB*T*V, as per the Green-Kubo relation.
    '''
    if isinstance(analyzer_or_doc, OnsagerTransportAnalyzer):
        doc = TransportDoc.from_analyzer(analyzer_or_doc)
    else:
        doc = analyzer_or_doc
    
    if isinstance(i, str):
        i = doc.mapping[i]
    if isinstance(j, str):
        j = doc.mapping[j]
    if isinstance(i, Element):
        i = doc.mapping[i.symbol]
    if isinstance(j, Element):
        j = doc.mapping[j.symbol]
    plot_msd(doc.times*doc.time_step*doc.step_skip, doc.msds[(i, j)], labels=[doc.species[i], doc.species[j]], log=log)
    if return_msd:
        return doc.msds[(i, j)]

def plot_correlation_pairwise(analyzer_or_doc: OnsagerTransportAnalyzer | TransportDoc, size=10):
    '''
    Plots the time correlation function for all pairs of species in the system. 
    Note : all numbers are scaled by kB*T*V, as per the Green-Kubo relation.
    '''
    if isinstance(analyzer_or_doc, OnsagerTransportAnalyzer):
        doc = TransportDoc.from_analyzer(analyzer_or_doc)
    else:
        doc = analyzer_or_doc
    for i in range(len(doc.species)):
        for j in range(i, len(doc.species)):
            if i != j:
                plot_msd(doc.times*doc.time_step*doc.step_skip, doc.msds[(i, j)], labels=[doc.species[i], doc.species[j]], size=size)


def plot_all_correlations(analyzer_or_doc: OnsagerTransportAnalyzer | TransportDoc, size=10):
    '''
    Plots all possible correlation functions (auto and cross) for the system, including self effects. 
    Note : all numbers are scaled by kB*T*V, as per the Green-Kubo relation.
    '''
    fig = plt.figure(figsize=(size, size))
    if isinstance(analyzer_or_doc, OnsagerTransportAnalyzer):
        doc = TransportDoc.from_analyzer(analyzer_or_doc)
    else:
        doc = analyzer_or_doc
    for i in range(len(doc.species)):
        for j in range(i, len(doc.species)):
            if i != j:
                plt.plot(doc.times*doc.time_step*doc.step_skip, doc.msds[(i, j)][4], label=doc.species[i] + '-' + doc.species[j])
                plt.plot(doc.times*doc.time_step*doc.step_skip, doc.msds[(i, j)][1], label=doc.species[i] + '-' + doc.species[i] + '_self')
                plt.plot(doc.times*doc.time_step*doc.step_skip, doc.msds[(i, j)][0], label=doc.species[i] + '-' + doc.species[i] + '_net')
    plt.xlabel('Time (fs)')
    plt.ylabel('Correlation function in 1/(cm-eV)')
    plt.legend()
    plt.show()