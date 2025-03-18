from pymatgen.core import Structure, Element
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.vasp import Xdatcar
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from typing import List, Union
import numpy as np
from ..utils.FFT import msd_fft, msd_fft_cross
from ..utils.fitting import fit_data
from ..utils.site_processing import find_species, prep_positions

class OnsagerTransportAnalyzer():
    '''
    Class to fit the transport coefficients to the Onsager relations. Based on repo by Kara Fong (@kdfong).
    Provide a list of pymatgen structure taken from an NVT MD run, alongwith simulation temperature and species to fit the transport coefficients.
    Pairwise MSDs can be plotted using the plot_msd method. The transport coefficients can be accessed using the L_tensor attribute. 
    The order in which the transport coefficients can be accessed by calling the OnsagerFitting.species attribute.
    Smoothing can be specificed: 'blockavg' for block averaging, 'best_fit' for best fit, 'None' for no smoothing. The default is best_fit.
    '''
    def __init__(self, 
                 structures: list[Structure], 
                 temperature: float, 
                 species: List[Union[str, Element]], 
                 time_step: float = 2, 
                 step_skip: int = 1, 
                 smoothing: str = None
                 ):
        self.structures = structures
        self.mapping = dict()
        self.composition = self.structures[0].composition.reduced_formula
        if species:
            self.species = species
        else:
            self.species = set()
            for s in self.structures[0].species:
                self.species.add(s.symbol)
            self.species = list(self.species)
        
        for s, c in zip(self.species, range(len(self.species))):
            self.mapping[s] = int(np.argwhere(self.species)[c][0])

        self.temperature = temperature
        self.time_step = time_step
        self.step_skip = step_skip
        self.smoothing = smoothing if smoothing else 'best_fit'
        self.kbT = 8.617333262e-5*temperature * 10 #eV/atom * 10 to convert A0^2/fs to cm^2/s
        self.times = time_step*np.linspace(0, len(self.structures)*self.step_skip, int(len(self.structures)))
        self.index = []
        self.positions = []
        self.msds = None
        self.L_tensor = np.zeros((len(self.species), len(self.species)))      
        self.L_tensor_self = np.zeros((len(self.species), len(self.species)))
        self.L_tensor_dis = np.zeros((len(self.species), len(self.species)))
        self.diffusivity = {}

        for specie in self.species:
            self.index.append(find_species(specie, self.structures))
            self.positions.append(prep_positions(self.index[-1], self.structures))

        self.volume = np.mean([s.volume for s in self.structures]) * 1e-24 #convert to cm^3
        self.compute_L_tensor()
    
    def compute_L_tensor(self, start_step: int = None, end_step: int = None):
        """
        Computes the transport coefficients for a given species.
        :param start: int, start time index for fitting
        :param end: int, end time index for fitting
        :return: L_tensor: array[float], transport coefficients
        """
        if start_step is None:
            start_step = int(len(self.structures)/10)
        if end_step is None:
            end_step = int(9/10*len(self.structures))
        self.L_tensor = np.zeros((len(self.species), len(self.species)))
        self.L_tensor_self = np.zeros((len(self.species), len(self.species)))
        self.msds = []
        self.msd_map = dict()
        self.fit_dicts = np.zeros((len(self.species), len(self.species)), dtype=dict)
        self.fit_dicts_self = np.zeros((len(self.species), len(self.species)), dtype=dict)

        c = 0
        vals = list(self.mapping.values())
        for i in range(len(vals)):
            for j in range(len(vals)):
                if i != j:
                    self.msds.append(self.compute_all_Lij_pairs(self.positions[i], self.positions[j], self.volume))
                    self.msd_map[(vals[i], vals[j])] = c
                    self.L_tensor[j, j], self.fit_dicts[j, j] = fit_data(f=self.msds[-1][2], start=start_step, end=end_step, times=self.times, smoothing=self.smoothing)
                    self.L_tensor[i, j], self.fit_dicts[i, j] = fit_data(f=self.msds[-1][4], start=start_step, end=end_step, times=self.times, smoothing=self.smoothing)
                    self.L_tensor[j, i] = self.L_tensor[i, j]
                    self.fit_dicts[j, i] = self.fit_dicts[i, j]
                    self.L_tensor_self[j, j], self.fit_dicts_self[j, j] = fit_data(f=self.msds[-1][3], start=start_step, end=end_step, times=self.times, smoothing=self.smoothing)
                    c+=1
        self.L_tensor_dis = self.L_tensor - self.L_tensor_self

    def calc_Lii_self(self, atom_positions):
        """ 
        Calculates the "MSD" for the self component for a diagonal transport coefficient (L^{ii}).
        :param atom_positions: array[float,float,float], position of each atom over time.
        Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively.
        :param times: array[float], times at which position data was collected in the simulation
        :return msd: array[float], "MSD" corresponding to the L^{ii}_{self} transport 
        coefficient at each time
        """
        Lii_self = np.zeros(len(self.times))
        n_atoms = np.shape(atom_positions)[1]
        for atom_num in (range(n_atoms)):
            r = atom_positions[:,atom_num, :]
            msd_temp = msd_fft(np.array(r))
            Lii_self += msd_temp
        msd = np.array(Lii_self)
        return msd

    def calc_Lii(self, atom_positions):
        """ 
        Calculates the "MSD" for the diagonal transport coefficient L^{ii}. 
        :param atom_positions: array[float,float,float], position of each atom over time.
        Indices correspond to time, ion index, and spatial dimension (x,y,z), respectively.
        :param times: array[float], times at which position data was collected in the simulation
        :return msd: array[float], "MSD" corresponding to the L^{ii} transport 
        coefficient at each time
        """
        r_sum = np.sum(atom_positions, axis = 1)
        msd = msd_fft(r_sum)
        return np.array(msd)

    def calc_Lij(self, cation_positions, anion_positions):
        """
        Calculates the "MSD" for the off-diagonal transport coefficient L^{ij}, i \neq j.
        :param cation_positions, anion_positions: array[float,float,float], position of each 
        atom (anion or cation, respectively) over time. Indices correspond to time, ion index,
        and spatial dimension (x,y,z), respectively.
        :param times: array[float], times at which position data was collected in the simulation
        :return msd: array[float], "MSD" corresponding to the L^{ij} transport coefficient at 
        each time.
        """
        r_cat = np.sum(cation_positions, axis = 1)
        r_an = np.sum(anion_positions, axis = 1)
        msd = msd_fft_cross(np.array(r_cat),np.array(r_an))
        return np.array(msd)   
    
    def compute_all_Lij_pairs(self, positions_1, positions_2, volume):
        """
        Computes the "MSDs" for all transport coefficients.
        :param cation_positions, anion_positions: array[float,float,float], position of each 
        atom (anion or cation, respectively) over time. Indices correspond to time, ion index,
        and spatial dimension (x,y,z), respectively.
        :param times: array[float], times at which position data was collected in the simulation
        :param volume: float, volume of simulation box
        :return msds_all: list[array[float]], the "MSDs" corresponding to each transport coefficient,
        """
        msd_self_1 = self.calc_Lii_self(positions_1)/6.0/self.kbT/volume
        msd_self_2 =  self.calc_Lii_self(positions_2)/6.0/self.kbT/volume
        msd_1 = self.calc_Lii(positions_1)/6.0/self.kbT/volume
        msd_2 = self.calc_Lii(positions_2)/6.0/self.kbT/volume
        msd_distinct_12 = self.calc_Lij(positions_1, positions_2)/6.0/self.kbT/volume
        msds_all = [msd_1, msd_self_1, msd_2, msd_self_2, msd_distinct_12]
        return msds_all
    
    def refit_Lij(self, i, j, start_step, end_step, smoothing = None):
        """
        Refits the transport coefficients for a given species. Useful to manually check the fits.
        :param i, j: int, indices of species to refit
        :param start: int, start time index for fitting
        :param end: int, end time index for fitting
        :return: L_tensor: array[float], transport coefficients
        """
        if smoothing is None:
            smoothing = self.smoothing
        if i == j:
            self.L_tensor[j, j], self.fit_dicts[j, j] = fit_data(self.msds[self.msd_map[(i, j)]][2], start_step, end_step, self.times, smoothing=smoothing)
        else:
            self.L_tensor[i, j], self.fit_dicts[i, j] = fit_data(self.msds[self.msd_map[(i, j)]][4], start_step, end_step, self.times, smoothing=smoothing)
        self.L_tensor[j, i] = self.L_tensor[i, j]
        self.fit_dicts[j, i] = self.fit_dicts[i, j]
        self.L_tensor_dis = self.L_tensor - self.L_tensor_self
        
    def compute_D_from_msd(self, specie: str, smoothed: bool = False):
        """
        Computes the diffusion coefficient from the mean squared displacement.
        :param specie: str, species for which to compute the diffusion coefficient
        :param smoothed: bool, whether to smooth the MSD data
        :return D: float, diffusion coefficient
        """
        diff_analyzer = DiffusionAnalyzer.from_structures(self.structures, specie, self.temperature, self.time_step, self.step_skip, smoothed=smoothed)
        return diff_analyzer.diffusivity, diff_analyzer.msd
        
    @classmethod
    def from_xdatcar(cls, xdatcar: Union[str, Xdatcar], temperature: float, species: list[str] = None, time_step : float = 2, step_skip : int = 1, smoothing : str = None):
        """
        Initialize the OnsagerTransport object from a VASP XDATCAR file.
        :param Xdatcar: str, path to XDATCAR file
        :param temperature: float, simulation temperature
        :param species: list[str], list of species to fit
        :param time_step: float, time step of simulation
        :param step_skip: int, number of steps to skip in the simulation
        :param smoothing: str, type of smoothing to apply to the data
        :return: OnsagerTransport object
        """
        if isinstance(xdatcar, str):
            xdatcar = Xdatcar(xdatcar)
        return cls(xdatcar.structures, temperature, species, time_step, step_skip, smoothing)

    @classmethod
    def from_trajectory(cls, 
                        trajectory: Trajectory, 
                        temperature: float, 
                        species: list[str] = None, 
                        smoothing : str = None
                        ):
        """
        Initialize the OnsagerTransport object from a pymatgen Trajectory object.
        :param trajectory: Trajectory, pymatgen Trajectory object
        :param temperature: float, simulation temperature
        :param species: list[str], list of species to fit
        :param smoothing: str, type of smoothing to apply to the data
        :return: OnsagerTransport object
        """
        structures = [trajectory.get_structure(i) for i in range(len(trajectory))]
        return cls(structures, temperature, species, trajectory.time_step, trajectory.step_skip, smoothing)
    
    @classmethod
    def from_lammps_dump(cls, 
                         dump_file: str, 
                         temperature: float, 
                         species: list[str] = None, 
                         time_step : float = 2, 
                         step_skip : int = 1, 
                         smoothing : str = None,
                         skip_extra : int = 1
                         ):
        """
        Initialize the OnsagerTransport object from a LAMMPS dump file.
        :param dump_file: str, path to LAMMPS dump file
        :param temperature: float, simulation temperature
        :param species: list[str], list of species to fit
        :param time_step: float, time step of simulation
        :param step_skip: int, number of steps to skip in the simulation
        :param smoothing: str, type of smoothing to apply to the data
        :return: OnsagerTransport object
        """
        ase_trajectory = read(dump_file, format="lammps-dump-text", index=":")
        structures = [AseAtomsAdaptor.get_structure(frame) for frame in ase_trajectory[::skip_extra]]
        if species:
            if all(structures[0].composition.get_el_amt_dict().keys()) not in species:  # if species are not in the structure, substitute them in the order passed into the function
                for id, frame in enumerate(structures):
                    for specie, f_specie in zip(species, frame.composition.get_el_amt_dict().keys()):
                        structures[id][f_specie] = specie
        return cls(structures, temperature, species, time_step, step_skip, smoothing)

        
    
    