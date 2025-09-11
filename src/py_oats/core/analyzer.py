from pymatgen.core import Structure, Element, Species
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.vasp import Xdatcar
from ase.io import read
from ase.io import Trajectory as AseTrajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
from typing import List, Union, Dict
import numpy as np
from ..utils.FFT import msd_fft, msd_fft_cross
from ..utils.fitting import fit_data
from ..utils.site_processing import find_species, prep_positions
from ..utils.decorators import ChargeDecorator, NaiveChargeDecorator
from ..utils.charge import get_unique_charged_species
from ..utils.oxidation import assign_oxidation_states_by_magmoms

class OnsagerTransportAnalyzer():
    '''
    Class to fit the transport coefficients to the Onsager relations. Based on repo by Kara Fong (@kdfong).
    Provide a list of pymatgen structure taken from an NVT MD run, alongwith simulation temperature and species to fit the transport coefficients.
    Pairwise MSDs can be plotted using the plot_msd method. The transport coefficients can be accessed using the L_tensor attribute. 
    The order in which the transport coefficients can be accessed by calling the OnsagerFitting.species attribute.
    Smoothing can be specificed: 'blockavg' for block averaging, 'best_fit' for best fit, 'None' for no smoothing. The default is best_fit.
    
    Oxidation states can be specified to distinguish between different charge states of the same element.
    For example, {"Li": [1], "Ti": [2, 4], "O": [-2]} will create separate transport coefficients for Li+1, Ti+2, Ti+4, and O-2.
    '''
    def __init__(self, 
                 structures: list[Structure], 
                 temperature: float, 
                 species: List[Union[str, Element]] = None, 
                 oxidation_states: Dict[str, List[int]] = None,
                 charge_decorator: ChargeDecorator | None = NaiveChargeDecorator,
                 time_step: float = 2, 
                 step_skip: int = 1, 
                 decorate_freq : int = -1,
                 smoothing: str = None
                 ):
        self.structures = structures
        self.first_structure = structures[0].copy()
        self.mapping, self.inv_mapping = dict(), dict()
        self.composition = self.first_structure.composition.reduced_formula
        
        # Handle oxidation states if provided
        if oxidation_states:
            # Make accessible early for downstream use
            self.oxidation_states = oxidation_states
            # Check if structures have magnetic moments
            has_magmoms = any(s.site_properties.get('final_magmom', None) is not None for s in self.structures) | any(s.site_properties.get('magmoms', None) is not None for s in self.structures)
            
            if has_magmoms:
                # Use magnetic moments to determine oxidation states
                print('Using magnetic moments to determine oxidation states')
                from ..utils.charge import _initial_boundaries   
                possible_species = [elem for elem, states in oxidation_states.items() if len(states) > 1]
                
                self.decorated_structures = []
                for structure in self.structures:
                    magmoms = structure.site_properties['final_magmom'] if structure.site_properties.get('final_magmom', None) is not None else structure.site_properties['magmoms']
                    oxi_states = assign_oxidation_states_by_magmoms(
                        structure=structure,
                        oxidation_states=self.oxidation_states,
                        magmoms=magmoms,
                        adaptive_boundaries=True,
                        bounds=_initial_boundaries,
                    )
                    
                    decorated_structure = structure.add_oxidation_state_by_site(oxi_states)
                    self.decorated_structures.append(decorated_structure)
            else:
                # Decorate all structures with oxidation states obtained from charge decorator
                if not charge_decorator:
                    charge_decorator = NaiveChargeDecorator(oxidation_states=oxidation_states)
                else:
                    charge_decorator.oxidation_states = oxidation_states
                if decorate_freq > 0:
                    self.decorated_structures = [charge_decorator.decorate_structure(s) for s in self.structures]
                else:
                    last_structure = charge_decorator.decorate_structure(self.structures[-1])
                    predicted_oxidation_states = [site.specie.oxi_state for site in last_structure]
                    self.decorated_structures = [s.add_oxidation_state_by_site(predicted_oxidation_states) for s in self.structures]
            
            self.first_structure = self.decorated_structures[0].copy()
            
            # Get unique charged species from oxidation states
            self.species = get_unique_charged_species(oxidation_states)
            print("Got the decorated structures!")
        else:
            # Original behavior without oxidation states
            self.decorated_structures = self.structures
            if species:
                self.species = species
            else:
                self.species = set()
                for s in self.first_structure.species:
                    self.species.add(s.symbol)
                self.species = list(self.species)
            self.oxidation_states = None
        
        for s, c in zip(self.species, range(len(self.species))):
            self.mapping[s] = c
            self.inv_mapping[c] = s
            
        self.temperature = temperature
        self.time_step = time_step
        self.step_skip = step_skip
        self.smoothing = smoothing if smoothing else 'best_fit'
        self.kbT = 8.617333262e-5*temperature * 10 #eV/atom * 10 to convert A0^2/fs to cm^2/s
        self.times = time_step*np.linspace(0, len(self.structures)*self.step_skip, int(len(self.structures)))
        self.index = {}
        self.positions = {}
        self.msds = None
        self.L_tensor = np.zeros((len(self.species), len(self.species)))      
        self.L_tensor_self = np.zeros((len(self.species), len(self.species)))
        self.L_tensor_dis = np.zeros((len(self.species), len(self.species)))
        self.diffusivity = {}

        for specie in self.species:
            self.index[specie] = find_species(specie, self.first_structure)
            self.positions[specie] = prep_positions(self.index[specie], self.decorated_structures)
            
        self.species = [specie for specie in self.species if len(self.index[specie]) > 0]

        self.volume = np.mean([s.volume for s in self.decorated_structures]) * 1e-24 #convert to cm^3
        self.compute_L_tensor()
    
    def get_available_species(self) -> List[str]:
        """
        Returns a list of available species in the system.
        If oxidation states were specified, returns the charged species.
        Otherwise, returns the element symbols.
        
        Returns:
            List[str]: List of available species
        """
        return self.species.copy()
    
    def compute_L_tensor(self, start_step: int = None, end_step: int = None):
        """
        Computes the transport coefficients for a given species.
        :param start: int, start time index for fitting
        :param end: int, end time index for fitting
        :return: L_tensor: array[float], transport coefficients
        """
        if start_step is None:
            start_step = int(len(self.decorated_structures)/10)
        if end_step is None:
            end_step = int(9/10*len(self.decorated_structures))
        self.L_tensor = np.zeros((len(self.species), len(self.species)))
        self.L_tensor_self = np.zeros((len(self.species), len(self.species)))
        self.msds = {}
        #self.msd_map = dict()
        self.fit_dicts = np.zeros((len(self.species), len(self.species)), dtype=dict)
        self.fit_dicts_self = np.zeros((len(self.species), len(self.species)), dtype=dict)

        if len(self.species) == 0:
            print("Warning: No species found in the structure!")
            return
        
        self.mapping = {specie: i for i, specie in enumerate(self.species)}
        self.inv_mapping = {i: specie for i, specie in enumerate(self.species)}
        
        if len(self.species) == 1:
            specie = self.species[0]
            self.msds[(0, 0)] = self.compute_all_Lij_pairs(self.positions[specie], self.positions[specie], self.volume)
            self.L_tensor[0, 0], self.fit_dicts[0, 0] = fit_data(f=self.msds[(0, 0)][2], start=start_step, end=end_step, times=self.times, smoothing=self.smoothing)
            self.L_tensor_self[0, 0], self.fit_dicts_self[0, 0] = fit_data(f=self.msds[(0, 0)][3], start=start_step, end=end_step, times=self.times, smoothing=self.smoothing)
            self.L_tensor_dis[0, 0] = self.L_tensor[0, 0] - self.L_tensor_self[0, 0]
        
        else:
            for i, specie_i in enumerate(self.species):
                for j, specie_j in enumerate(self.species):
                    if len(self.index[specie_i]) == 0 or len(self.index[specie_j]) == 0:
                        continue
                    if i != j:
                        self.msds[(i, j)] = self.compute_all_Lij_pairs(self.positions[specie_i], self.positions[specie_j], self.volume) 
                        self.L_tensor[j, j], self.fit_dicts[j, j] = fit_data(f=self.msds[(i, j)][2], start=start_step, end=end_step, times=self.times, smoothing=self.smoothing)
                        self.L_tensor[i, j], self.fit_dicts[i, j] = fit_data(f=self.msds[(i, j)][4], start=start_step, end=end_step, times=self.times, smoothing=self.smoothing)
                        self.L_tensor[j, i] = self.L_tensor[i, j]
                        self.fit_dicts[j, i] = self.fit_dicts[i, j]
                        self.L_tensor_self[j, j], self.fit_dicts_self[j, j] = fit_data(f=self.msds[(i, j)][3], start=start_step, end=end_step, times=self.times, smoothing=self.smoothing)
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
            self.L_tensor[j, j], self.fit_dicts[j, j] = fit_data(self.msds[(self.inv_mapping[i], self.inv_mapping[j])][2], start_step, end_step, self.times, smoothing=smoothing)
        else:
            self.L_tensor[i, j], self.fit_dicts[i, j] = fit_data(self.msds[(self.inv_mapping[i], self.inv_mapping[j])][4], start_step, end_step, self.times, smoothing=smoothing)
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
        diff_analyzer = DiffusionAnalyzer.from_structures(self.decorated_structures, specie, self.temperature, self.time_step, self.step_skip, smoothed=smoothed)
        return diff_analyzer.diffusivity, diff_analyzer.msd
        
    @classmethod
    def from_xdatcar(cls, xdatcar: Union[str, Xdatcar], 
                     temperature: float, 
                     species: list[str] = None, 
                     oxidation_states: Dict[str, List[int]] = None, 
                     time_step : float = 2, 
                     step_skip : int = 1, 
                     smoothing : str = None,
                     charge_decorator: Union[ChargeDecorator, None] = NaiveChargeDecorator,
                     decorate_freq : int = -1
                     ):
        """
        Initialize the OnsagerTransport object from a VASP XDATCAR file.
        :param Xdatcar: str, path to XDATCAR file
        :param temperature: float, simulation temperature
        :param species: list[str], list of species to fit
        :param oxidation_states: dict, mapping of element symbols to oxidation states
        :param time_step: float, time step of simulation
        :param step_skip: int, number of steps to skip in the simulation
        :param smoothing: str, type of smoothing to apply to the data
        :return: OnsagerTransport object
        """
        if isinstance(xdatcar, str):
            xdatcar = Xdatcar(xdatcar)
        return cls(xdatcar.structures, temperature, species, oxidation_states, charge_decorator, time_step, step_skip, decorate_freq, smoothing)

    @classmethod
    def from_trajectory(cls, 
                        trajectory: Trajectory, 
                        temperature: float, 
                        species: list[str] = None, 
                        oxidation_states: Dict[str, List[int]] = None,
                        smoothing : str = None,
                        charge_decorator: Union[ChargeDecorator, None] = NaiveChargeDecorator,
                        decorate_freq : int = -1
                        ):
        """
        Initialize the OnsagerTransport object from a pymatgen Trajectory object.
        :param trajectory: Trajectory, pymatgen Trajectory object
        :param temperature: float, simulation temperature
        :param species: list[str], list of species to fit
        :param oxidation_states: dict, mapping of element symbols to oxidation states
        :param smoothing: str, type of smoothing to apply to the data
        :return: OnsagerTransport object
        """
        structures = [trajectory.get_structure(i) for i in range(len(trajectory))]
        return cls(structures, temperature, species, oxidation_states, charge_decorator, trajectory.time_step, trajectory.step_skip, decorate_freq, smoothing)
    
    @classmethod
    def from_lammps_dump(cls, 
                         dump_file: str, 
                         temperature: float, 
                         species: list[str] = None, 
                         oxidation_states: Dict[str, List[int]] = None,
                         time_step : float = 2, 
                         step_skip : int = 1, 
                         smoothing : str = None,
                         skip_extra : int = 1,
                         charge_decorator: Union[ChargeDecorator, None] = NaiveChargeDecorator,
                         decorate_freq : int = -1
                         ):
        """
        Initialize the OnsagerTransport object from a LAMMPS dump file.
        :param dump_file: str, path to LAMMPS dump file
        :param temperature: float, simulation temperature
        :param species: list[str], list of species to fit
        :param oxidation_states: dict, mapping of element symbols to oxidation states
        :param time_step: float, time step of simulation
        :param step_skip: int, number of steps to skip in the simulation
        :param smoothing: str, type of smoothing to apply to the data
        :return: OnsagerTransport object
        """
        ase_trajectory = read(dump_file, format="lammps-dump-text", index=f"::{skip_extra}")
        structures = [AseAtomsAdaptor.get_structure(frame) for frame in ase_trajectory]
        if species:
            if all(structures[0].composition.get_el_amt_dict().keys()) not in species:  # if species are not in the structure, substitute them in the order passed into the function
                for id, frame in enumerate(structures):
                    for specie, f_specie in zip(species, frame.composition.get_el_amt_dict().keys()):
                        structures[id][f_specie] = specie
        return cls(structures, temperature, species, oxidation_states, charge_decorator, time_step, step_skip, decorate_freq, smoothing)
    
    @classmethod
    def from_ase_trajectory(cls, 
                            trajectory: Union[AseTrajectory, str], 
                            temperature: float, 
                            species: list[str] = None, 
                            oxidation_states: Dict[str, List[int]] = None,
                            smoothing : str = None,
                            time_step : float = 2,
                            step_skip : int = 1,
                            skip_extra : int = 1,
                            charge_decorator: Union[ChargeDecorator, None] = NaiveChargeDecorator,
                            decorate_freq : int = -1
                            ):
        """
        Initialize the OnsagerTransport object from a ASE trajectory object.
        :param trajectory: AseTrajectory, ASE trajectory object
        :param temperature: float, simulation temperature
        :param species: list[str], list of species to fit
        :param oxidation_states: dict, mapping of element symbols to oxidation states
        :param smoothing: str, type of smoothing to apply to the data
        :return: OnsagerTransport object
        """
        if isinstance(trajectory, str):
            trajectory = read(trajectory, index=f"::{skip_extra}")
        else:
            trajectory = trajectory[skip_extra::step_skip]
        structures = [AseAtomsAdaptor.get_structure(frame) for frame in trajectory]
        try:
            for i,frame in enumerate(trajectory):
                structures[i].site_properties['magmoms'] = trajectory[i].get_magnetic_moments()
        except:
            pass            
        return cls(structures, temperature, species, oxidation_states, charge_decorator, time_step, step_skip, decorate_freq, smoothing)

        
    
    