"""
Coordination environment analyzer: Radial Distribution Function and cage time analyzer.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import ceil

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..io.trajectory import TrajectoryData
from ..schemas.coordination import CageResult, RDFResult
from ..utils.analyzers.coordination.neighbors import neighbor_list
from .base import BaseAnalyzer

#: Key used for the all-species total RDF in ``CoordinationAnalyzer.rdfs``.
TOTAL_RDF_KEY = "total"


def rdf_pair_key(s1: str, s2: str) -> str:
    """Canonical key for an (s1, s2) RDF pair — alphabetically sorted."""
    a, b = sorted([s1, s2])
    return f"{a}-{b}"


class CoordinationAnalyzer(BaseAnalyzer):
    """
    Radial distribution function and cage time analyzer for TrajectoryData.

    The RDF follows the RadialDistributionFunctionFast algorithm from
    pymatgen.analysis.diffusion.aimd.rdf: distances are collected via
    primitive_neighbor_list, binned into a histogram, and normalized by the
    ideal-gas reference density and the spherical-shell volume. Smoothing uses
    scipy.ndimage.gaussian_filter1d as in the original code.

    After ``analyze()`` is called, ``rdfs`` holds a pre-computed ``RDFResult``
    for every unique species pair (including self-pairs) and one entry keyed
    ``"total"`` that treats all species as both reference and neighbor. The 
    ``"total"`` can be used to compute the static structure factor S(k) via 
    a Fourier transform (not implemented here).

    Usage::

        an = CoordinationAnalyzer(traj, rmax=8.0, ngrid=201, sigma=0.1)
        an.analyze()
        li_o  = an.rdfs["Li-O"]        # RDFResult
        total = an.rdfs["total"]
        cage  = an.get_cage_correlation("Li", "O", cutoff=2.8)
    """

    def __init__(
        self,
        trajectory: TrajectoryData,
        rmax: float = 8.0,
        ngrid: int = 201,
        sigma: float = 0.1,
        frame_sample: int | None = None,
    ) -> None:
        """
        Args:
            trajectory: TrajectoryData to analyze.
            rmax: Maximum radial distance for the RDF grid (Å).
            ngrid: Number of radial grid points.
            sigma: Gaussian smoothing width (Å). Set to 0 for no smoothing.
            frame_sample: If given, use this many evenly-spaced frames instead
                of all frames. Useful for long trajectories.
        """
        super().__init__(trajectory, name="coordination_analyzer")

        if ngrid < 2:
            raise ValueError("ngrid must be >= 2")
        if rmax <= 0:
            raise ValueError("rmax must be positive")

        self.rmax = float(rmax)
        self.ngrid = int(ngrid)
        self.sigma = float(sigma)

        self.rmin = 0.0
        self.dr = self.rmax / (self.ngrid - 1)
        self.r = np.linspace(self.rmin, self.rmax, self.ngrid)

        # Spherical-shell volumes — used for normalization (as in pmg RDFFast)
        self._volumes = 4.0 * np.pi * self.r**2 * self.dr
        self._volumes[self._volumes < 1e-8] = 1e8  # avoid /0 at r=0

        # sigma converted to grid units for gaussian_filter1d
        self._sigma_grid = ceil(self.sigma / self.dr) if self.sigma > 1e-8 else 0

        # Frame selection
        all_f = list(range(trajectory.n_frames))
        if frame_sample is not None and frame_sample < len(all_f):
            idx = np.round(np.linspace(0, len(all_f) - 1, frame_sample)).astype(int)
            self.frame_indices: list[int] = [all_f[i] for i in idx]
        else:
            self.frame_indices = all_f

        # Species info (as plain strings, avoiding heavy pmg wrappers)
        self._species_str: np.ndarray = np.array(
            [str(s) for s in trajectory.species], dtype=str
        )
        unique: list[str] = list(dict.fromkeys(self._species_str))
        self.unique_species = unique
        self._natoms: dict[str, int] = {s: int(np.sum(self._species_str == s)) for s in unique}

        # Populated by analyze()
        self._center_elements: list[np.ndarray] = []
        self._neighbor_elements: list[np.ndarray] = []
        self._distances: list[np.ndarray] = []
        self._density: list[dict[str, float]] = []
        self._analyzed = False

        #: Pre-computed RDFs keyed by ``rdf_pair_key(s1, s2)`` and ``"total"``.
        #: Populated after ``analyze()`` is called.
        self.rdfs: dict[str, RDFResult] = {}

    def analyze(self) -> None:
        """
        Build neighbor lists and compute RDFs for all species pairs plus total.

        After this call ``self.rdfs`` contains an ``RDFResult`` for every
        unique unordered pair (including self-pairs) and one keyed ``"total"``.
        """
        max_r = self.rmax + self.dr / 2.0  # small buffer, matches pmg RDFFast
        traj = self.trajectory

        self._center_elements = []
        self._neighbor_elements = []
        self._distances = []
        self._density = []

        for frame_idx in self.frame_indices:
            pos = traj.positions[frame_idx]             # (n_atoms, 3)
            lattice = traj.lattice_at_frame(frame_idx)  # (3, 3)
            volume = float(abs(np.linalg.det(lattice)))

            ci, ni, dists = neighbor_list(lattice, pos, max_r)

            self._center_elements.append(self._species_str[ci])
            self._neighbor_elements.append(self._species_str[ni])
            self._distances.append(dists)

            density = {s: self._natoms[s] / volume for s in self.unique_species}
            self._density.append(density)

        self._analyzed = True

        # Pre-compute RDFs for all unique unordered pairs and total
        self.rdfs = {}
        for s1, s2 in combinations_with_replacement(self.unique_species, 2):
            key = rdf_pair_key(s1, s2)
            self.rdfs[key] = self.get_rdf(s1, s2)
        self.rdfs[TOTAL_RDF_KEY] = self.get_rdf(self.unique_species, self.unique_species)

    def _get_one_rdf(
        self,
        ref_species: list[str],
        species: list[str],
        frame_index: int,
    ) -> np.ndarray:
        """
        RDF for a single frame — mirrors RadialDistributionFunctionFast.get_one_rdf.

        Args:
            ref_species: center species list.
            species: neighbor species list.
            frame_index: index into self.frame_indices (not raw trajectory index).
        """
        ce = self._center_elements[frame_index]
        ne = self._neighbor_elements[frame_index]
        d = self._distances[frame_index]

        mask = (
            np.isin(ce, ref_species)
            & np.isin(ne, species)
            & (d >= self.rmin - self.dr / 2.0)
            & (d <= self.rmax + self.dr / 2.0)
            & (d > 1e-8)
        )

        density = sum(self._density[frame_index][s] for s in species)
        natoms_ref = sum(self._natoms[s] for s in ref_species)
        filtered_d = d[mask]

        # Bin distances — identical to RadialDistributionFunctionFast._dist_to_counts
        counts = np.zeros(self.ngrid)
        bin_idx = np.array(
            np.floor((filtered_d - self.rmin + 0.5 * self.dr) / self.dr), dtype=int
        )
        valid = (bin_idx >= 0) & (bin_idx < self.ngrid)
        unique_bins, bin_counts = np.unique(bin_idx[valid], return_counts=True)
        counts[unique_bins] = bin_counts

        rdf = counts / density / self._volumes / natoms_ref
        if self._sigma_grid > 1e-8:
            rdf = gaussian_filter1d(rdf, self._sigma_grid)
        return rdf

    def get_rdf(
        self,
        ref_species: str | list[str],
        species: str | list[str],
        is_average: bool = True,
    ) -> RDFResult:
        """
        Radial distribution function g(r) for a species pair.

        Mirrors RadialDistributionFunctionFast.get_rdf interface.

        Args:
            ref_species: Center species (e.g. "Li" or ["Li"]).
            species: Neighbor species (e.g. "O" or ["O", "S"]).
            is_average: If True (default), return the frame-averaged g(r).

        Returns:
            RDFResult with .r, .rdf, .rdf_per_frame, .coordination_number,
            .peak_r, .peak_rdf.
        """
        if not self._analyzed:
            raise RuntimeError("Call analyze() before get_rdf()")

        if isinstance(ref_species, str):
            ref_species = [ref_species]
        if isinstance(species, str):
            species = [species]

        all_rdfs = np.stack(
            [self._get_one_rdf(ref_species, species, i) for i in range(len(self.frame_indices))],
            axis=0,
        )  # (n_frames, ngrid)

        rdf_mean = np.mean(all_rdfs, axis=0) if is_average else all_rdfs[0]

        # Running coordination number (average density over frames)
        avg_density = np.mean(
            [sum(self._density[i][s] for s in species) for i in range(len(self.frame_indices))]
        )
        cn = np.cumsum(rdf_mean * avg_density * 4.0 * np.pi * self.r**2 * self.dr)

        peak_idx = find_peaks(rdf_mean)[0]
        return RDFResult(
            r=self.r.copy(),
            rdf=rdf_mean,
            rdf_per_frame=all_rdfs,
            coordination_number=cn,
            peak_r=[float(self.r[i]) for i in peak_idx],
            peak_rdf=[float(rdf_mean[i]) for i in peak_idx],
            ref_species=ref_species,
            species=species,
        )

    def get_coordination_number(
        self,
        ref_species: str | list[str],
        species: str | list[str],
        is_average: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Running coordination number ⟨CN(r)⟩.

        Returns:
            (r, cn) where cn[i] = average number of `species` atoms within
            radius r[i] of a `ref_species` atom.
        """
        result = self.get_rdf(ref_species, species, is_average=is_average)
        return result.r, result.coordination_number

    def get_cage_correlation(
        self,
        mobile_species: str | list[str],
        cage_species: str | list[str],
        cutoff: float,
        max_lag: int | None = None,
    ) -> CageResult:
        """
        Cage correlation function C(τ) and integral cage time.

        For each mobile atom i, define its cage at time t as the set of
        cage_species atoms within `cutoff` Å.  The per-atom correlation is:

            C_i(τ) = |cage_i(0) ∩ cage_i(τ)| / |cage_i(0)|

        and the ensemble average C(τ) = ⟨C_i(τ)⟩ is computed with a
        sliding origin window (same averaging as an MSD calculation).

        The cage time is the integral:
            τ_cage = ∫₀^{T/2} C(τ) dτ

        where T is the trajectory length (or max_lag frames).

        Args:
            mobile_species: Species of mobile ions (e.g. "Li").
            cage_species: Species forming the cage (e.g. "O").
            cutoff: Cutoff radius defining the cage (Å).  Typically the
                position of the first minimum in the mobile–cage RDF.
            max_lag: Maximum lag in frames. Defaults to n_frames // 2.

        Returns:
            CageResult with .lag_times (fs), .cage_correlation, .cage_time,
            .cage_time_1e.
        """
        traj = self.trajectory

        if isinstance(mobile_species, str):
            mobile_species = [mobile_species]
        if isinstance(cage_species, str):
            cage_species = [cage_species]

        mobile_idx = np.where(np.isin(self._species_str, mobile_species))[0]
        cage_idx = np.where(np.isin(self._species_str, cage_species))[0]

        if len(mobile_idx) == 0:
            raise ValueError(f"No atoms found for mobile_species={mobile_species}")
        if len(cage_idx) == 0:
            raise ValueError(f"No atoms found for cage_species={cage_species}")

        n_mobile = len(mobile_idx)
        n_cage = len(cage_idx)
        n_frames = traj.n_frames

        if max_lag is None:
            max_lag = n_frames // 2
        max_lag = min(max_lag, n_frames - 1)

        # Build binary neighbor matrix: neigh[frame, i_mobile, j_cage] = bool
        # Shape: (n_frames, n_mobile, n_cage). float32 keeps memory reasonable.
        neigh = np.zeros((n_frames, n_mobile, n_cage), dtype=np.float32)

        for t in range(n_frames):
            pos_all = traj.positions[t]
            lat = traj.lattice_at_frame(t)

            # Stack mobile + cage positions so primitive_neighbor_list indices align
            combined = np.vstack([pos_all[mobile_idx], pos_all[cage_idx]])
            ci, ni, _ = neighbor_list(lat, combined, cutoff)
            m_mask = (ci < n_mobile) & (ni >= n_mobile)
            neigh[t, ci[m_mask], ni[m_mask] - n_mobile] = 1.0

        time_step = float(traj.metadata.get("time_step", 2.0))
        step_skip = int(traj.metadata.get("step_skip", 1))
        dt_fs = time_step * step_skip

        lag_times = np.arange(max_lag + 1, dtype=np.float64) * dt_fs
        cage_corr = np.zeros(max_lag + 1, dtype=np.float64)

        cage_size = neigh.sum(axis=2)                                  # (n_frames, n_mobile)
        cage_size_safe = np.where(cage_size > 0, cage_size, np.inf)

        for tau in range(max_lag + 1):
            overlap = (neigh[:n_frames - tau] * neigh[tau:]).sum(axis=2)
            frac = overlap / cage_size_safe[:n_frames - tau]
            has_cage = cage_size[:n_frames - tau] > 0
            cage_corr[tau] = float(frac[has_cage].mean()) if has_cage.any() else 0.0

        cage_time = float(np.trapz(cage_corr, lag_times))

        target = cage_corr[0] / np.e
        cage_time_1e = float("nan")
        crossings = np.where((cage_corr[:-1] >= target) & (cage_corr[1:] < target))[0]
        if len(crossings) > 0:
            t0, t1 = lag_times[crossings[0]], lag_times[crossings[0] + 1]
            c0, c1 = cage_corr[crossings[0]], cage_corr[crossings[0] + 1]
            cage_time_1e = float(t0 + (target - c0) / (c1 - c0) * (t1 - t0))

        return CageResult(
            lag_times=lag_times,
            cage_correlation=cage_corr,
            cage_time=cage_time,
            cage_time_1e=cage_time_1e,
            mobile_species=mobile_species,
            cage_species=cage_species,
            cutoff=cutoff,
        )
