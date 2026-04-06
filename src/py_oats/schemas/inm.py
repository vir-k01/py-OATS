"""Schemas for Instantaneous Normal Modes (INM) analysis results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from monty.json import MontyDecoder
from pydantic import BaseModel, Field

from py_oats.analyzers.inm import INMAnalyzer


class INMModeSet(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    frame_index: int = Field(..., description="trajectory frame index")
    time_fs: float = Field(..., description="time at frame (fs)")
    eigenvalues: np.ndarray = Field(..., description="eigenvalues λ of mass-weighted Hessian (3N,)")
    eigenvectors: np.ndarray = Field(..., description="eigenvectors (3N,3N) columns are modes")
    omega: np.ndarray = Field(..., description="signed sqrt(|λ|) in sqrt(eV/(Å^2 amu)) (3N,)")
    omega_ps: np.ndarray = Field(..., description="signed ω in ps^-1 under ASE unit convention (3N,)")
    unstable_mask: np.ndarray = Field(..., description="boolean mask for unstable modes (λ<0) (3N,)")
    elements: List[str] = Field(..., description="unique element symbols in order")
    element_participation: np.ndarray = Field(
        ...,
        description="mode participation by element w_E(k); shape (3N, n_elements)",
    )


class INMDOS(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    frame_index: Optional[int] = Field(None, description="frame index (None for trajectory-average)")
    time_fs: Optional[float] = Field(None, description="time at frame (fs) (None for trajectory-average)")
    omega_bin_edges_ps: np.ndarray = Field(..., description="histogram bin edges (ps^-1)")
    rho_total: np.ndarray = Field(..., description="DOS histogram for all modes (per bin)")
    rho_stable: np.ndarray = Field(..., description="DOS histogram for stable modes (λ>=0)")
    rho_unstable: np.ndarray = Field(..., description="DOS histogram for unstable modes (λ<0)")
    element_rho_total: Dict[str, np.ndarray] = Field(
        ..., description="element-resolved DOS weighted by participation w_E(k)"
    )
    element_rho_stable: Dict[str, np.ndarray] = Field(
        ..., description="element-resolved DOS for stable modes weighted by participation"
    )
    element_rho_unstable: Dict[str, np.ndarray] = Field(
        ..., description="element-resolved DOS for unstable modes weighted by participation"
    )


class INMDoc(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    composition: Optional[str] = Field(None, description="composition / reduced formula if available")
    num_atoms: int = Field(..., description="number of atoms")
    frame_indices: List[int] = Field(..., description="sampled frames")
    times_fs: List[float] = Field(..., description="times for sampled frames (fs)")
    compute_frequency: int = Field(..., description="frame sampling frequency used")
    finite_difference_step: float = Field(..., description="finite difference step (Å)")
    calculator: Optional[str] = Field(None, description="ASE calculator identifier")

    modes: List[INMModeSet] = Field(..., description="per-frame INM eigenpairs")
    dos: List[INMDOS] = Field(..., description="per-frame and trajectory-averaged DOS")

    f_unstable: List[float] = Field(..., description="fraction of unstable modes per sampled frame")
    f_unstable_mean: float = Field(..., description="mean fraction of unstable modes across sampled frames")

    f_unstable_by_element: List[Dict[str, float]] = Field(
        ..., description="per-frame participation-weighted unstable fraction by element"
    )
    f_unstable_by_element_mean: Dict[str, float] = Field(
        ..., description="mean participation-weighted unstable fraction by element"
    )

    def as_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> "INMDoc":
        decoded = MontyDecoder().process_decoded(d)
        if isinstance(decoded, cls):
            return decoded
        return cls.model_validate(decoded)

    @classmethod
    def from_analyzer(
        cls,
        analyzer: INMAnalyzer,
        dos_bins: int = 400,
        dos_omega_ps_range: tuple[float, float] | None = None,
    ) -> "INMDoc":
        if not analyzer.frames:
            raise ValueError("Analyzer has no results; call analyze() first.")

        # build per-frame mode sets
        modes: List[INMModeSet] = []
        for fr in analyzer.frames:
            modes.append(
                INMModeSet(
                    frame_index=fr.frame_index,
                    time_fs=fr.time_fs,
                    eigenvalues=fr.eigenvalues,
                    eigenvectors=fr.eigenvectors,
                    omega=fr.omega,
                    omega_ps=fr.omega_ps,
                    unstable_mask=fr.unstable_mask,
                    elements=list(fr.elements),
                    element_participation=fr.element_participation,
                )
            )

        # DOS binning across all sampled frames
        omega_all = np.concatenate([fr.omega_ps for fr in analyzer.frames])
        if dos_omega_ps_range is None:
            lo = float(np.min(omega_all))
            hi = float(np.max(omega_all))
            if lo == hi:
                lo -= 1.0
                hi += 1.0
            dos_omega_ps_range = (lo, hi)
        edges = np.linspace(dos_omega_ps_range[0], dos_omega_ps_range[1], dos_bins + 1)

        def _hist(vals: np.ndarray) -> np.ndarray:
            h, _ = np.histogram(vals, bins=edges)
            return h.astype(np.float64) / float(analyzer.trajectory.n_atoms * 3)

        def _hist_weighted(vals: np.ndarray, weights: np.ndarray) -> np.ndarray:
            h, _ = np.histogram(vals, bins=edges, weights=weights)
            return h.astype(np.float64) / float(analyzer.trajectory.n_atoms * 3)

        dos: List[INMDOS] = []
        for fr in analyzer.frames:
            stable_vals = fr.omega_ps[~fr.unstable_mask]
            unstable_vals = fr.omega_ps[fr.unstable_mask]

            element_total: Dict[str, np.ndarray] = {}
            element_stable: Dict[str, np.ndarray] = {}
            element_unstable: Dict[str, np.ndarray] = {}
            for e_idx, el in enumerate(fr.elements):
                w = fr.element_participation[:, e_idx]
                element_total[el] = _hist_weighted(fr.omega_ps, w)
                element_stable[el] = _hist_weighted(fr.omega_ps[~fr.unstable_mask], w[~fr.unstable_mask])
                element_unstable[el] = _hist_weighted(fr.omega_ps[fr.unstable_mask], w[fr.unstable_mask])
            dos.append(
                INMDOS(
                    frame_index=fr.frame_index,
                    time_fs=fr.time_fs,
                    omega_bin_edges_ps=edges,
                    rho_total=_hist(fr.omega_ps),
                    rho_stable=_hist(stable_vals),
                    rho_unstable=_hist(unstable_vals),
                    element_rho_total=element_total,
                    element_rho_stable=element_stable,
                    element_rho_unstable=element_unstable,
                )
            )

        # trajectory-averaged DOS (concatenate)
        unstable_mask_all = np.concatenate([fr.unstable_mask for fr in analyzer.frames])
        unstable_all = omega_all[unstable_mask_all]
        stable_all = omega_all[~unstable_mask_all]

        elements0 = analyzer.frames[0].elements
        element_total_all: Dict[str, np.ndarray] = {}
        element_stable_all: Dict[str, np.ndarray] = {}
        element_unstable_all: Dict[str, np.ndarray] = {}
        # concatenate element participation for each element in the same order
        part_all = np.concatenate([fr.element_participation for fr in analyzer.frames], axis=0)  # (n_frames*3N, n_el)
        for e_idx, el in enumerate(elements0):
            w_all = part_all[:, e_idx]
            element_total_all[el] = _hist_weighted(omega_all, w_all)
            element_stable_all[el] = _hist_weighted(omega_all[~unstable_mask_all], w_all[~unstable_mask_all])
            element_unstable_all[el] = _hist_weighted(omega_all[unstable_mask_all], w_all[unstable_mask_all])
        dos.append(
            INMDOS(
                frame_index=None,
                time_fs=None,
                omega_bin_edges_ps=edges,
                rho_total=_hist(omega_all),
                rho_stable=_hist(stable_all),
                rho_unstable=_hist(unstable_all),
                element_rho_total=element_total_all,
                element_rho_stable=element_stable_all,
                element_rho_unstable=element_unstable_all,
            )
        )

        # f_u per frame
        f_unstable = [float(np.mean(fr.unstable_mask)) for fr in analyzer.frames]
        f_unstable_mean = float(np.mean(f_unstable))

        # participation-weighted unstable fraction by element
        f_unstable_by_element: List[Dict[str, float]] = []
        for fr in analyzer.frames:
            per_el: Dict[str, float] = {}
            for e_idx, el in enumerate(fr.elements):
                w = fr.element_participation[:, e_idx]  # (3N,)
                denom = float(np.sum(w))
                num = float(np.sum(w[fr.unstable_mask]))
                per_el[el] = num / denom if denom > 0 else float("nan")
            f_unstable_by_element.append(per_el)

        # mean by element across frames (average the per-frame ratios)
        mean_by_el: Dict[str, float] = {}
        for el in analyzer.frames[0].elements:
            vals = [d.get(el, float("nan")) for d in f_unstable_by_element]
            mean_by_el[el] = float(np.nanmean(vals))

        calc_name = type(analyzer.calculator).__name__ if analyzer.calculator is not None else None

        return cls(
            composition=analyzer.trajectory.composition,
            num_atoms=analyzer.trajectory.n_atoms,
            frame_indices=list(analyzer.frame_indices),
            times_fs=list(analyzer.times_fs),
            compute_frequency=int(analyzer.compute_frequency or 0),
            finite_difference_step=float(analyzer.finite_difference_step or 0.0),
            calculator=calc_name,
            modes=modes,
            dos=dos,
            f_unstable=f_unstable,
            f_unstable_mean=f_unstable_mean,
            f_unstable_by_element=f_unstable_by_element,
            f_unstable_by_element_mean=mean_by_el,
        )

