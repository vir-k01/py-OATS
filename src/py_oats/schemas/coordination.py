"""Schema dataclasses for coordination environment analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..analyzers.coordination import CoordinationAnalyzer


@dataclass
class RDFResult:
    """Per-pair RDF result from CoordinationAnalyzer."""
    r: np.ndarray                   # radial grid (Å)
    rdf: np.ndarray                 # g(r) averaged over frames
    rdf_per_frame: np.ndarray       # g(r) per frame, shape (n_frames, ngrid)
    coordination_number: np.ndarray # running coordination number
    peak_r: list[float]             # positions of g(r) peaks (Å)
    peak_rdf: list[float]           # heights of g(r) peaks
    ref_species: list[str]
    species: list[str]


@dataclass
class CageResult:
    """Cage correlation function result from CoordinationAnalyzer."""
    lag_times: np.ndarray           # lag times in fs
    cage_correlation: np.ndarray    # C(τ) ∈ [0, 1]
    cage_time: float                # integral cage time (fs)
    cage_time_1e: float             # time to 1/e crossing (fs); NaN if never reached
    mobile_species: list[str]
    cage_species: list[str]
    cutoff: float


@dataclass
class CoordinationAnalysisDoc:
    """
    Serializable document holding all results from a CoordinationAnalyzer run.

    ``rdfs`` contains one ``RDFResult`` per unique species pair (keyed by
    ``rdf_pair_key``, e.g. ``"Li-O"``) plus one entry under ``"total"``
    that uses all species as both reference and neighbor.

    ``cage_results`` is optional; populate it by calling
    ``get_cage_correlation`` on the analyzer and passing the results here,
    or by using ``from_analyzer`` with a pre-built dict.
    """
    rdfs: dict[str, RDFResult]
    unique_species: list[str]
    rmax: float
    ngrid: int
    sigma: float
    n_frames: int
    cage_results: dict[str, CageResult] = field(default_factory=dict)

    @classmethod
    def from_analyzer(
        cls,
        an: CoordinationAnalyzer,
        cage_results: dict[str, CageResult] | None = None,
    ) -> CoordinationAnalysisDoc:
        """
        Build a ``CoordinationAnalysisDoc`` from a post-``analyze()`` analyzer.

        Args:
            an: A ``CoordinationAnalyzer`` on which ``analyze()`` has been called.
            cage_results: Optional dict of ``CageResult`` objects keyed by an
                arbitrary label (e.g. ``"Li-O"``).  If omitted, ``cage_results``
                is left empty.

        Returns:
            Populated ``CoordinationAnalysisDoc``.
        """
        if not an._analyzed:
            raise RuntimeError("Call analyze() on the CoordinationAnalyzer before building a doc.")
        return cls(
            rdfs=dict(an.rdfs),
            unique_species=list(an.unique_species),
            rmax=an.rmax,
            ngrid=an.ngrid,
            sigma=an.sigma,
            n_frames=len(an.frame_indices),
            cage_results=dict(cage_results) if cage_results else {},
        )

    def get_rdf(self, ref_species: str, species: str) -> RDFResult:
        """
        Look up a stored RDF result by species pair.

        The lookup is order-insensitive: ``get_rdf("O", "Li")`` returns the
        same result as ``get_rdf("Li", "O")``.

        Args:
            ref_species: One member of the pair.
            species: The other member of the pair.

        Returns:
            The corresponding ``RDFResult``.
        """
        from ..analyzers.coordination import rdf_pair_key
        key = rdf_pair_key(ref_species, species)
        if key not in self.rdfs:
            raise KeyError(
                f"No RDF stored for pair '{key}'. "
                f"Available keys: {list(self.rdfs)}"
            )
        return self.rdfs[key]
