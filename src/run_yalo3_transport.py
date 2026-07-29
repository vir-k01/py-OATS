"""End-to-end script: read YAlO3 dump, run TransportAnalyzer + CoordinationAnalyzer,
plot, and write schema JSON files."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripts

import matplotlib.pyplot as plt
import numpy as np

from py_oats.io import TrajectoryData
from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.analyzers.coordination import CoordinationAnalyzer
from py_oats.schemas.transport import TransportDoc
from py_oats.schemas.coordination import CoordinationAnalysisDoc
from py_oats.plotter.transport import plot_all_correlations, plot_correlation_pairwise
from py_oats.plotter.coordination import plot_rdf_grid, plot_rdf


def _to_jsonable(obj):
    """Recursively convert numpy types and non-JSONable keys to plain Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _first_minimum(r: np.ndarray, rdf: np.ndarray, rmin: float = 1.0) -> float:
    """Return r at the first local minimum of rdf after the first peak."""
    mask = r > rmin
    gr = rdf[mask]
    rr = r[mask]
    # find first peak
    peak_idx = int(np.argmax(gr))
    # find first minimum after peak
    for i in range(peak_idx + 1, len(gr) - 1):
        if gr[i] < gr[i - 1] and gr[i] < gr[i + 1]:
            return float(rr[i])
    return float(rr[peak_idx + 1])  # fallback: just past peak


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dump_path = repo_root / "examples" / "YAlO3_1500.0K.dump"

    if not dump_path.is_file():
        raise FileNotFoundError(f"Could not find YAlO3 dump at {dump_path}")

    # ------------------------------------------------------------------
    # 1) Read trajectory
    # ------------------------------------------------------------------
    print("Reading trajectory ...", flush=True)
    td = TrajectoryData.read(
        dump_path,
        time_step=2.0,
        step_skip=1,
        temperature=1500.0,
        metadata={},
        unwrap=True,
    )
    print(f"  {td.n_frames} frames, {td.n_atoms} atoms, species: {td.unique_species}")

    # ------------------------------------------------------------------
    # 2) Transport analysis
    # ------------------------------------------------------------------
    print("Running TransportAnalyzer ...", flush=True)
    transport_analyzer = TransportAnalyzer(td)
    transport_analyzer.analyze()
    transport_doc = TransportDoc.from_analyzer(transport_analyzer)

    out_dir = repo_root / "examples" / "yalo3_transport_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ax_pairwise = plot_correlation_pairwise(transport_doc)
    ax_pairwise.figure.savefig(out_dir / "correlations_pairwise.png", dpi=200)
    plt.close(ax_pairwise.figure)

    fig_grid = plot_all_correlations(transport_doc)
    fig_grid.savefig(out_dir / "correlations_grid.png", dpi=200)
    plt.close(fig_grid)

    json_path = repo_root / "examples" / "yalo3_transport_doc.json"
    data = transport_doc.as_dict()
    with json_path.open("w") as f:
        json.dump(_to_jsonable(data), f, indent=2)
    print(f"  Transport JSON → {json_path}")

    # ------------------------------------------------------------------
    # 3) Coordination analysis (RDF + cage times)
    # ------------------------------------------------------------------
    print("Running CoordinationAnalyzer ...", flush=True)
    coord_dir = repo_root / "examples" / "yalo3_coordination_plots"
    coord_dir.mkdir(parents=True, exist_ok=True)

    coord_analyzer = CoordinationAnalyzer(td, rmax=8.0, ngrid=201, sigma=0.1)
    coord_analyzer.analyze()
    print(f"  RDF pairs: {list(coord_analyzer.rdfs.keys())}")

    # ------------------------------------------------------------------
    # 3a) RDF plots
    # ------------------------------------------------------------------
    fig_grid = plot_rdf_grid(coord_analyzer, include_total=True, show_coordination_number=True)
    fig_grid.suptitle("YAlO₃ 1500 K — RDF", fontsize=22, y=1.01)
    fig_grid.savefig(coord_dir / "rdf_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"  RDF grid → {coord_dir / 'rdf_grid.png'}")

    # Individual O-Al and O-Y panels with CN
    for pair in [("O", "Al"), ("O", "Y")]:
        key = "-".join(sorted(pair))
        ax = plot_rdf(coord_analyzer, pairs=[key], show_coordination_number=True)
        ax.set_title(f"YAlO₃ 1500 K — {key}", fontsize=14)
        ax.figure.savefig(coord_dir / f"rdf_{key.lower()}_cn.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

    # ------------------------------------------------------------------
    # 3b) Cage correlations: O mobile vs Al and Y cage
    # Use first minimum of the respective RDF as the cutoff.
    # ------------------------------------------------------------------
    cage_results: dict = {}
    for cage_sp in ("Al", "Y"):
        key = "-".join(sorted(("O", cage_sp)))
        result = coord_analyzer.rdfs[key]
        cutoff = _first_minimum(result.r, result.rdf)
        print(f"  Cage O-{cage_sp}: cutoff = {cutoff:.2f} Å (first RDF minimum)", flush=True)
        cage_results[f"O-{cage_sp}"] = coord_analyzer.get_cage_correlation(
            "O", cage_sp, cutoff=cutoff
        )

    # ------------------------------------------------------------------
    # 3c) Cage correlation plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(cage_results), figsize=(7 * len(cage_results), 5))
    if len(cage_results) == 1:
        axes = [axes]
    colors = ["#2196F3", "#FF9800"]
    for ax, ((label, res), color) in zip(axes, zip(cage_results.items(), colors)):
        ax.plot(res.lag_times, res.cage_correlation, color=color, linewidth=2)
        ax.axhline(1 / np.e, color="0.5", linewidth=1, linestyle="--", label="1/e")
        if not np.isnan(res.cage_time_1e):
            ax.axvline(res.cage_time_1e, color=color, linewidth=1, linestyle=":",
                       label=f"τ₁/e = {res.cage_time_1e:.0f} fs")
        ax.set_xlabel("Lag time (fs)", fontsize=13)
        ax.set_ylabel("C(τ)", fontsize=13)
        ax.set_title(f"{label}  (cutoff={res.cutoff:.2f} Å)", fontsize=13)
        ax.legend(fontsize=11)
        print(f"  {label}: cage_time={res.cage_time:.1f} fs, "
              f"cage_time_1e={res.cage_time_1e:.1f} fs")
    fig.suptitle("YAlO₃ 1500 K — Cage correlations (O mobile)", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(coord_dir / "cage_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cage plot → {coord_dir / 'cage_correlations.png'}")

    # ------------------------------------------------------------------
    # 3d) Build CoordinationAnalysisDoc and serialise to JSON
    # ------------------------------------------------------------------
    coord_doc = CoordinationAnalysisDoc.from_analyzer(coord_analyzer, cage_results=cage_results)
    coord_json_path = repo_root / "examples" / "yalo3_coordination_doc.json"
    coord_data = dataclasses.asdict(coord_doc)
    with coord_json_path.open("w") as f:
        json.dump(_to_jsonable(coord_data), f, indent=2)
    print(f"  Coordination JSON → {coord_json_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
