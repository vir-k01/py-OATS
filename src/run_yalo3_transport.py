"""End-to-end script: read YAlO3 dump, run TransportAnalyzer, plot, and write schema JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripts

import matplotlib.pyplot as plt
import numpy as np

from py_oats.io import TrajectoryData
from py_oats.analyzers.transport import TransportAnalyzer
from py_oats.schemas.transport import TransportDoc
from py_oats.plotter.transport import plot_all_correlations, plot_correlation_pairwise


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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dump_path = repo_root / "examples" / "YAlO3_1500.0K.dump"

    if not dump_path.is_file():
        raise FileNotFoundError(f"Could not find YAlO3 dump at {dump_path}")

    # 1) Read trajectory
    td = TrajectoryData.read(
        dump_path,
        time_step=2.0,
        step_skip=1,
        temperature=1500.0,
        metadata={},
        unwrap=True,
    )

    # 2) Run transport analysis
    analyzer = TransportAnalyzer(td)
    analyzer.analyze()

    # 3) Build schema (stores tensors, correlation_functions, fit_dicts, etc.)
    doc = TransportDoc.from_analyzer(analyzer)

    # 4) Generate plots using the plotter; save to PNGs
    out_dir = repo_root / "examples" / "yalo3_transport_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pairwise summary plot
    ax_pairwise = plot_correlation_pairwise(doc)
    ax_pairwise.figure.savefig(out_dir / "correlations_pairwise.png", dpi=200)
    plt.close(ax_pairwise.figure)

    # Grid of all correlations
    fig_grid = plot_all_correlations(doc)
    fig_grid.savefig(out_dir / "correlations_grid.png", dpi=200)
    plt.close(fig_grid)

    # 5) Save schema to JSON
    json_path = repo_root / "examples" / "yalo3_transport_doc.json"
    data = doc.as_dict()
    with json_path.open("w") as f:
        json.dump(_to_jsonable(data), f, indent=2)

    print(f"Wrote schema to {json_path}")
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()

