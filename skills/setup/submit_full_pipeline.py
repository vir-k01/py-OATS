"""Run the full ProductionOnsagerMaker pipeline: structure gen + MD + analysis."""

from __future__ import annotations

import argparse
import json
import sys

try:
    from jobflow import run_locally
    from py_oats.workflow.core import (
        AmorphousStateMaker,
        AnalysisMaker,
        ProductionMDMaker,
        ProductionOnsagerMaker,
    )
except ImportError:
    sys.exit("This script requires atomate2 and jobflow: pip install atomate2 jobflow")

from pymatgen.core import Composition, Structure

from py_oats.analyzers.coordination import CoordinationAnalyzer
from py_oats.analyzers.transport import TransportAnalyzer

_ANALYZER_MAP = {
    "transport": TransportAnalyzer,
    "coordination": CoordinationAnalyzer,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full structure -> MD -> analysis pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--composition", help="Chemical formula, e.g. 'Li2S'")
    group.add_argument("--structure", help="Path to structure file")
    parser.add_argument("--temperature", type=float, help="Temperature (K)")
    parser.add_argument("--potential-path", help="Path to GRACE potential")
    parser.add_argument("--pair-style", help="LAMMPS pair_style")
    parser.add_argument("--prod-steps", type=int, help="Production steps")
    parser.add_argument("--eq-steps", type=int, help="Equilibration steps")
    parser.add_argument("--dump-interval", type=int, help="Dump interval in steps")
    parser.add_argument("--analyzers", nargs="+", choices=list(_ANALYZER_MAP.keys()), default=["transport"],
                        help="Analyzers to run (default: transport)")
    parser.add_argument("--run-locally", action="store_true", help="Run via jobflow.run_locally()")
    args = parser.parse_args(argv)

    md_settings = {}
    for key, attr in [
        ("temperature", "temperature"),
        ("potential_path", "potential_path"),
        ("pair_style", "pair_style"),
        ("prod_steps", "prod_steps"),
        ("eq_steps", "eq_steps"),
        ("dump_interval", "dump_interval"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            md_settings[key] = val

    amorphous_settings = {}
    if args.potential_path:
        amorphous_settings["potential_path"] = args.potential_path
    if args.pair_style:
        amorphous_settings["pair_style"] = args.pair_style

    analyzer_classes = [_ANALYZER_MAP[name] for name in args.analyzers]

    maker = ProductionOnsagerMaker(
        amorphous_state_maker=AmorphousStateMaker(settings=amorphous_settings),
        production_md_maker=ProductionMDMaker(settings=md_settings),
        analysis_maker=AnalysisMaker(analyzers=analyzer_classes),
    )

    from py_oats.utils.workflow.helpers import check_lammps_grace
    lmp_path = check_lammps_grace(pair_style=maker.production_md_maker.settings.get("pair_style", "grace"))
    print(f"LAMMPS binary: {lmp_path}")

    if args.composition:
        target = Composition(args.composition)
        print(f"Full pipeline for {target.reduced_formula}")
    else:
        target = Structure.from_file(args.structure)
        print(f"Full pipeline for {target.composition.reduced_formula} ({target.num_sites} sites)")

    print(f"  Analyzers: {args.analyzers}")
    print(f"  MD settings: {maker.production_md_maker.settings}")

    flow = maker.make(structure_or_composition=target)
    print(f"  Flow: {flow.name}, {len(flow.jobs)} jobs")

    if args.run_locally:
        print("\nRunning locally...")
        result = run_locally(flow, create_folders=True)
        for uuid, outputs in result.items():
            for idx, out in outputs.items():
                val = out.output
                if hasattr(val, "state"):
                    print(f"  [{uuid[:8]}] state={val.state}, label={val.task_label}")
                elif isinstance(val, list):
                    print(f"  [{uuid[:8]}] returned {len(val)} analysis doc(s)")
                else:
                    print(f"  [{uuid[:8]}] {type(val).__name__}")
    else:
        print(f"\nFlow created with {len(flow.jobs)} jobs.")
        print("Use jobflow or FireWorks to submit.")


if __name__ == "__main__":
    main()
