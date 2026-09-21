"""Run the AmorphousStateMaker workflow to equilibrate an amorphous structure."""

from __future__ import annotations

import argparse
import json
import sys

try:
    from jobflow import run_locally
    from py_oats.workflow.core import AmorphousStateMaker
except ImportError:
    sys.exit("This script requires atomate2 and jobflow: pip install atomate2 jobflow")

from pymatgen.core import Composition


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the amorphous state equilibration workflow.")
    parser.add_argument("--composition", required=True, help="Chemical formula, e.g. 'Li2S'")
    parser.add_argument("--temperature", type=float, help="Target temperature (K)")
    parser.add_argument("--ramp-temperature", type=float, help="Ramp temperature (K)")
    parser.add_argument("--potential-path", help="Path to GRACE potential directory")
    parser.add_argument("--pair-style", help="LAMMPS pair_style (default: 'grace padding 0.05')")
    parser.add_argument("--time-step", type=float, help="Timestep in ps (default: 0.001)")
    parser.add_argument("--t-npt", type=float, help="NPT run time in ps")
    parser.add_argument("--t-nvt", type=float, help="NVT run time in ps")
    parser.add_argument("--t-ramp", type=float, help="Ramp time in ps")
    parser.add_argument("--density-threshold", type=float, help="Density change threshold for branching")
    parser.add_argument("--n-frames", type=int, help="Number of NVT frames to save and minimize")
    parser.add_argument("--run-locally", action="store_true", help="Run via jobflow.run_locally()")
    args = parser.parse_args(argv)

    settings = {}
    for key, attr in [
        ("temperature", "temperature"),
        ("ramp_temperature", "ramp_temperature"),
        ("potential_path", "potential_path"),
        ("pair_style", "pair_style"),
        ("time_step", "time_step"),
        ("t_npt", "t_npt"),
        ("t_nvt", "t_nvt"),
        ("t_ramp", "t_ramp"),
        ("density_threshold", "density_threshold"),
        ("n_frames", "n_frames"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            settings[key] = val

    maker = AmorphousStateMaker(settings=settings)

    from py_oats.utils.workflow.helpers import check_lammps_grace
    lmp_path = check_lammps_grace(pair_style=maker.settings.get("pair_style", "grace"))
    print(f"LAMMPS binary: {lmp_path}")

    comp = Composition(args.composition)
    print(f"Building amorphous state job for {comp.reduced_formula}...")
    print(f"  Settings: {maker.settings}")

    job = maker.make(composition=comp)

    if args.run_locally:
        print("Running locally...")
        result = run_locally(job, create_folders=True)
        for uuid, outputs in result.items():
            for idx, out in outputs.items():
                doc = out.output
                print(f"\nResult: state={doc.state}, label={doc.task_label}")
                print(f"  dir: {doc.dir_name}")
    else:
        print(f"\nJob created: {job.name} ({job.uuid})")
        print("Use jobflow or FireWorks to submit this job.")
        print(json.dumps(job.as_dict(), indent=2, default=str)[:500] + "...")


if __name__ == "__main__":
    main()
