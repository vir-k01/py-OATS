"""Run ProductionMDMaker for a production NVT MD job from an existing structure."""

from __future__ import annotations

import argparse
import json
import sys

try:
    from jobflow import run_locally
    from py_oats.workflow.core import ProductionMDMaker
except ImportError:
    sys.exit("This script requires atomate2 and jobflow: pip install atomate2 jobflow")

from pymatgen.core import Structure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a production NVT MD job.")
    parser.add_argument("--structure", required=True, help="Path to structure file (JSON, POSCAR, CIF, etc.)")
    parser.add_argument("--temperature", type=float, help="Temperature (K)")
    parser.add_argument("--potential-path", help="Path to GRACE potential directory")
    parser.add_argument("--pair-style", help="LAMMPS pair_style (default: 'grace')")
    parser.add_argument("--time-step", type=float, help="Timestep in ps (default: 0.001)")
    parser.add_argument("--eq-steps", type=int, help="Equilibration steps (default: 50000)")
    parser.add_argument("--prod-steps", type=int, help="Production steps (default: 10000000)")
    parser.add_argument("--dump-interval", type=int, help="Dump interval in steps (default: 1000)")
    parser.add_argument("--seed", type=int, help="Random seed for velocity initialization")
    parser.add_argument("--run-locally", action="store_true", help="Run via jobflow.run_locally()")
    args = parser.parse_args(argv)

    struct = Structure.from_file(args.structure)
    print(f"Structure: {struct.composition.reduced_formula}, {struct.num_sites} sites")

    settings = {}
    for key, attr in [
        ("temperature", "temperature"),
        ("potential_path", "potential_path"),
        ("pair_style", "pair_style"),
        ("time_step", "time_step"),
        ("eq_steps", "eq_steps"),
        ("prod_steps", "prod_steps"),
        ("dump_interval", "dump_interval"),
        ("seed", "seed"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            settings[key] = val

    maker = ProductionMDMaker(settings=settings)

    from py_oats.utils.workflow.helpers import check_lammps_grace
    lmp_path = check_lammps_grace(pair_style=maker.settings.get("pair_style", "grace"))
    print(f"LAMMPS binary: {lmp_path}")
    print(f"Settings: {maker.settings}")

    job = maker.make(input_structure=struct)

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


if __name__ == "__main__":
    main()
