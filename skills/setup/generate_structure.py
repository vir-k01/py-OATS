"""Generate an amorphous structure from a composition."""

from __future__ import annotations

import argparse
import sys

from pymatgen.core import Composition

from py_oats.structure_generator.generator import get_amorphous_structure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate an amorphous structure from a composition.")
    parser.add_argument("--composition", required=True, help="Chemical formula, e.g. 'Li2S', 'YAlO3'")
    parser.add_argument("--temperature", type=float, default=5000.0, help="Temperature for density prediction (K, default: 5000)")
    parser.add_argument("--target-atoms", type=int, default=100, help="Target number of atoms (default: 100)")
    parser.add_argument("--vol-multiply", type=float, default=1.0, help="Volume scaling factor (default: 1.0)")
    parser.add_argument("--output", default="amorphous_structure.json", help="Output file path (default: amorphous_structure.json)")
    parser.add_argument("--format", choices=["json", "poscar", "cif"], default="json", help="Output format (default: json)")
    args = parser.parse_args(argv)

    comp = Composition(args.composition)
    print(f"Generating amorphous structure for {comp.reduced_formula}...")
    print(f"  Temperature: {args.temperature} K")
    print(f"  Target atoms: {args.target_atoms}")

    struct = get_amorphous_structure(
        comp,
        temperature=args.temperature,
        generator_kwargs={"target_atoms": args.target_atoms, "vol_multiply": args.vol_multiply},
    )

    print(f"\nGenerated structure:")
    print(f"  Formula: {struct.composition.reduced_formula}")
    print(f"  Sites: {struct.num_sites}")
    print(f"  Volume: {struct.volume:.1f} A^3")
    print(f"  Density: {struct.density:.3f} g/cm^3")
    print(f"  Cell: {struct.lattice.a:.2f} x {struct.lattice.b:.2f} x {struct.lattice.c:.2f} A")

    fmt_map = {"json": "json", "poscar": "poscar", "cif": "cif"}
    struct.to(filename=args.output, fmt=fmt_map[args.format])
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
