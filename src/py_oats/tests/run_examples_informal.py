"""
Informal checks using example trajectory files under examples/.

Run after pytest. Uses TrajectoryData.read() as drop-in for how notebooks
read trajectories (ase.io.read(path, index=':') for .dump; path → list of Atoms
then our io builds TrajectoryData). Does not run inside pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path when run as script (file is in src/py_oats/tests/)
_src_root = Path(__file__).resolve().parent.parent.parent  # src
_repo_root = _src_root.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from py_oats.io.trajectory import TrajectoryData


EXAMPLES_DIR = _repo_root / "examples"
# Same files as in example.ipynb / LMO_example.ipynb
DUMP_FILES = ["LiMn2O4-1250K.dump", "YAlO3_1500.0K.dump"]
XDATCAR_FILE = "Li2Ti3O7_1750.0K.XDATCAR"


def main() -> None:
    print("Informal trajectory checks (examples/)")
    print("=" * 50)

    for name in DUMP_FILES:
        path = EXAMPLES_DIR / name
        if not path.exists():
            print(f"  SKIP {name} (not found)")
            continue
        try:
            td = TrajectoryData.read(str(path), metadata={})
            print(f"  {name}: n_frames={td.n_frames}, n_atoms={td.n_atoms}, species={sorted(set(td.species))}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    xpath = EXAMPLES_DIR / XDATCAR_FILE
    if xpath.exists():
        try:
            # XDATCAR is read via pymatgen (ASE does not parse it); drop-in: structures -> TrajectoryData.read(list[Structure])
            from pymatgen.io.vasp import Xdatcar
            xdatcar = Xdatcar(str(xpath))
            structures = xdatcar.structures
            td = TrajectoryData.read(structures, metadata={})
            print(f"  {XDATCAR_FILE}: n_frames={td.n_frames}, n_atoms={td.n_atoms}, species={sorted(set(td.species))}")
        except Exception as e:
            print(f"  {XDATCAR_FILE}: FAILED - {e}")
    else:
        print(f"  SKIP {XDATCAR_FILE} (not found)")

    print("Done.")


if __name__ == "__main__":
    main()
