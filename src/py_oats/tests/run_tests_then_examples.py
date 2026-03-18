"""
Run pytest for py_oats, then informal checks on examples/ trajectories.

Usage (from repo root, with conda env activated):
  PYTHONPATH=src python -m py_oats.tests.run_tests_then_examples

Or:
  conda activate charged_oats
  PYTHONPATH=src python src/py_oats/tests/run_tests_then_examples.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent.parent
_repo_root = _src_root.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))


def main() -> int:
    # 1. Run pytest
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(_src_root / "py_oats" / "tests"), "-v", "--tb=short"],
        cwd=str(_src_root),
        env={**__import__("os").environ, "PYTHONPATH": str(_src_root)},
    )
    if r.returncode != 0:
        print("Pytest failed; skipping informal examples run.")
        return r.returncode

    # 2. Run informal examples
    print("\n")
    from py_oats.tests.run_examples_informal import main as examples_main
    examples_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
