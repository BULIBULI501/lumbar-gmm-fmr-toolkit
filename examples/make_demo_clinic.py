"""Regenerate the synthetic demo clinic table (generic column names).

Thin wrapper around :func:`lumbarmri.synth.write_demo_data`.  The output is
deterministic (seed 42): regenerating produces the same
``demo_clinic.csv`` + ``synth_truth.json`` as shipped.  The committed
``demo_baseline.json`` (reproducibility anchor) is left untouched.

Usage::

    python examples/make_demo_clinic.py
    python examples/make_demo_clinic.py --output-dir /tmp/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumbarmri.synth import write_demo_data  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "examples" / "demo_data",
        help="output directory (default: examples/demo_data)",
    )
    args = parser.parse_args(argv)

    csv_path, truth_path = write_demo_data(args.output_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
