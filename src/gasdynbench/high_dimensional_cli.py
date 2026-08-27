"""Console entry point for the dimensional-scaling audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .high_dimensional import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/revision"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    run(args.output, quick=args.quick, seed=args.seed)


if __name__ == "__main__":
    main()
