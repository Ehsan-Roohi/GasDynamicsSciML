"""Run the differentiability and 100,000-query application audits."""

from __future__ import annotations

import argparse
from pathlib import Path

from gasdynbench.application_audits import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/revision"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    summary = run(args.output, quick=args.quick, seed=args.seed)
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
