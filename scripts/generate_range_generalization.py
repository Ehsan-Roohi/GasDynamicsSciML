"""Regenerate only the reviewer-requested edge-holdout table."""

from pathlib import Path

import numpy as np

from gasdynbench.run_revision import (
    _build_fanno,
    _build_nozzle,
    _build_oblique,
    _build_rayleigh,
    _build_shock_tube,
    _range_generalization,
)


def main():
    builders = [_build_rayleigh, _build_fanno, _build_oblique, _build_nozzle, _build_shock_tube]
    evidences = []
    for i, builder in enumerate(builders):
        evidence, _ = builder(np.random.default_rng(11 + 100 * i), 900, 11 + i, False)
        evidences.append(evidence)
    output = Path("results/revision/range_generalization.csv")
    _range_generalization(evidences, quick=False).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
