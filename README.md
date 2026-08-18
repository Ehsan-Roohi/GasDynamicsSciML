# Gas-dynamics neural-network examples

Standalone teaching/research scripts that learn mappings generated from
classical compressible-flow relations.

## Examples

- `5Ray.py` — forward and inverse Rayleigh-flow mappings.
- `12FannoGoodForBook (1).py` and `33Fanno.py` — Fanno-flow forward/inverse
  models.
- `19BethaTheta.py` — theta–beta–Mach relation over a wide Mach range.
- `28Nozzle.py` — area–Mach/nozzle and normal-shock calculations.
- `32ShockTube.py` — shock-tube relation and x–t visualization.
- `Thin_Airfoil_theoryV2.ipynb` — Colab thin-airfoil-theory notebook.

The scripts generally train a model when executed and write plots in the
current directory.  They require combinations of NumPy, SciPy, Pandas,
Matplotlib, scikit-learn, and TensorFlow.  Use an isolated environment and
inspect the hyperparameters before launching a long training run.

## Scientific status

These are illustrative prototypes, not drop-in replacements for the analytical
relations used to generate their training data.  For quantitative use, report
an untouched test set, error by subsonic/supersonic branch, behavior near sonic
or shock singularities, extrapolation bounds, random seeds, and comparison with
the exact formula.

No repository-wide license or citation metadata is declared yet.

