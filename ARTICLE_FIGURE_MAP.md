# Article figure map

| Manuscript item | Generated asset | Evidence source |
| --- | --- | --- |
| Figure 1: blind agreement | `results/revision/figures/benchmark_accuracy.pdf` | `primary_metrics.csv` plus blind arrays generated in the run |
| Figure 2: controlled ablations | `results/revision/figures/ablation_branch_singularity.pdf` | `ablations.csv` |
| Figure 3: sensitivity/limits | `results/revision/figures/sensitivity_and_singular_error.pdf` | `training_size_scaling.csv`, `near_singular_metrics.csv` |
| Figure 4: measured cost | `results/revision/figures/measured_timing.pdf` | `timing.csv` |
| Rayleigh supplementary panel | `results/revision/figures/rayleigh_results.pdf` | analytical and forward/inverse blind predictions |
| Fanno supplementary panel | `results/revision/figures/fanno_results.pdf` | analytical and structured predictions |
| Oblique supplementary panel | `results/revision/figures/oblique_results.pdf` | direct and branch inverse predictions |
| Nozzle supplementary panel | `results/revision/figures/nozzle_results.pdf` | bounded shock-location grid |
| Shock-tube supplementary panel | `results/revision/figures/shock_tube_results.pdf` | two-parameter residual audit |
| Dimensional-scaling supplementary panel | `results/revision/figures/high_dimensional_scaling.pdf` | `high_dimensional_scaling.csv` |

The edge-holdout range experiment is tabulated in `results/revision/range_generalization.csv` and reported in the manuscript text.

The PDF versions are vector graphics. PNG siblings are saved at 400 dpi for submission systems that require raster images.
