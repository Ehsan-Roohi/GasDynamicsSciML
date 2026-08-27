"""Fail if manuscript anchors and frozen evidence diverge."""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revision"
MANUSCRIPT = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
GENERATED = (ROOT / "manuscript" / "generated_timing_values.tex").read_text(encoding="utf-8")


def require_generated(name: str, value: str) -> None:
    anchor = rf"\newcommand{{\{name}}}{{{value}}}"
    if anchor not in GENERATED:
        raise AssertionError(f"Generated timing value drift: {anchor}")


def latex_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def main() -> None:
    metrics = pd.read_csv(RESULTS / "primary_metrics.csv").set_index("problem")
    anchors = {
        "Rayleigh inverse": ("1.642e-3", metrics.loc["Rayleigh inverse", "rel_l2"]),
        "Fanno inverse": ("3.431e-3", metrics.loc["Fanno inverse", "rel_l2"]),
        "Oblique inverse": ("7.990e-4", metrics.loc["Oblique inverse", "rel_l2"]),
        "Nozzle inverse": ("1.936e-3", metrics.loc["Nozzle inverse", "rel_l2"]),
        "Shock tube implicit": ("9.490e-4", metrics.loc["Shock tube implicit", "rel_l2"]),
    }
    for problem, (text_anchor, value) in anchors.items():
        if text_anchor not in MANUSCRIPT:
            raise AssertionError(f"Missing manuscript anchor for {problem}: {text_anchor}")
        expected = float(text_anchor)
        if abs(value - expected) > 5e-7:
            raise AssertionError(f"Evidence drift for {problem}: CSV={value}, text={expected}")
    required = [
        "Rayleigh_Comparison_Revised.pdf",
        "Rayleigh_Inverse_Revised.pdf",
        "Rayleigh_Ts_Revised.pdf",
        "Fanno_Ratios_Revised.pdf",
        "Fanno_Friction_Revised.pdf",
        "Fanno_Inverse_Revised.pdf",
        "Fanno_Ts_Revised.pdf",
        "Oblique_Manifold_Revised.pdf",
    ]
    range_table = RESULTS / "range_generalization.csv"
    if not range_table.is_file() or len(pd.read_csv(range_table)) != 5:
        raise AssertionError("Edge-holdout table must contain all five problems")
    dimension_table = pd.read_csv(RESULTS / "high_dimensional_scaling.csv").set_index("dimension")
    dimension_anchors = {
        2: ("2 & 64 & 4096 & 0.0205", 0.00020531342849885735),
        5: ("5 & 5 & 3125 & 4.885", 0.048849607952961456),
    }
    for dimension, (text_anchor, expected) in dimension_anchors.items():
        if text_anchor not in MANUSCRIPT:
            raise AssertionError(f"Missing dimensional manuscript anchor: {text_anchor}")
        value = float(dimension_table.loc[dimension, "interpolation_rel_l2"])
        if abs(value - expected) > 5.0e-10:
            raise AssertionError(f"Dimensional evidence drift for d={dimension}: {value}")
    mlp_d5 = float(dimension_table.loc[5, "mlp_rel_l2"])
    if "& 0.1771 &" not in MANUSCRIPT or abs(mlp_d5 - 0.0017710570639642381) > 5.0e-10:
        raise AssertionError(f"Five-dimensional MLP evidence drift: {mlp_d5}")
    application = pd.read_csv(RESULTS / "nozzle_gradient_audit.csv")
    require_generated("AuditNozzleMaxIterations", str(int(application["newton_iterations"].max())))
    require_generated("AuditNozzleMaxShockError", latex_scientific(float(application["shock_target_abs_error"].max())))
    require_generated("AuditNozzleMaxPressureError", latex_scientific(float(application["back_pressure_abs_error"].max())))
    require_generated("AuditNozzleMaxJacobianDiscrepancy", latex_scientific(float(application["gradient_relative_difference"].max())))
    workload = pd.read_csv(RESULTS / "shock_tube_many_query.csv").set_index("method")
    if "100,000-state" not in MANUSCRIPT or "\\TimingWorkloadSpeedup" not in MANUSCRIPT:
        raise AssertionError("Missing many-query manuscript anchor")
    if int(workload.loc["physics_guided_mlp", "query_count"]) != 100000:
        raise AssertionError("Shock-tube workload must contain 100,000 queries")
    speedup = float(workload.loc["physics_guided_mlp", "speedup_vs_brent"])
    require_generated("TimingWorkloadBrentSeconds", f"{workload.loc['bracketed_brent', 'elapsed_seconds']:.3f}")
    require_generated("TimingWorkloadMlpSeconds", f"{workload.loc['physics_guided_mlp', 'elapsed_seconds']:.3f}")
    require_generated("TimingWorkloadSpeedup", f"{speedup:.1f}")
    require_generated("TimingWorkloadRelLtwoPercent", f"{100.0 * workload.loc['physics_guided_mlp', 'rel_l2_vs_brent']:.4f}")
    timing = pd.read_csv(RESULTS / "timing.csv")
    batch = timing[timing["batch_size"] == 5000].pivot(
        index="problem", columns="method", values="median_ms"
    )
    root_speedups = batch["bracketed_root"] / batch["physics_guided_mlp"]
    require_generated("TimingRootSpeedMin", str(math.floor(root_speedups.min())))
    require_generated("TimingRootSpeedMax", str(math.ceil(root_speedups.max())))
    shock_speedup = float(root_speedups.loc["Shock tube implicit"])
    if abs(shock_speedup - speedup) / speedup > 0.25:
        raise AssertionError(
            f"Shock-tube timing protocols disagree: 5000={shock_speedup}, workload={speedup}"
        )
    for name in required:
        path = RESULTS / "article_figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty figure: {path}")
    print("release evidence validated")


if __name__ == "__main__":
    main()
