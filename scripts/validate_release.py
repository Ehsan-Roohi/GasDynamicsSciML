"""Fail if manuscript anchors and frozen evidence diverge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revision"
MANUSCRIPT = (ROOT / "manuscript" / "main_revised_body.tex").read_text(encoding="utf-8")


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
        "benchmark_accuracy.pdf",
        "ablation_branch_singularity.pdf",
        "sensitivity_and_singular_error.pdf",
        "measured_timing.pdf",
        "nozzle_results.pdf",
    ]
    for name in required:
        path = RESULTS / "figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty figure: {path}")
    print("release evidence validated")


if __name__ == "__main__":
    main()
