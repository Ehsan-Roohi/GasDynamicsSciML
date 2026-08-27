"""Refresh scaling evidence and repair the two publication figures independently."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gasdynbench.figures import COLORS, _style
from gasdynbench.run_revision import _build_fanno, _build_nozzle, _build_oblique, _scaling


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision"
FIG = OUT / "figures"


def atomic_save(fig, stem: str):
    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 400})]:
        tmp = FIG / f".{stem}.tmp.{suffix}"
        final = FIG / f"{stem}.{suffix}"
        fig.savefig(tmp, bbox_inches="tight", **kwargs)
        os.replace(tmp, final)
    plt.close(fig)


def refresh_scaling():
    builders = [("Fanno inverse", _build_fanno), ("Oblique inverse", _build_oblique)]
    scaling = _scaling(builders, seed=11, quick=False)
    scaling.to_csv(OUT / "training_size_scaling.csv", index=False)
    near = pd.read_csv(OUT / "near_singular_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for problem, frame in scaling.groupby("problem"):
        stats = frame.groupby("training_samples")["rel_l2"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(stats["training_samples"], stats["mean"], yerr=stats["std"], marker="o", markersize=3.5, capsize=2.5, label=problem.replace(" inverse", ""))
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Training samples"); axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_title(r"Training-size sensitivity (mean $\pm$ SD, three seeds)")
    axes[0].legend(frameon=False)
    pivot = near.pivot(index="problem", columns="bin", values="mae")
    im = axes[1].imshow(np.log10(np.maximum(pivot.values, 1e-12)), aspect="auto", cmap="viridis")
    axes[1].set_yticks(np.arange(len(pivot)), [x.replace(" inverse", "") for x in pivot.index])
    axes[1].set_xticks(np.arange(4), ["closest", "near", "mid", "far"])
    axes[1].set_xlabel("Distance from limiting regime")
    axes[1].set_title(r"Local MAE, $\log_{10}$ scale")
    fig.colorbar(im, ax=axes[1], label=r"$\log_{10}(\mathrm{MAE})$")
    for ax in axes: ax.grid(alpha=0.15)
    fig.tight_layout(); atomic_save(fig, "sensitivity_and_singular_error")


def repair_timing():
    timing = pd.read_csv(OUT / "timing.csv")
    batch = int(timing["batch_size"].max())
    pivot = timing[timing["batch_size"] == batch].pivot(index="problem", columns="method", values="median_ms")
    fig, ax = plt.subplots(figsize=(7.8, 3.7)); x = np.arange(len(pivot)); width = 0.25
    for j, (method, label, color) in enumerate([
        ("bracketed_root", "Bracketed root", COLORS["exact"]),
        ("classical_interpolation", "Interpolation", COLORS["baseline"]),
        ("physics_guided_mlp", "MLP batch", COLORS["ml"]),
    ]): ax.bar(x + (j - 1) * width, pivot[method], width, label=label, color=color)
    ax.set_yscale("log"); ax.set_xticks(x, [i.replace(" inverse", "\ninverse").replace(" implicit", "\nimplicit") for i in pivot.index])
    ax.set_ylabel(f"Median wall time for {batch:,} queries (ms)")
    ax.set_title("Measured CPU query cost; lower is better")
    ax.grid(axis="y", alpha=0.25, which="both"); ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); atomic_save(fig, "measured_timing")


def repair_nozzle():
    rng = np.random.default_rng(311)
    ev, _ = _build_nozzle(rng, 900, 14, False)
    pred = ev.predict(ev.x_test); ctx = ev.test_context
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    sc = axes[0].scatter(ctx["ae"], ctx["pb"], c=pred, s=11, cmap="viridis")
    fig.colorbar(sc, ax=axes[0], label=r"Predicted $A_s/A_t$")
    axes[1].scatter(ev.y_test, pred, s=8, color=COLORS["ml"])
    lo, hi = ev.y_test.min(), ev.y_test.max(); axes[1].plot([lo, hi], [lo, hi], "--", color=COLORS["exact"])
    axes[0].set(xlabel=r"$A_e/A_t$", ylabel=r"$P_b/P_{01}$", title="Admissible internal-shock map")
    axes[1].set(xlabel=r"Analytical $A_s/A_t$", ylabel=r"Predicted $A_s/A_t$", title="Bounded inverse surrogate")
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout(); atomic_save(fig, "nozzle_results")


def main():
    _style(); refresh_scaling(); repair_timing(); repair_nozzle()


if __name__ == "__main__":
    main()
