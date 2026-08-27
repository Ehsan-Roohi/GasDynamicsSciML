#!/usr/bin/env python3
"""Regenerate the corrected scientific line figures retained by the article.

The article deliberately keeps the figure set small. Summary error,
timing, sensitivity, and dimensional-scaling evidence is reported in tables.
The Rayleigh pointwise-error dashboard is intentionally omitted because its
aggregate norms are reported in the manuscript table. This script regenerates
the corrected Rayleigh, Fanno, and oblique-shock line figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from gasdynbench.physics import entropy_over_r, mach_angle
from gasdynbench.run_revision import _build_fanno, _build_oblique, _build_rayleigh


EXACT = "#222222"
ML = "#0072B2"
ALT = "#D55E00"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def _rayleigh(output: Path) -> None:
    ev, extra = _build_rayleigh(np.random.default_rng(11), 900, 11, False)
    mach = extra["forward_mach"]
    true = extra["forward_true"]
    pred = extra["forward_pred"]
    columns = [0, 1, 2, 3, 5]
    labels = [r"$T/T^*$", r"$P/P^*$", r"$\rho/\rho^*$", r"$u/u^*$", r"$P_0/P_0^*$"]

    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.6))
    for ax, col, label in zip(axes.ravel(), columns, labels):
        ax.plot(mach, true[:, col], color=EXACT, label="Analytical")
        ax.plot(mach, pred[:, col], color=ML, linestyle="--", label="Structured MLP")
        ax.set(xlabel="Mach number", ylabel=label)
        ax.grid(alpha=0.2)
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "Rayleigh_Comparison_Revised")

    pred_m = ev.predict(ev.x_test)
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    for branch, color, name in [(0, ALT, "Subsonic expert"), (1, ML, "Supersonic expert")]:
        mask = ev.x_test[:, 1] == branch
        order = np.argsort(ev.y_test[mask])
        exact_m = ev.y_test[mask][order]
        ratio = ev.x_test[mask, 0][order]
        ax.plot(exact_m, ratio, color=EXACT, label="Analytical" if branch == 0 else None)
        ax.plot(pred_m[mask][order], ratio, color=color, linestyle="--", label=name)
    ax.axvline(1.0, color="0.55", linewidth=0.8, linestyle=":")
    ax.set(xlabel="Mach number", ylabel=r"$T_0/T_0^*$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "Rayleigh_Inverse_Revised")

    s_true = entropy_over_r(true[:, 0], true[:, 1])
    s_pred = entropy_over_r(pred[:, 0], pred[:, 1])
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    ax.plot(s_true, true[:, 0], color=EXACT, label="Analytical")
    ax.plot(s_pred, pred[:, 0], color=ML, linestyle="--", label="Structured MLP")
    ax.set(xlabel=r"$(s-s^*)/R$", ylabel=r"$T/T^*$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "Rayleigh_Ts_Revised")


def _fanno(output: Path) -> None:
    ev, extra = _build_fanno(np.random.default_rng(111), 900, 12, False)
    mach = extra["forward_mach"]
    true = extra["forward_true"]
    pred = extra["forward_pred"]
    labels = [r"$T/T^*$", r"$P/P^*$", r"$\rho/\rho^*$", r"$P_0/P_0^*$"]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    for ax, col, label in zip(axes.ravel(), range(4), labels):
        ax.plot(mach, true[:, col], color=EXACT, label="Analytical")
        ax.plot(mach, pred[:, col], color=ML, linestyle="--", label="Structured MLP")
        ax.set(xlabel="Mach number", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "Fanno_Ratios_Revised")

    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    distance = np.abs(mach - 1.0)
    for j, branch in enumerate((mach < 1.0, mach > 1.0)):
        order = np.argsort(distance[branch])
        ax.loglog(
            distance[branch][order],
            np.maximum(true[branch, 4][order], 1.0e-16),
            color=EXACT,
            label="Analytical" if j == 0 else None,
        )
        ax.loglog(
            distance[branch][order],
            np.maximum(pred[branch, 4][order], 1.0e-16),
            color=ML,
            linestyle="--",
            label="Structured MLP" if j == 0 else None,
        )
    ax.set(xlabel=r"Distance from sonic state, $|M-1|$", ylabel=r"$4fL^*/D$")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "Fanno_Friction_Revised")

    pred_m = ev.predict(ev.x_test)
    fld = ev.test_context["fld"]
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    for branch, color, name in [(0, ALT, "Subsonic expert"), (1, ML, "Supersonic expert")]:
        mask = ev.x_test[:, 1] == branch
        order = np.argsort(fld[mask])
        ax.semilogx(fld[mask][order], ev.y_test[mask][order], color=EXACT, label="Analytical" if branch == 0 else None)
        ax.semilogx(fld[mask][order], pred_m[mask][order], color=color, linestyle="--", label=name)
    ax.set(xlabel=r"$4fL^*/D$", ylabel="Mach number")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "Fanno_Inverse_Revised")

    s_true = entropy_over_r(true[:, 0], true[:, 1])
    s_pred = entropy_over_r(pred[:, 0], pred[:, 1])
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    ax.plot(s_true, true[:, 0], color=EXACT, label="Analytical")
    ax.plot(s_pred, pred[:, 0], color=ML, linestyle="--", label="Structured MLP")
    ax.set(xlabel=r"$(s-s^*)/R$", ylabel=r"$T/T^*$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "Fanno_Ts_Revised")


def _oblique(output: Path) -> None:
    _, extra = _build_oblique(np.random.default_rng(211), 900, 13, False)
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    for mach in np.unique(extra["direct_mach"]):
        mask = extra["direct_mach"] == mach
        q = extra["direct_q"][mask]
        beta = mach_angle(mach) + q * (0.5 * np.pi - mach_angle(mach))
        ax.plot(np.degrees(beta), np.degrees(extra["direct_true"][mask]), color=EXACT, alpha=0.72)
        ax.plot(np.degrees(beta), np.degrees(extra["direct_pred"][mask]), color=ML, linestyle="--", alpha=0.82)
        peak = int(np.argmax(extra["direct_true"][mask]))
        ax.text(np.degrees(beta[peak]), np.degrees(extra["direct_true"][mask][peak]) + 0.7, f"M={mach:g}", fontsize=8)
    ax.plot([], [], color=EXACT, label="Analytical")
    ax.plot([], [], color=ML, linestyle="--", label="Hard-envelope MLP")
    ax.set(xlabel=r"Shock angle $\beta$ (deg)", ylabel=r"Deflection angle $\theta$ (deg)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output, "Oblique_Manifold_Revised")


def main() -> None:
    _style()
    output = Path("results/revision/article_figures")
    _rayleigh(output)
    _fanno(output)
    _oblique(output)


if __name__ == "__main__":
    main()
