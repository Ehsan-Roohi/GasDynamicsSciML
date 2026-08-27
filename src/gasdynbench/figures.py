"""Publication-quality vector figures generated from machine-readable evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .physics import shock_tube_residual


COLORS = {"ml": "#0072B2", "exact": "#202020", "baseline": "#D55E00", "accent": "#009E73"}


def _style():
    mpl.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "savefig.dpi": 400,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, folder: Path, stem: str):
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def make_all_figures(output: Path, evidences, extras: dict, tables: dict[str, pd.DataFrame]):
    _style()
    folder = output / "figures"
    names = ["Rayleigh", "Fanno", "Oblique shock", "Nozzle shock", "Shock tube"]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2))
    axes = axes.ravel()
    for ax, ev, name in zip(axes, evidences, names):
        true = np.asarray(ev.y_test)
        pred = np.asarray(ev.predict(ev.x_test))
        if true.ndim == 2:
            true = true.ravel()
            pred = pred.ravel()
        ax.scatter(true, pred, s=8, alpha=0.55, color=COLORS["ml"], edgecolor="none")
        lo, hi = float(np.min(true)), float(np.max(true))
        ax.plot([lo, hi], [lo, hi], color=COLORS["exact"], linestyle="--", label="Ideal")
        ax.set_title(name)
        ax.set_xlabel("Analytical")
        ax.set_ylabel("Surrogate")
        ax.grid(alpha=0.2)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.91, 0.12), frameon=False)
    fig.suptitle("Blind-test agreement across the five canonical benchmarks", y=1.01, fontweight="bold")
    fig.tight_layout()
    _save(fig, folder, "benchmark_accuracy")

    metrics = tables["primary_metrics.csv"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width / 2, metrics["rel_l2"], width, label=r"Relative $L_2$", color=COLORS["ml"])
    ax.bar(x + width / 2, metrics["rel_linf"], width, label=r"Relative $L_\infty$", color=COLORS["baseline"])
    ax.set_yscale("log")
    ax.set_xticks(x, [p.replace(" inverse", "\ninverse").replace(" implicit", "\nimplicit") for p in metrics["problem"]])
    ax.set_ylabel("Relative error")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, folder, "error_norms")

    ab = tables["ablations.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    f = extras["Fanno inverse"]
    m = f["forward_mach"]
    near = np.abs(m - 1.0) < 0.12
    axes[0].plot(m[near], f["forward_true"][near, 4], color=COLORS["exact"], label="Analytical")
    axes[0].plot(m[near], f["raw_pred"][near], color=COLORS["baseline"], linestyle=":", label="Raw MLP")
    axes[0].plot(m[near], f["forward_pred"][near, 4], color=COLORS["ml"], linestyle="--", label="Structured MLP")
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_xlabel("Mach number")
    axes[0].set_ylabel(r"$4fL^*/D$")
    axes[0].set_title("Near-sonic Fanno limit")
    axes[0].legend(frameon=False)
    ev = evidences[2]
    idx = np.argsort(ev.x_test[:, 1])
    tau = ev.x_test[idx, 1]
    exact = ev.y_test[idx]
    naive = extras["Oblique inverse"]["naive_pred"][idx]
    branch = ev.predict(ev.x_test)[idx]
    show = np.isclose(ev.x_test[idx, 0], np.unique(ev.x_test[:, 0])[len(np.unique(ev.x_test[:, 0])) // 2])
    axes[1].plot(tau[show], np.degrees(exact[show, 0]), color=COLORS["exact"], label="Weak exact")
    axes[1].plot(tau[show], np.degrees(exact[show, 1]), color=COLORS["exact"], linestyle="--", label="Strong exact")
    axes[1].plot(tau[show], np.degrees(naive[show]), color=COLORS["baseline"], linestyle=":", label="Naive MLP")
    axes[1].scatter(tau[show][::3], np.degrees(branch[show, 0][::3]), s=10, color=COLORS["ml"], label="Branch experts")
    axes[1].scatter(tau[show][::3], np.degrees(branch[show, 1][::3]), s=10, color=COLORS["ml"])
    axes[1].set_xlabel(r"Normalized turning, $\theta/\theta_{\max}$")
    axes[1].set_ylabel(r"Shock angle $\beta$ (deg)")
    axes[1].set_title("Why branch decomposition is required")
    axes[1].legend(frameon=False, fontsize=7.2)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, folder, "ablation_branch_singularity")

    scaling = tables["training_size_scaling.csv"]
    near_df = tables["near_singular_metrics.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for problem, frame in scaling.groupby("problem"):
        stats = frame.groupby("training_samples")["rel_l2"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(stats["training_samples"], stats["mean"], yerr=stats["std"].fillna(0.0), marker="o", markersize=3.5, capsize=2.5, label=problem.replace(" inverse", ""))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[0].set_title("Training-size sensitivity")
    axes[0].legend(frameon=False, fontsize=6.8, ncol=2)
    pivot = near_df.pivot(index="problem", columns="bin", values="mae")
    im = axes[1].imshow(np.log10(np.maximum(pivot.values, 1e-12)), aspect="auto", cmap="viridis")
    axes[1].set_yticks(np.arange(len(pivot)), [x.replace(" inverse", "") for x in pivot.index])
    axes[1].set_xticks(np.arange(4), ["closest", "near", "mid", "far"])
    axes[1].set_xlabel("Distance from limiting regime")
    axes[1].set_title(r"Local MAE, $\log_{10}$ scale")
    fig.colorbar(im, ax=axes[1], label=r"$\log_{10}(\mathrm{MAE})$")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.tight_layout()
    _save(fig, folder, "sensitivity_and_singular_error")

    timing = tables["timing.csv"]
    batch = int(timing["batch_size"].max())
    frame = timing[timing["batch_size"] == batch]
    pivot_t = frame.pivot(index="problem", columns="method", values="median_ms")
    fig, ax = plt.subplots(figsize=(7.8, 3.7))
    x = np.arange(len(pivot_t))
    width = 0.25
    for j, (method, label, color) in enumerate([
        ("bracketed_root", "Bracketed root", COLORS["exact"]),
        ("classical_interpolation", "Interpolation", COLORS["baseline"]),
        ("physics_guided_mlp", "MLP batch", COLORS["ml"]),
    ]):
        ax.bar(x + (j - 1) * width, pivot_t[method], width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x, [i.replace(" inverse", "\ninverse").replace(" implicit", "\nimplicit") for i in pivot_t.index])
    ax.set_ylabel(f"Median wall time for {batch:,} queries (ms)")
    ax.set_title("Measured CPU query cost; lower is better")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    _save(fig, folder, "measured_timing")

    # Individual problem figures used directly by the manuscript.
    r = extras["Rayleigh inverse"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    axes[0].plot(r["forward_mach"], r["forward_true"][:, 4], color=COLORS["exact"], label="Analytical")
    axes[0].plot(r["forward_mach"], r["forward_pred"][:, 4], color=COLORS["ml"], linestyle="--", label="MLP")
    ev = evidences[0]
    axes[1].plot(ev.y_test, ev.x_test[:, 0], color=COLORS["exact"], linewidth=1.0, label="Analytical branches")
    axes[1].scatter(ev.predict(ev.x_test)[::10], ev.x_test[::10, 0], s=8, color=COLORS["ml"], label="Expert MLPs")
    axes[0].set(xlabel="Mach number", ylabel=r"$T_0/T_0^*$", title="Forward relation")
    axes[1].set(xlabel="Mach number", ylabel=r"$T_0/T_0^*$", title="Branch-wise inverse")
    for ax in axes: ax.grid(alpha=0.2); ax.legend(frameon=False)
    fig.tight_layout(); _save(fig, folder, "rayleigh_results")

    f = extras["Fanno inverse"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    axes[0].loglog(np.abs(f["forward_mach"] - 1.0), np.maximum(f["forward_true"][:, 4], 1e-15), color=COLORS["exact"], label="Analytical")
    axes[0].loglog(np.abs(f["forward_mach"] - 1.0), np.maximum(f["forward_pred"][:, 4], 1e-15), color=COLORS["ml"], linestyle="--", label="Structured MLP")
    ev = evidences[1]
    axes[1].scatter(ev.y_test[::8], ev.predict(ev.x_test)[::8], s=9, color=COLORS["ml"])
    lo, hi = ev.y_test.min(), ev.y_test.max(); axes[1].plot([lo,hi],[lo,hi],"--",color=COLORS["exact"])
    axes[0].set(xlabel=r"$|M-1|$", ylabel=r"$4fL^*/D$", title="Exact sonic structure")
    axes[1].set(xlabel="Analytical Mach", ylabel="Predicted Mach", title="Branch-wise inverse")
    for ax in axes: ax.grid(alpha=0.2); axes[0].legend(frameon=False)
    fig.tight_layout(); _save(fig, folder, "fanno_results")

    o = extras["Oblique inverse"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    for mach in np.unique(o["direct_mach"]):
        mask = o["direct_mach"] == mach
        axes[0].plot(o["direct_q"][mask], np.degrees(o["direct_true"][mask]), color=COLORS["exact"], alpha=0.65)
        axes[0].plot(o["direct_q"][mask], np.degrees(o["direct_pred"][mask]), color=COLORS["ml"], linestyle="--", alpha=0.75)
    ev = evidences[2]; pred = ev.predict(ev.x_test)
    axes[1].scatter(np.degrees(ev.y_test[:,0]), np.degrees(pred[:,0]), s=7, color=COLORS["ml"], label="Weak")
    axes[1].scatter(np.degrees(ev.y_test[:,1]), np.degrees(pred[:,1]), s=7, color=COLORS["accent"], label="Strong")
    lo,hi=np.degrees(ev.y_test).min(),np.degrees(ev.y_test).max();axes[1].plot([lo,hi],[lo,hi],"--",color=COLORS["exact"])
    axes[0].set(xlabel=r"Normalized $\beta$ interval", ylabel=r"Turning angle $\theta$ (deg)", title="Hard anchored direct map")
    axes[1].set(xlabel=r"Analytical $\beta$ (deg)", ylabel=r"Predicted $\beta$ (deg)", title="Weak/strong inverse experts")
    for ax in axes: ax.grid(alpha=0.2); axes[1].legend(frameon=False)
    fig.tight_layout(); _save(fig, folder, "oblique_results")

    ev = evidences[3]; pred = ev.predict(ev.x_test); ctx=ev.test_context
    fig, axes = plt.subplots(1, 2, figsize=(8.8,3.5))
    sc=axes[0].scatter(ctx["ae"],ctx["pb"],c=pred,s=11,cmap="viridis");fig.colorbar(sc,ax=axes[0],label=r"Predicted $A_s/A_t$")
    axes[1].scatter(ev.y_test,pred,s=8,color=COLORS["ml"]);lo,hi=ev.y_test.min(),ev.y_test.max();axes[1].plot([lo,hi],[lo,hi],"--",color=COLORS["exact"])
    axes[0].set(xlabel=r"$A_e/A_t$",ylabel=r"$P_b/P_{01}$",title="Admissible internal-shock map")
    axes[1].set(xlabel=r"Analytical $A_s/A_t$",ylabel=r"Predicted $A_s/A_t$",title="Bounded inverse surrogate")
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout();_save(fig,folder,"nozzle_results")

    ev=evidences[4];pred=ev.predict(ev.x_test);ctx=ev.test_context
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.5))
    sc=axes[0].scatter(ctx["p4"],ctx["t4"],c=np.abs(shock_tube_residual(pred,ctx["p4"],ctx["t4"])),s=10,cmap="magma",norm=mpl.colors.LogNorm(vmin=max(1e-9,np.nanmin(np.abs(shock_tube_residual(pred,ctx["p4"],ctx["t4"])))),vmax=max(1e-8,np.nanmax(np.abs(shock_tube_residual(pred,ctx["p4"],ctx["t4"]))))));fig.colorbar(sc,ax=axes[0],label="Equation residual")
    axes[0].set_xscale("log")
    axes[1].scatter(ev.y_test,pred,s=8,color=COLORS["ml"]);lo,hi=ev.y_test.min(),ev.y_test.max();axes[1].plot([lo,hi],[lo,hi],"--",color=COLORS["exact"])
    axes[0].set(xlabel=r"$P_4/P_1$",ylabel=r"$T_4/T_1$",title="Two-parameter residual audit")
    axes[1].set(xlabel=r"Analytical $P_2/P_1$",ylabel=r"Predicted $P_2/P_1$",title="Bounded implicit surrogate")
    for ax in axes:ax.grid(alpha=0.2)
    fig.tight_layout();_save(fig,folder,"shock_tube_results")
