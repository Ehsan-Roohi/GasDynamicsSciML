"""Dimensional-scaling audit for a generalized ideal shock tube."""

from __future__ import annotations

import pickle
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning

from .modeling import ScaledMLP, regression_metrics, safe_logit
from .physics import shock_tube_pressure_ratio_general


RANGES = {
    "log_p4_p1": (np.log(1.5), np.log(300.0)),
    "log_t4_t1": (np.log(0.5), np.log(2.0)),
    "gamma": (1.25, 1.67),
    "gamma1": (1.25, 1.67),
    "gamma4": (1.25, 1.67),
    "log_r4_r1": (np.log(0.5), np.log(8.0)),
}


def _spec(dimension: int) -> list[str]:
    if dimension == 2:
        return ["log_p4_p1", "log_t4_t1"]
    if dimension == 3:
        return ["log_p4_p1", "log_t4_t1", "gamma"]
    if dimension == 4:
        return ["log_p4_p1", "log_t4_t1", "gamma1", "gamma4"]
    if dimension == 5:
        return ["log_p4_p1", "log_t4_t1", "gamma1", "gamma4", "log_r4_r1"]
    raise ValueError("The physical audit is defined for dimensions 2 through 5.")


def _bounds(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    names = _spec(dimension)
    return (
        np.array([RANGES[name][0] for name in names]),
        np.array([RANGES[name][1] for name in names]),
    )


def _sample(dimension: int, count: int, seed: int) -> np.ndarray:
    lo, hi = _bounds(dimension)
    unit = qmc.LatinHypercube(d=dimension, seed=seed).random(count)
    return qmc.scale(unit, lo, hi)


def _decode(x: np.ndarray, dimension: int):
    x = np.atleast_2d(np.asarray(x, float))
    p4 = np.exp(x[:, 0])
    t4 = np.exp(x[:, 1])
    if dimension == 2:
        g1 = g4 = np.full(len(x), 1.4)
        rr = np.ones(len(x))
    elif dimension == 3:
        g1 = g4 = x[:, 2]
        rr = np.ones(len(x))
    elif dimension == 4:
        g1, g4 = x[:, 2], x[:, 3]
        rr = np.ones(len(x))
    else:
        g1, g4 = x[:, 2], x[:, 3]
        rr = np.exp(x[:, 4])
    return p4, t4, g1, g4, rr


def _exact(x: np.ndarray, dimension: int) -> np.ndarray:
    p4, t4, g1, g4, rr = _decode(x, dimension)
    return np.array([
        shock_tube_pressure_ratio_general(float(p), float(t), float(a), float(b), float(r))
        for p, t, a, b, r in zip(p4, t4, g1, g4, rr)
    ])


def _fraction(p2: np.ndarray, x: np.ndarray) -> np.ndarray:
    p4 = np.exp(np.asarray(x)[:, 0])
    return (np.asarray(p2) - 1.0) / (p4 - 1.0)


def _restore(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    p4 = np.exp(np.asarray(x)[:, 0])
    return 1.0 + np.asarray(q) * (p4 - 1.0)


def _timed(fn, x: np.ndarray, repeats: int = 7) -> float:
    fn(x[: min(32, len(x))])
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(x)
        values.append(time.perf_counter() - start)
    return float(np.median(values))


def _figure(table: pd.DataFrame, output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9.5,
        "axes.labelsize": 10, "axes.titlesize": 10.5,
        "legend.fontsize": 8.2, "savefig.dpi": 400,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.45))
    d = table["dimension"].to_numpy()
    axes[0].semilogy(d, table["grid_64_storage_bytes"] / 2**20, "o-", color="#D55E00", label="64-node regular grid")
    axes[0].semilogy(d, table["mlp_serialized_bytes"] / 2**20, "s--", color="#0072B2", label="Serialized MLP")
    axes[0].set(xlabel="Physical input dimension", ylabel="Storage (MiB)", title="Fixed per-coordinate resolution")
    axes[0].set_xticks(d)
    axes[0].legend(frameon=False)
    axes[1].semilogy(d, table["interpolation_rel_l2"], "o-", color="#D55E00", label="Regular-grid interpolation")
    axes[1].semilogy(d, table["mlp_rel_l2"], "s--", color="#0072B2", label="Physics-bounded MLP")
    axes[1].set(xlabel="Physical input dimension", ylabel=r"Blind relative $L_2$ error", title="Comparable offline-state budget")
    axes[1].set_xticks(d)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "high_dimensional_scaling.pdf", bbox_inches="tight")
    fig.savefig(folder / "high_dimensional_scaling.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def run(output: Path, quick: bool = False, seed: int = 41) -> pd.DataFrame:
    """Run the matched-budget interpolation/MLP comparison."""
    output.mkdir(parents=True, exist_ok=True)
    budget = 512 if quick else 4096
    test_count = 256 if quick else 2048
    query_count = 1000 if quick else 5000
    max_iter = 120 if quick else 500
    rows: list[dict[str, float | int | str]] = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for dimension in range(2, 6):
        names = _spec(dimension)
        nodes = max(3, int(np.floor(budget ** (1.0 / dimension))))
        axes = [np.linspace(*RANGES[name], nodes) for name in names]
        meshes = np.meshgrid(*axes, indexing="ij")
        x_grid = np.column_stack([mesh.ravel() for mesh in meshes])
        grid_start = time.perf_counter()
        y_grid = _exact(x_grid, dimension)
        grid_generation_s = time.perf_counter() - grid_start
        q_grid = _fraction(y_grid, x_grid).reshape([nodes] * dimension)
        interpolator = RegularGridInterpolator(axes, q_grid, method="linear", bounds_error=True)

        x_train = _sample(dimension, budget, seed + 10 * dimension)
        y_train = _exact(x_train, dimension)
        model = ScaledMLP(hidden=(64, 64), seed=seed + dimension, max_iter=max_iter)
        fit_start = time.perf_counter()
        model.fit(x_train, safe_logit(_fraction(y_train, x_train)))
        fit_s = time.perf_counter() - fit_start

        x_test = _sample(dimension, test_count, seed + 100 + dimension)
        y_test = _exact(x_test, dimension)
        interp_predict = lambda z: _restore(interpolator(z), z)
        mlp_predict = lambda z: _restore(expit(model.predict(z)), z)
        interp_metrics = regression_metrics(y_test, interp_predict(x_test))
        mlp_metrics = regression_metrics(y_test, mlp_predict(x_test))

        x_query = np.resize(x_test, (query_count, dimension))
        grid_bytes = int(q_grid.nbytes + sum(axis.nbytes for axis in axes))
        mlp_bytes = len(pickle.dumps(model, protocol=5))
        mlp_parameters = int(sum(array.size for array in model.model.coefs_ + model.model.intercepts_))
        rows.append({
            "dimension": dimension,
            "physical_inputs": ";".join(names),
            "offline_budget": budget,
            "grid_nodes_per_axis": nodes,
            "grid_training_states": int(nodes**dimension),
            "mlp_training_states": budget,
            "grid_actual_storage_bytes": grid_bytes,
            "mlp_serialized_bytes": mlp_bytes,
            "mlp_trainable_parameters": mlp_parameters,
            "grid_16_states": int(16**dimension),
            "grid_16_storage_bytes": int(8 * 16**dimension + 8 * 16 * dimension),
            "grid_32_states": int(32**dimension),
            "grid_32_storage_bytes": int(8 * 32**dimension + 8 * 32 * dimension),
            "grid_64_states": int(64**dimension),
            "grid_64_storage_bytes": int(8 * 64**dimension + 8 * 64 * dimension),
            "interpolation_rel_l2": interp_metrics["rel_l2"],
            "interpolation_rel_linf": interp_metrics["rel_linf"],
            "mlp_rel_l2": mlp_metrics["rel_l2"],
            "mlp_rel_linf": mlp_metrics["rel_linf"],
            "interpolation_query_ms": 1.0e3 * _timed(interp_predict, x_query),
            "mlp_query_ms": 1.0e3 * _timed(mlp_predict, x_query),
            "grid_generation_s": grid_generation_s,
            "mlp_fit_s": fit_s,
            "blind_test_states": test_count,
        })
        print(f"dimension={dimension} grid={nodes}^{dimension} fit={fit_s:.2f}s", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output / "high_dimensional_scaling.csv", index=False)
    _figure(table, output)
    return table
