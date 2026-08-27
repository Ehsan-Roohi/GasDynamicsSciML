"""Application-level audits for differentiability and many-query use."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .modeling import regression_metrics
from .physics import nozzle_back_pressure, shock_tube_pressure_ratio
from .run_revision import _build_nozzle, _build_shock_tube


def _nozzle_prediction_and_gradient(model, area: float, back_pressure: float) -> tuple[float, float]:
    """Return bounded shock location and d(As/At)/d(Pb/P01)."""
    x = np.array([[area, back_pressure]], dtype=float)
    latent = float(model.predict(x)[0])
    latent_gradient = model.jacobian(x)[0, 0]
    q = float(expit(latent))
    shock = 1.0 + q * (area - 1.0)
    dshock_dpb = (area - 1.0) * q * (1.0 - q) * float(latent_gradient[1])
    return shock, dshock_dpb


def _safeguarded_newton(model, area: float, target_shock: float) -> tuple[float, int, float]:
    """Invert the differentiable nozzle surrogate with safeguarded Newton steps."""
    eps = 1.0e-8
    endpoints = sorted(
        (
            nozzle_back_pressure(area, 1.0 + eps),
            nozzle_back_pressure(area, area - eps),
        )
    )
    lower, upper = endpoints
    pressure = 0.5 * (lower + upper)
    for iteration in range(1, 16):
        shock, gradient = _nozzle_prediction_and_gradient(model, area, pressure)
        residual = shock - target_shock
        if abs(residual) < 1.0e-8:
            return pressure, iteration, residual
        # Shock location decreases monotonically as back pressure increases.
        if residual > 0.0:
            lower = pressure
        else:
            upper = pressure
        candidate = pressure - residual / gradient if abs(gradient) > 1.0e-12 else np.nan
        pressure = candidate if np.isfinite(candidate) and lower < candidate < upper else 0.5 * (lower + upper)
    shock, _ = _nozzle_prediction_and_gradient(model, area, pressure)
    return pressure, 15, shock - target_shock


def nozzle_gradient_audit(output: Path, quick: bool = False, seed: int = 11) -> pd.DataFrame:
    """Demonstrate smooth analytical gradients in a nozzle inverse-design task."""
    count = 250 if quick else 900
    evidence, _ = _build_nozzle(np.random.default_rng(seed + 300), count, seed + 3, quick)
    areas = (2.0, 3.25, 4.5)
    fractions = (0.25, 0.50, 0.75)
    rows: list[dict[str, float | int]] = []
    for area in areas:
        for fraction in fractions:
            target = 1.0 + fraction * (area - 1.0)
            exact_pressure = nozzle_back_pressure(area, target)
            surrogate_pressure, iterations, shock_residual = _safeguarded_newton(
                evidence.model, area, target
            )
            _, analytical_gradient = _nozzle_prediction_and_gradient(
                evidence.model, area, surrogate_pressure
            )
            pressure_span = abs(
                nozzle_back_pressure(area, 1.0 + 1.0e-8)
                - nozzle_back_pressure(area, area - 1.0e-8)
            )
            step = max(1.0e-7, 1.0e-5 * pressure_span)
            plus = evidence.predict(np.array([[area, surrogate_pressure + step]]))[0]
            minus = evidence.predict(np.array([[area, surrogate_pressure - step]]))[0]
            finite_gradient = float((plus - minus) / (2.0 * step))
            rows.append(
                {
                    "exit_area_ratio": area,
                    "target_fraction": fraction,
                    "target_shock_area_ratio": target,
                    "exact_back_pressure_ratio": exact_pressure,
                    "surrogate_back_pressure_ratio": surrogate_pressure,
                    "newton_iterations": iterations,
                    "shock_target_abs_error": abs(shock_residual),
                    "back_pressure_abs_error": abs(surrogate_pressure - exact_pressure),
                    "analytic_gradient": analytical_gradient,
                    "finite_difference_gradient": finite_gradient,
                    "gradient_relative_difference": abs(analytical_gradient - finite_gradient)
                    / max(abs(finite_gradient), 1.0e-14),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "nozzle_gradient_audit.csv", index=False)
    return table


def shock_tube_many_query_audit(
    output: Path, quick: bool = False, seed: int = 11
) -> pd.DataFrame:
    """Run a prescribed log-uniform 100,000-state shock-tube workload."""
    training_count = 250 if quick else 900
    query_count = 2_000 if quick else 100_000
    evidence, _ = _build_shock_tube(
        np.random.default_rng(seed + 400), training_count, seed + 4, quick
    )
    rng = np.random.default_rng(20260827)
    p4 = np.exp(rng.uniform(np.log(1.5), np.log(300.0), query_count))
    t4 = np.exp(rng.uniform(np.log(0.5), np.log(2.0), query_count))
    x = np.column_stack([np.log(p4), np.log(t4)])

    evidence.predict(x[:32])
    start = time.perf_counter()
    mlp = evidence.predict(x)
    mlp_seconds = time.perf_counter() - start

    [shock_tube_pressure_ratio(float(p), float(t)) for p, t in zip(p4[:32], t4[:32])]
    start = time.perf_counter()
    exact = np.array(
        [shock_tube_pressure_ratio(float(p), float(t)) for p, t in zip(p4, t4)]
    )
    brent_seconds = time.perf_counter() - start

    metrics = regression_metrics(exact, mlp)
    rows = []
    for method, values, elapsed in (
        ("bracketed_brent", exact, brent_seconds),
        ("physics_guided_mlp", mlp, mlp_seconds),
    ):
        rows.append(
            {
                "method": method,
                "query_count": query_count,
                "p4_p1_distribution": "log_uniform_[1.5,300]",
                "t4_t1_distribution": "log_uniform_[0.5,2]",
                "mean_p2_p1": float(np.mean(values)),
                "std_p2_p1": float(np.std(values)),
                "q05_p2_p1": float(np.quantile(values, 0.05)),
                "q50_p2_p1": float(np.quantile(values, 0.50)),
                "q95_p2_p1": float(np.quantile(values, 0.95)),
                "elapsed_seconds": elapsed,
                "speedup_vs_brent": brent_seconds / elapsed,
                "rel_l2_vs_brent": 0.0 if method == "bracketed_brent" else metrics["rel_l2"],
                "valid_pressure_rate": float(np.mean((values > 1.0) & (values < p4))),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "shock_tube_many_query.csv", index=False)
    return table


def run(output: Path, quick: bool = False, seed: int = 11) -> dict[str, object]:
    """Run both application audits and write their machine-readable evidence."""
    output.mkdir(parents=True, exist_ok=True)
    nozzle = nozzle_gradient_audit(output, quick=quick, seed=seed)
    shock = shock_tube_many_query_audit(output, quick=quick, seed=seed)
    summary = {
        "nozzle_cases": int(len(nozzle)),
        "nozzle_max_iterations": int(nozzle["newton_iterations"].max()),
        "nozzle_max_shock_target_abs_error": float(nozzle["shock_target_abs_error"].max()),
        "nozzle_max_back_pressure_abs_error": float(nozzle["back_pressure_abs_error"].max()),
        "nozzle_max_gradient_relative_difference": float(nozzle["gradient_relative_difference"].max()),
        "shock_tube_queries": int(shock["query_count"].iloc[0]),
        "shock_tube_rel_l2": float(shock.loc[shock["method"] == "physics_guided_mlp", "rel_l2_vs_brent"].iloc[0]),
        "shock_tube_speedup": float(shock.loc[shock["method"] == "physics_guided_mlp", "speedup_vs_brent"].iloc[0]),
    }
    (output / "application_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
