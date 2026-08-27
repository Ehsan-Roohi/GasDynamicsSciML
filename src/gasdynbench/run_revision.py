"""Generate all reviewer-facing evidence from one deterministic command."""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.interpolate import LinearNDInterpolator, PchipInterpolator
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning

from .modeling import ScaledMLP, regression_metrics, safe_logit
from .physics import (
    GAMMA,
    entropy_over_r,
    fanno_inverse_fld,
    fanno_ratios,
    mach_angle,
    nozzle_back_pressure,
    nozzle_shock_area,
    oblique_beta,
    oblique_detachment,
    oblique_theta,
    rayleigh_inverse_t0,
    rayleigh_ratios,
    shock_tube_pressure_ratio,
    shock_tube_residual,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class Evidence:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    model: ScaledMLP
    predict: callable
    exact_one: callable
    interpolation_predict: callable
    test_context: dict[str, np.ndarray]


def _mlp(seed: int, quick: bool, hidden: tuple[int, ...] = (32, 32)) -> ScaledMLP:
    return ScaledMLP(hidden=hidden, seed=seed, max_iter=220 if quick else 700)


def _sample_mach(rng: np.random.Generator, n: int, low: float = 0.2,
                 high: float = 5.0) -> np.ndarray:
    n_log = int(0.7 * n)
    base = np.exp(rng.uniform(np.log(low), np.log(high), n_log))
    sonic = np.clip(1.0 + rng.normal(0.0, 0.12, n - n_log), low, high)
    m = np.concatenate([base, sonic])
    rng.shuffle(m)
    return m


def _linear_baseline(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape[1] == 1:
        order = np.argsort(x[:, 0])
        unique_x, unique_idx = np.unique(x[order, 0], return_index=True)
        yy = y[order][unique_idx]
        interpolator = PchipInterpolator(unique_x, yy, axis=0, extrapolate=False)
        return lambda z: np.asarray(interpolator(np.asarray(z)[:, 0]))
    interpolator = LinearNDInterpolator(x, y, fill_value=np.nan)
    return lambda z: np.asarray(interpolator(np.asarray(z)))


def _build_rayleigh(rng: np.random.Generator, n: int, seed: int, quick: bool) -> tuple[Evidence, dict]:
    # The inverse uses T0/T0*, whose extremum is exactly at M=1. Static T/T*
    # peaks at M=1/sqrt(gamma) and is therefore not a valid two-expert sonic split.
    n_sub = n // 2
    m_sub = rng.uniform(0.2, 0.985, n_sub)
    m_sup = rng.uniform(1.015, 5.0, n - n_sub)
    m_train = np.concatenate([m_sub, m_sup])
    branch_train = np.concatenate([np.zeros(n_sub), np.ones(n - n_sub)])
    t0_train = rayleigh_ratios(m_train)[:, 4]
    x_train = np.column_stack([t0_train, branch_train])
    q_train = np.where(branch_train == 0, (m_train - 0.2) / 0.8, (m_train - 1.0) / 4.0)
    sonic_feature = np.sqrt(np.maximum(1.0 - t0_train, 0.0)).reshape(-1, 1)
    sub_model = _mlp(seed, quick).fit(sonic_feature[:n_sub], safe_logit(q_train[:n_sub]))
    sup_model = _mlp(seed + 1, quick).fit(sonic_feature[n_sub:], safe_logit(q_train[n_sub:]))
    model = (sub_model, sup_model)

    m_sub_t = np.linspace(0.205, 0.98, 450)
    m_sup_t = np.linspace(1.02, 4.95, 450)
    m_test = np.concatenate([m_sub_t, m_sup_t])
    branch_test = np.concatenate([np.zeros_like(m_sub_t), np.ones_like(m_sup_t)])
    t0_test = rayleigh_ratios(m_test)[:, 4]
    x_test = np.column_stack([t0_test, branch_test])

    def predict(x):
        x = np.asarray(x, float)
        feature = np.sqrt(np.maximum(1.0 - x[:, 0], 0.0)).reshape(-1, 1)
        q = np.empty(len(x))
        sub = x[:, 1] < 0.5
        if np.any(sub):
            q[sub] = expit(sub_model.predict(feature[sub]))
        if np.any(~sub):
            q[~sub] = expit(sup_model.predict(feature[~sub]))
        return np.where(x[:, 1] < 0.5, 0.2 + 0.8 * q, 1.0 + 4.0 * q)

    sub_interp = PchipInterpolator(np.sort(t0_train[:n_sub]), m_train[:n_sub][np.argsort(t0_train[:n_sub])], extrapolate=False)
    sup_interp = PchipInterpolator(np.sort(t0_train[n_sub:]), m_train[n_sub:][np.argsort(t0_train[n_sub:])], extrapolate=False)
    def interp(x):
        x = np.asarray(x, float)
        out = np.empty(len(x))
        sub = x[:, 1] < 0.5
        out[sub] = sub_interp(x[sub, 0])
        out[~sub] = sup_interp(x[~sub, 0])
        return out
    exact_one = lambda row: rayleigh_inverse_t0(float(row[0]), "subsonic" if row[1] < 0.5 else "supersonic")
    evidence = Evidence("Rayleigh inverse", x_train, m_train, x_test, m_test, model, predict, exact_one, interp, {"branch": branch_test})

    # Positive forward surrogate and entropy audit.
    m_f = _sample_mach(rng, n)
    x_f = np.column_stack([m_f, np.log(m_f), m_f**2])
    y_f = rayleigh_ratios(m_f)
    fwd = _mlp(seed + 101, quick).fit(x_f, np.log(y_f))
    m_ft = np.geomspace(0.205, 4.95, 1000)
    x_ft = np.column_stack([m_ft, np.log(m_ft), m_ft**2])
    y_ft = rayleigh_ratios(m_ft)
    pred_ft = np.exp(fwd.predict(x_ft))
    s_true = entropy_over_r(y_ft[:, 0], y_ft[:, 1])
    s_pred = entropy_over_r(pred_ft[:, 0], pred_ft[:, 1])
    extra = {
        "forward_true": y_ft,
        "forward_pred": pred_ft,
        "forward_mach": m_ft,
        "forward_metrics": regression_metrics(y_ft, pred_ft),
        "entropy_linf": float(np.max(np.abs(s_pred - s_true))),
        "entropy_peak_mach_error": float(abs(m_ft[np.argmax(s_pred)] - m_ft[np.argmax(s_true)])),
        "positive_rate": float(np.mean(pred_ft > 0.0)),
    }
    return evidence, extra


def _build_fanno(rng: np.random.Generator, n: int, seed: int, quick: bool) -> tuple[Evidence, dict]:
    n_sub = n // 2
    m_sub = rng.uniform(0.2, 0.985, n_sub)
    m_sup = rng.uniform(1.015, 5.0, n - n_sub)
    m_train = np.concatenate([m_sub, m_sup])
    branch_train = np.concatenate([np.zeros(n_sub), np.ones(n - n_sub)])
    fld_train = fanno_ratios(m_train)[:, 4]
    x_train = np.column_stack([np.sqrt(fld_train), branch_train])
    q_train = np.where(branch_train == 0, (m_train - 0.2) / 0.8, (m_train - 1.0) / 4.0)
    sub_model = _mlp(seed, quick).fit(x_train[:n_sub, :1], safe_logit(q_train[:n_sub]))
    sup_model = _mlp(seed + 1, quick).fit(x_train[n_sub:, :1], safe_logit(q_train[n_sub:]))
    model = (sub_model, sup_model)

    m_sub_t = np.concatenate([np.linspace(0.205, 0.9, 300), 1.0 - np.geomspace(1e-4, 0.09, 220)])
    m_sup_t = np.concatenate([1.0 + np.geomspace(1e-4, 0.09, 220), np.linspace(1.1, 4.95, 300)])
    m_test = np.concatenate([m_sub_t, m_sup_t])
    branch_test = np.concatenate([np.zeros_like(m_sub_t), np.ones_like(m_sup_t)])
    fld_test = fanno_ratios(m_test)[:, 4]
    x_test = np.column_stack([np.sqrt(fld_test), branch_test])

    def predict(x):
        x = np.asarray(x, float)
        q = np.empty(len(x))
        sub = x[:, 1] < 0.5
        if np.any(sub):
            q[sub] = expit(sub_model.predict(x[sub, :1]))
        if np.any(~sub):
            q[~sub] = expit(sup_model.predict(x[~sub, :1]))
        return np.where(x[:, 1] < 0.5, 0.2 + 0.8 * q, 1.0 + 4.0 * q)

    sub_order = np.argsort(x_train[:n_sub, 0])
    sup_order = np.argsort(x_train[n_sub:, 0])
    sub_interp = PchipInterpolator(x_train[:n_sub, 0][sub_order], m_train[:n_sub][sub_order], extrapolate=False)
    sup_interp = PchipInterpolator(x_train[n_sub:, 0][sup_order], m_train[n_sub:][sup_order], extrapolate=False)
    def interp(x):
        x = np.asarray(x, float)
        out = np.empty(len(x))
        sub = x[:, 1] < 0.5
        out[sub] = sub_interp(x[sub, 0])
        out[~sub] = sup_interp(x[~sub, 0])
        return out
    exact_one = lambda row: fanno_inverse_fld(float(row[0]) ** 2, "subsonic" if row[1] < 0.5 else "supersonic")
    evidence = Evidence("Fanno inverse", x_train, m_train, x_test, m_test, model, predict, exact_one, interp, {"branch": branch_test, "fld": fld_test})

    # Forward model uses the exact quadratic sonic envelope for fLD, preventing
    # the negative near-sonic values produced by the submitted raw-output code.
    m_f = _sample_mach(rng, n)
    y_f = fanno_ratios(m_f)
    distance2 = np.maximum((m_f - 1.0) ** 2, 1.0e-14)
    latent = np.column_stack([np.log(y_f[:, :4]), np.log(np.maximum(y_f[:, 4] / distance2, 1e-14))])
    x_f = np.column_stack([m_f, np.log(m_f), m_f**2])
    fwd = _mlp(seed + 102, quick).fit(x_f, latent)
    m_ft = np.concatenate([np.geomspace(0.205, 0.95, 420), 1.0 + np.geomspace(1e-5, 3.95, 580)])
    x_ft = np.column_stack([m_ft, np.log(m_ft), m_ft**2])
    z = fwd.predict(x_ft)
    pred_ft = np.column_stack([np.exp(z[:, :4]), (m_ft - 1.0) ** 2 * np.exp(z[:, 4])])
    y_ft = fanno_ratios(m_ft)
    s_true = entropy_over_r(y_ft[:, 0], y_ft[:, 1])
    s_pred = entropy_over_r(pred_ft[:, 0], pred_ft[:, 1])

    # Naive raw fLD ablation on the identical samples.
    raw = _mlp(seed + 202, quick).fit(x_f, y_f[:, 4])
    raw_pred = raw.predict(x_ft)
    near = np.abs(m_ft - 1.0) < 0.05
    extra = {
        "forward_true": y_ft,
        "forward_pred": pred_ft,
        "forward_mach": m_ft,
        "forward_metrics": regression_metrics(y_ft, pred_ft),
        "entropy_linf": float(np.max(np.abs(s_pred - s_true))),
        "entropy_peak_mach_error": float(abs(m_ft[np.argmax(s_pred)] - m_ft[np.argmax(s_true)])),
        "nonnegative_rate": float(np.mean(pred_ft[:, 4] >= 0.0)),
        "structured_near_sonic_mae": float(np.mean(np.abs(pred_ft[near, 4] - y_ft[near, 4]))),
        "raw_near_sonic_mae": float(np.mean(np.abs(raw_pred[near] - y_ft[near, 4]))),
        "raw_negative_rate_near_sonic": float(np.mean(raw_pred[near] < 0.0)),
        "raw_pred": raw_pred,
    }
    return evidence, extra


def _oblique_samples(rng: np.random.Generator, n: int):
    m = rng.uniform(1.25, 8.0, n)
    tau = rng.uniform(0.02, 0.96, n)
    weak = np.empty(n)
    strong = np.empty(n)
    theta = np.empty(n)
    for i, (mi, ti) in enumerate(zip(m, tau)):
        _, tmax = oblique_detachment(round(float(mi), 10))
        theta[i] = ti * tmax
        weak[i] = oblique_beta(float(mi), float(theta[i]), "weak")
        strong[i] = oblique_beta(float(mi), float(theta[i]), "strong")
    return m, tau, theta, weak, strong


def _build_oblique(rng: np.random.Generator, n: int, seed: int, quick: bool) -> tuple[Evidence, dict]:
    m, tau, theta, weak, strong = _oblique_samples(rng, n)
    mu = mach_angle(m)
    span = 0.5 * np.pi - mu
    qweak = (weak - mu) / span
    qstrong = (strong - mu) / span
    x_train = np.column_stack([m, tau])
    y_latent = np.column_stack([safe_logit(qweak), safe_logit(qstrong)])
    model = _mlp(seed, quick, (64, 64)).fit(x_train, y_latent)

    m_t = np.linspace(1.27, 7.95, 34)
    tau_t = np.linspace(0.025, 0.95, 25)
    mg, tg = np.meshgrid(m_t, tau_t)
    mt, taut = mg.ravel(), tg.ravel()
    theta_t = np.empty_like(mt)
    weak_t = np.empty_like(mt)
    strong_t = np.empty_like(mt)
    for i, (mi, ti) in enumerate(zip(mt, taut)):
        _, tmax = oblique_detachment(round(float(mi), 10))
        theta_t[i] = ti * tmax
        weak_t[i] = oblique_beta(float(mi), float(theta_t[i]), "weak")
        strong_t[i] = oblique_beta(float(mi), float(theta_t[i]), "strong")
    x_test = np.column_stack([mt, taut])
    y_test = np.column_stack([weak_t, strong_t])

    def predict(x):
        x = np.asarray(x, float)
        z = model.predict(x)
        mui = mach_angle(x[:, 0])
        spani = 0.5 * np.pi - mui
        return np.column_stack([mui + spani * expit(z[:, 0]), mui + spani * expit(z[:, 1])])

    interp_latent = _linear_baseline(x_train, y_latent)
    def interp(x):
        z = interp_latent(x)
        mui = mach_angle(np.asarray(x)[:, 0])
        spani = 0.5 * np.pi - mui
        return np.column_stack([mui + spani * expit(z[:, 0]), mui + spani * expit(z[:, 1])])

    def exact_one(row):
        mi, ti = float(row[0]), float(row[1])
        _, tmax = oblique_detachment(round(mi, 10))
        th = ti * tmax
        return np.array([oblique_beta(mi, th, "weak"), oblique_beta(mi, th, "strong")])

    evidence = Evidence("Oblique inverse", x_train, np.column_stack([weak, strong]), x_test, y_test, model, predict, exact_one, interp, {"theta": theta_t, "tau": taut})

    # A naive single-output network sees duplicated inputs with incompatible
    # weak/strong labels and therefore regresses toward the unphysical average.
    x_naive = np.vstack([x_train, x_train])
    y_naive = np.concatenate([weak, strong])
    naive = _mlp(seed + 301, quick).fit(x_naive, y_naive)
    naive_pred = naive.predict(x_test)
    branch_pred = predict(x_test)
    naive_distance = np.minimum(np.abs(naive_pred - weak_t), np.abs(naive_pred - strong_t))

    # Direct theta model with a hard Mach-wave/normal-shock output envelope.
    n_direct = n
    md = rng.uniform(1.2, 8.0, n_direct)
    qd = rng.uniform(0.01, 0.99, n_direct)
    mud = mach_angle(md)
    bd = mud + qd * (0.5 * np.pi - mud)
    thd = oblique_theta(md, bd)
    gd = np.log(np.maximum(thd / (qd * (1.0 - qd)), 1.0e-14))
    direct = _mlp(seed + 302, quick).fit(np.column_stack([md, qd]), gd)
    md_t = np.repeat(np.array([1.5, 2.0, 3.0, 5.0, 8.0]), 240)
    qd_t = np.tile(np.linspace(0.0, 1.0, 240), 5)
    mud_t = mach_angle(md_t)
    bd_t = mud_t + qd_t * (0.5 * np.pi - mud_t)
    th_true = oblique_theta(md_t, bd_t)
    th_pred = qd_t * (1.0 - qd_t) * np.exp(direct.predict(np.column_stack([md_t, qd_t])))

    extra = {
        "naive_branch_distance_mae": float(np.mean(naive_distance)),
        "branchwise_mae": float(np.mean(np.abs(branch_pred - y_test))),
        "naive_pred": naive_pred,
        "direct_mach": md_t,
        "direct_q": qd_t,
        "direct_true": th_true,
        "direct_pred": th_pred,
        "direct_metrics": regression_metrics(th_true, th_pred),
        "anchor_linf": float(max(np.max(np.abs(th_pred[qd_t == 0.0])), np.max(np.abs(th_pred[qd_t == 1.0])))),
        "valid_branch_rate": float(np.mean((branch_pred[:, 0] >= mach_angle(mt)) & (branch_pred[:, 1] <= 0.5 * np.pi) & (branch_pred[:, 0] < branch_pred[:, 1]))),
    }
    return evidence, extra


def _nozzle_samples(rng: np.random.Generator, n: int):
    ae = rng.uniform(1.5, 5.0, n)
    q = rng.uniform(0.025, 0.975, n)
    shock = 1.0 + q * (ae - 1.0)
    pb = np.array([nozzle_back_pressure(float(a), float(s)) for a, s in zip(ae, shock)])
    return ae, pb, q, shock


def _build_nozzle(rng: np.random.Generator, n: int, seed: int, quick: bool) -> tuple[Evidence, dict]:
    ae, pb, q, shock = _nozzle_samples(rng, n)
    x_train = np.column_stack([ae, pb])
    model = _mlp(seed, quick, (64, 64)).fit(x_train, safe_logit(q))
    ae_t = np.linspace(1.55, 4.95, 32)
    q_t = np.linspace(0.035, 0.965, 28)
    ag, qg = np.meshgrid(ae_t, q_t)
    aet, qt = ag.ravel(), qg.ravel()
    shock_t = 1.0 + qt * (aet - 1.0)
    pb_t = np.array([nozzle_back_pressure(float(a), float(s)) for a, s in zip(aet, shock_t)])
    x_test = np.column_stack([aet, pb_t])

    def predict(x):
        x = np.asarray(x, float)
        return 1.0 + expit(model.predict(x)) * (x[:, 0] - 1.0)

    latent_interp = _linear_baseline(x_train, safe_logit(q))
    def interp(x):
        x = np.asarray(x, float)
        return 1.0 + expit(latent_interp(x)) * (x[:, 0] - 1.0)

    exact_one = lambda row: nozzle_shock_area(float(row[0]), float(row[1]))
    evidence = Evidence("Nozzle inverse", x_train, shock, x_test, shock_t, model, predict, exact_one, interp, {"ae": aet, "pb": pb_t, "q": qt})
    pred = predict(x_test)
    back_reconstructed = np.array([nozzle_back_pressure(float(a), float(s)) for a, s in zip(aet, pred)])
    extra = {
        "back_pressure_residual_linf": float(np.max(np.abs(back_reconstructed - pb_t))),
        "valid_location_rate": float(np.mean((pred >= 1.0) & (pred <= aet))),
    }
    return evidence, extra


def _shock_tube_samples(rng: np.random.Generator, n: int):
    p4 = np.exp(rng.uniform(np.log(1.5), np.log(300.0), n))
    t4 = np.exp(rng.uniform(np.log(0.5), np.log(2.0), n))
    p2 = np.array([shock_tube_pressure_ratio(float(p), float(t)) for p, t in zip(p4, t4)])
    return p4, t4, p2


def _build_shock_tube(rng: np.random.Generator, n: int, seed: int, quick: bool) -> tuple[Evidence, dict]:
    p4, t4, p2 = _shock_tube_samples(rng, n)
    x_train = np.column_stack([np.log(p4), np.log(t4)])
    q = (p2 - 1.0) / (p4 - 1.0)
    model = _mlp(seed, quick, (64, 64)).fit(x_train, safe_logit(q))
    p4_t = np.geomspace(1.55, 295.0, 40)
    t4_t = np.geomspace(0.52, 1.95, 24)
    pg, tg = np.meshgrid(p4_t, t4_t)
    p4v, t4v = pg.ravel(), tg.ravel()
    p2v = np.array([shock_tube_pressure_ratio(float(p), float(t)) for p, t in zip(p4v, t4v)])
    x_test = np.column_stack([np.log(p4v), np.log(t4v)])

    def predict(x):
        x = np.asarray(x, float)
        p4i = np.exp(x[:, 0])
        return 1.0 + expit(model.predict(x)) * (p4i - 1.0)

    latent_interp = _linear_baseline(x_train, safe_logit(q))
    def interp(x):
        x = np.asarray(x, float)
        p4i = np.exp(x[:, 0])
        return 1.0 + expit(latent_interp(x)) * (p4i - 1.0)

    exact_one = lambda row: shock_tube_pressure_ratio(float(np.exp(row[0])), float(np.exp(row[1])))
    evidence = Evidence("Shock tube implicit", x_train, p2, x_test, p2v, model, predict, exact_one, interp, {"p4": p4v, "t4": t4v})
    pred = predict(x_test)
    residual = shock_tube_residual(pred, p4v, t4v)
    extra = {
        "equation_residual_l2": float(np.linalg.norm(residual) / np.linalg.norm(p4v)),
        "equation_residual_linf": float(np.nanmax(np.abs(residual)) / np.max(p4v)),
        "valid_pressure_rate": float(np.mean((pred > 1.0) & (pred < p4v))),
    }
    return evidence, extra


def _evaluate(evidences: list[Evidence]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    baseline_rows = []
    for ev in evidences:
        pred = ev.predict(ev.x_test)
        metric_rows.append({"problem": ev.name, "model": "physics_guided_mlp", **regression_metrics(ev.y_test, pred)})
        base = ev.interpolation_predict(ev.x_test)
        valid = np.all(np.isfinite(base), axis=1) if np.asarray(base).ndim > 1 else np.isfinite(base)
        if np.any(valid):
            baseline_rows.append({"problem": ev.name, "model": "classical_interpolation", "coverage": float(np.mean(valid)), **regression_metrics(np.asarray(ev.y_test)[valid], np.asarray(base)[valid])})
        baseline_rows.append({"problem": ev.name, "model": "physics_guided_mlp", "coverage": 1.0, **regression_metrics(ev.y_test, pred)})
    return pd.DataFrame(metric_rows), pd.DataFrame(baseline_rows)


def _near_singular(evidences: list[Evidence]) -> pd.DataFrame:
    rows = []
    for ev in evidences:
        pred = ev.predict(ev.x_test)
        if ev.name.startswith("Rayleigh"):
            distance = np.abs(ev.y_test - 1.0)
        elif ev.name.startswith("Fanno"):
            distance = np.abs(ev.y_test - 1.0)
        elif ev.name.startswith("Oblique"):
            distance = 1.0 - ev.test_context["tau"]
        elif ev.name.startswith("Nozzle"):
            q = ev.test_context["q"]
            distance = np.minimum(q, 1.0 - q)
        else:
            distance = 1.0 / ev.test_context["p4"]
        err = np.abs(np.asarray(pred) - np.asarray(ev.y_test))
        if err.ndim > 1:
            err = np.max(err, axis=1)
        quantiles = np.quantile(distance, [0.0, 0.2, 0.5, 0.8, 1.0])
        for i in range(4):
            mask = (distance >= quantiles[i]) & (distance <= quantiles[i + 1] if i == 3 else distance < quantiles[i + 1])
            rows.append({
                "problem": ev.name,
                "bin": i + 1,
                "distance_min": float(quantiles[i]),
                "distance_max": float(quantiles[i + 1]),
                "count": int(np.sum(mask)),
                "mae": float(np.mean(err[mask])),
                "linf": float(np.max(err[mask])),
            })
    return pd.DataFrame(rows)


def _range_generalization(evidences: list[Evidence], quick: bool) -> pd.DataFrame:
    """Train on an interior box and test on omitted edge bands.

    This is controlled range extrapolation inside the globally declared
    physical domain; it is not evidence for unrestricted out-of-domain use.
    """
    rows = []
    for j, ev in enumerate(evidences):
        xtr, ytr = np.asarray(ev.x_train), np.asarray(ev.y_train)
        xte, yte = np.asarray(ev.x_test), np.asarray(ev.y_test)
        seed = 811 + j
        if ev.name.startswith("Rayleigh") or ev.name.startswith("Fanno"):
            branch = xtr[:, 1] < 0.5
            interior = (branch & (ytr >= 0.30) & (ytr <= 0.90)) | (~branch & (ytr >= 1.20) & (ytr <= 4.00))
            branch_t = xte[:, 1] < 0.5
            edge = (branch_t & ((yte < 0.30) | (yte > 0.90))) | (~branch_t & ((yte < 1.20) | (yte > 4.00)))
            q = np.where(branch, (ytr - 0.2) / 0.8, (ytr - 1.0) / 4.0)
            feature = np.sqrt(np.maximum(1.0 - xtr[:, 0], 0.0)).reshape(-1, 1) if ev.name.startswith("Rayleigh") else xtr[:, :1]
            feature_t = np.sqrt(np.maximum(1.0 - xte[:, 0], 0.0)).reshape(-1, 1) if ev.name.startswith("Rayleigh") else xte[:, :1]
            sub = branch & interior; sup = (~branch) & interior
            sub_model = _mlp(seed, quick).fit(feature[sub], safe_logit(q[sub]))
            sup_model = _mlp(seed + 1, quick).fit(feature[sup], safe_logit(q[sup]))
            pred = np.empty(len(xte)); pred[branch_t] = 0.2 + 0.8 * expit(sub_model.predict(feature_t[branch_t])); pred[~branch_t] = 1.0 + 4.0 * expit(sup_model.predict(feature_t[~branch_t]))
            valid = ((pred >= 0.2) & (pred <= 1.0) & branch_t) | ((pred >= 1.0) & (pred <= 5.0) & ~branch_t)
        elif ev.name.startswith("Oblique"):
            interior = (xtr[:, 0] >= 1.5) & (xtr[:, 0] <= 6.5) & (xtr[:, 1] >= 0.10) & (xtr[:, 1] <= 0.85)
            edge = (xte[:, 0] < 1.5) | (xte[:, 0] > 6.5) | (xte[:, 1] < 0.10) | (xte[:, 1] > 0.85)
            mu = mach_angle(xtr[:, 0]); span = 0.5 * np.pi - mu
            latent = np.column_stack([safe_logit((ytr[:, 0] - mu) / span), safe_logit((ytr[:, 1] - mu) / span)])
            model = _mlp(seed, quick, (64, 64)).fit(xtr[interior], latent[interior])
            z = model.predict(xte); mu_t = mach_angle(xte[:, 0]); span_t = 0.5 * np.pi - mu_t
            pred = np.column_stack([mu_t + span_t * expit(z[:, 0]), mu_t + span_t * expit(z[:, 1])])
            valid = (pred[:, 0] >= mu_t) & (pred[:, 0] < pred[:, 1]) & (pred[:, 1] <= 0.5 * np.pi)
        elif ev.name.startswith("Nozzle"):
            q = (ytr - 1.0) / (xtr[:, 0] - 1.0)
            interior = (xtr[:, 0] >= 2.0) & (xtr[:, 0] <= 4.5) & (q >= 0.10) & (q <= 0.90)
            q_t = (yte - 1.0) / (xte[:, 0] - 1.0)
            edge = (xte[:, 0] < 2.0) | (xte[:, 0] > 4.5) | (q_t < 0.10) | (q_t > 0.90)
            model = _mlp(seed, quick, (64, 64)).fit(xtr[interior], safe_logit(q[interior]))
            pred = 1.0 + expit(model.predict(xte)) * (xte[:, 0] - 1.0)
            valid = (pred >= 1.0) & (pred <= xte[:, 0])
        else:
            p4, t4 = np.exp(xtr[:, 0]), np.exp(xtr[:, 1])
            interior = (p4 >= 2.0) & (p4 <= 200.0) & (t4 >= 0.65) & (t4 <= 1.65)
            p4_t, t4_t = np.exp(xte[:, 0]), np.exp(xte[:, 1])
            edge = (p4_t < 2.0) | (p4_t > 200.0) | (t4_t < 0.65) | (t4_t > 1.65)
            q = (ytr - 1.0) / (p4 - 1.0)
            model = _mlp(seed, quick, (64, 64)).fit(xtr[interior], safe_logit(q[interior]))
            pred = 1.0 + expit(model.predict(xte)) * (p4_t - 1.0)
            valid = (pred > 1.0) & (pred < p4_t)
        rows.append({
            "problem": ev.name,
            "interior_training_count": int(np.sum(interior)),
            "edge_test_count": int(np.sum(edge)),
            "valid_rate": float(np.mean(valid[edge])),
            **regression_metrics(yte[edge], pred[edge]),
        })
    return pd.DataFrame(rows)


def _timing(evidences: list[Evidence], quick: bool) -> pd.DataFrame:
    rows = []
    sizes = [1, 100, 1000 if quick else 5000]
    repeats = 3 if quick else 7
    for ev in evidences:
        for size in sizes:
            idx = np.arange(size) % len(ev.x_test)
            x = ev.x_test[idx]
            ev.predict(x[: min(10, size)])
            ml_times = []
            exact_times = []
            interp_times = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                ev.predict(x)
                ml_times.append((time.perf_counter_ns() - start) / 1e6)
                start = time.perf_counter_ns()
                [ev.exact_one(row) for row in x]
                exact_times.append((time.perf_counter_ns() - start) / 1e6)
                start = time.perf_counter_ns()
                ev.interpolation_predict(x)
                interp_times.append((time.perf_counter_ns() - start) / 1e6)
            for name, values in [("physics_guided_mlp", ml_times), ("bracketed_root", exact_times), ("classical_interpolation", interp_times)]:
                rows.append({"problem": ev.name, "batch_size": size, "method": name, "median_ms": float(np.median(values)), "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25))})
    return pd.DataFrame(rows)


def build_main_evidences(
    quick: bool = False, seed: int = 11
) -> tuple[list[Evidence], dict[str, dict]]:
    """Build the five canonical models once for shared evaluation/timing audits."""
    n = 250 if quick else 900
    builders = [
        ("Rayleigh inverse", _build_rayleigh),
        ("Fanno inverse", _build_fanno),
        ("Oblique inverse", _build_oblique),
        ("Nozzle inverse", _build_nozzle),
        ("Shock tube implicit", _build_shock_tube),
    ]
    evidences: list[Evidence] = []
    extras: dict[str, dict] = {}
    for i, (label, builder) in enumerate(builders):
        print(f"building {label} ({i + 1}/{len(builders)})", flush=True)
        rng = np.random.default_rng(seed + 100 * i)
        evidence, extra = builder(rng, n, seed + i, quick)
        evidences.append(evidence)
        extras[label] = extra
    return evidences, extras


def _scaling(builders: list[tuple[str, callable]], seed: int, quick: bool) -> pd.DataFrame:
    sizes = [120, 250] if quick else [200, 500, 900]
    model_seeds = [seed] if quick else [11, 29, 47]
    rows = []
    for label, builder in builders:
        for size in sizes:
            for model_seed in model_seeds:
                # Reinitialize from the same seed at each N so that the smaller
                # design is nested in the larger one as far as the sampler permits.
                local_rng = np.random.default_rng(7000 + model_seed)
                ev, _ = builder(local_rng, size, model_seed, True)
                pred = ev.predict(ev.x_test)
                rows.append({"problem": label, "training_samples": size, "seed": model_seed, **regression_metrics(ev.y_test, pred)})
    return pd.DataFrame(rows)


def _seed_uncertainty(builders: list[tuple[str, callable]], n: int, quick: bool) -> pd.DataFrame:
    rows = []
    seeds = [11] if quick else [11, 29, 47]
    for label, builder in builders:
        for seed in seeds:
            local_rng = np.random.default_rng(1000 + seed)
            ev, _ = builder(local_rng, n, seed, quick)
            rows.append({"problem": label, "seed": seed, **regression_metrics(ev.y_test, ev.predict(ev.x_test))})
    return pd.DataFrame(rows)


def run(output: Path, quick: bool = False, seed: int = 11) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    n = 250 if quick else 900
    evidences, extras = build_main_evidences(quick=quick, seed=seed)
    builders = [
        ("Rayleigh inverse", _build_rayleigh),
        ("Fanno inverse", _build_fanno),
        ("Oblique inverse", _build_oblique),
        ("Nozzle inverse", _build_nozzle),
        ("Shock tube implicit", _build_shock_tube),
    ]

    metrics, baselines = _evaluate(evidences)
    near = _near_singular(evidences)
    range_generalization = _range_generalization(evidences, quick)
    timing = _timing(evidences, quick)
    print("evaluating scaling", flush=True)
    scaling_builders = [item for item in builders if item[0] in {"Fanno inverse", "Oblique inverse"}]
    scaling = _scaling(scaling_builders, seed, quick)
    print("evaluating seed uncertainty", flush=True)
    uncertainty = _seed_uncertainty(builders, 180 if quick else 400, quick)
    uncertainty_summary = uncertainty.groupby("problem")[["rel_l2", "rel_linf", "mae", "rmse", "max_abs"]].agg(["mean", "std"])
    uncertainty_summary.columns = ["_".join(c) for c in uncertainty_summary.columns]
    uncertainty_summary = uncertainty_summary.reset_index()

    physical_rows = [
        {"problem": "Rayleigh", "diagnostic": "entropy_linf", "value": extras["Rayleigh inverse"]["entropy_linf"]},
        {"problem": "Rayleigh", "diagnostic": "entropy_peak_mach_error", "value": extras["Rayleigh inverse"]["entropy_peak_mach_error"]},
        {"problem": "Fanno", "diagnostic": "entropy_linf", "value": extras["Fanno inverse"]["entropy_linf"]},
        {"problem": "Fanno", "diagnostic": "nonnegative_rate", "value": extras["Fanno inverse"]["nonnegative_rate"]},
        {"problem": "Oblique", "diagnostic": "anchor_linf", "value": extras["Oblique inverse"]["anchor_linf"]},
        {"problem": "Oblique", "diagnostic": "valid_branch_rate", "value": extras["Oblique inverse"]["valid_branch_rate"]},
        {"problem": "Nozzle", "diagnostic": "back_pressure_residual_linf", "value": extras["Nozzle inverse"]["back_pressure_residual_linf"]},
        {"problem": "Nozzle", "diagnostic": "valid_location_rate", "value": extras["Nozzle inverse"]["valid_location_rate"]},
        {"problem": "Shock tube", "diagnostic": "equation_residual_linf", "value": extras["Shock tube implicit"]["equation_residual_linf"]},
        {"problem": "Shock tube", "diagnostic": "valid_pressure_rate", "value": extras["Shock tube implicit"]["valid_pressure_rate"]},
    ]
    ablations = pd.DataFrame([
        {"problem": "Fanno forward near sonic", "variant": "raw output", "mae": extras["Fanno inverse"]["raw_near_sonic_mae"], "invalid_rate": extras["Fanno inverse"]["raw_negative_rate_near_sonic"]},
        {"problem": "Fanno forward near sonic", "variant": "quadratic positive transform", "mae": extras["Fanno inverse"]["structured_near_sonic_mae"], "invalid_rate": 0.0},
        {"problem": "Oblique inverse", "variant": "naive single MLP", "mae": extras["Oblique inverse"]["naive_branch_distance_mae"], "invalid_rate": np.nan},
        {"problem": "Oblique inverse", "variant": "branch experts", "mae": extras["Oblique inverse"]["branchwise_mae"], "invalid_rate": 1.0 - extras["Oblique inverse"]["valid_branch_rate"]},
    ])

    tables = {
        "primary_metrics.csv": metrics,
        "baseline_comparison.csv": baselines,
        "near_singular_metrics.csv": near,
        "range_generalization.csv": range_generalization,
        "timing.csv": timing,
        "training_size_scaling.csv": scaling,
        "seed_uncertainty.csv": uncertainty,
        "seed_uncertainty_summary.csv": uncertainty_summary,
        "physical_diagnostics.csv": pd.DataFrame(physical_rows),
        "ablations.csv": ablations,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)

    environment = {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": seed,
        "training_samples": n,
        "mode": "quick" if quick else "full",
        "timing_protocol": "warm-up then median and IQR across repeated wall-clock measurements; CPU; batch inference",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    from .figures import make_all_figures
    make_all_figures(output, evidences, extras, tables)

    summary = {
        "primary_metrics": metrics.to_dict(orient="records"),
        "physical_diagnostics": physical_rows,
        "ablations": ablations.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/revision"))
    parser.add_argument("--quick", action="store_true", help="CI-sized run")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run(args.output, quick=args.quick, seed=args.seed)


if __name__ == "__main__":
    main()
