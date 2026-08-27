"""Auditable analytical reference relations for the five benchmark families."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar


GAMMA = 1.4
EPS = 1.0e-12


def rayleigh_ratios(mach: np.ndarray | float, gamma: float = GAMMA) -> np.ndarray:
    """Return [T/T*, P/P*, rho/rho*, u/u*, T0/T0*, P0/P0*]."""
    m = np.asarray(mach, dtype=float)
    den = 1.0 + gamma * m**2
    t = (1.0 + gamma) ** 2 * m**2 / den**2
    p = (1.0 + gamma) / den
    rho = den / ((1.0 + gamma) * m**2)
    u = (1.0 + gamma) * m**2 / den
    stag_factor = (1.0 + 0.5 * (gamma - 1.0) * m**2) / (0.5 * (gamma + 1.0))
    t0 = t * stag_factor
    p0 = p * stag_factor ** (gamma / (gamma - 1.0))
    return np.stack([t, p, rho, u, t0, p0], axis=-1)


def rayleigh_inverse_t0(t0_ratio: float, branch: str, gamma: float = GAMMA) -> float:
    """Invert T0/T0* on a declared subsonic or supersonic branch."""
    if not 0.0 < t0_ratio <= 1.0:
        raise ValueError("T0/T0* must lie in (0, 1].")
    if abs(t0_ratio - 1.0) < 1.0e-13:
        return 1.0
    fn = lambda m: float(rayleigh_ratios(m, gamma)[4] - t0_ratio)
    if branch == "subsonic":
        return brentq(fn, 1.0e-5, 1.0 - 1.0e-9)
    if branch == "supersonic":
        return brentq(fn, 1.0 + 1.0e-9, 30.0)
    raise ValueError("branch must be 'subsonic' or 'supersonic'.")


def fanno_ratios(mach: np.ndarray | float, gamma: float = GAMMA) -> np.ndarray:
    """Return [T/T*, P/P*, rho/rho*, P0/P0*, 4fL*/D]."""
    m = np.asarray(mach, dtype=float)
    den = 2.0 + (gamma - 1.0) * m**2
    t = (gamma + 1.0) / den
    p = np.sqrt((gamma + 1.0) / den) / m
    rho = np.sqrt(den / (gamma + 1.0)) / m
    p0 = ((den / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))) / m
    fld = (1.0 - m**2) / (gamma * m**2)
    fld += (gamma + 1.0) / (2.0 * gamma) * np.log(
        ((gamma + 1.0) * m**2) / den
    )
    fld = np.maximum(fld, 0.0)
    return np.stack([t, p, rho, p0, fld], axis=-1)


def fanno_inverse_fld(fld: float, branch: str, gamma: float = GAMMA) -> float:
    """Invert 4fL*/D on a declared branch."""
    if fld < 0.0:
        raise ValueError("4fL*/D must be non-negative.")
    if fld < 1.0e-14:
        return 1.0
    fn = lambda m: float(fanno_ratios(m, gamma)[4] - fld)
    if branch == "subsonic":
        return brentq(fn, 1.0e-4, 1.0 - 1.0e-10)
    if branch == "supersonic":
        upper_limit = float(fanno_ratios(1.0e4, gamma)[4])
        if fld >= upper_limit:
            raise ValueError("Supersonic Fanno target exceeds the finite branch limit.")
        return brentq(fn, 1.0 + 1.0e-10, 1.0e4)
    raise ValueError("branch must be 'subsonic' or 'supersonic'.")


def mach_angle(mach: np.ndarray | float) -> np.ndarray:
    return np.arcsin(1.0 / np.asarray(mach, dtype=float))


def oblique_theta(mach: np.ndarray | float, beta: np.ndarray | float,
                  gamma: float = GAMMA) -> np.ndarray:
    """Theta-beta-M relation; angles are radians."""
    m, b = np.broadcast_arrays(np.asarray(mach, float), np.asarray(beta, float))
    numerator = 2.0 / np.tan(b) * (m**2 * np.sin(b) ** 2 - 1.0)
    denominator = m**2 * (gamma + np.cos(2.0 * b)) + 2.0
    return np.arctan(numerator / denominator)


@lru_cache(maxsize=4096)
def oblique_detachment(mach_rounded: float, gamma: float = GAMMA) -> tuple[float, float]:
    """Return (beta_at_max_theta, theta_max) for a scalar Mach number."""
    m = float(mach_rounded)
    mu = math.asin(1.0 / m)
    result = minimize_scalar(
        lambda b: -float(oblique_theta(m, b, gamma)),
        bounds=(mu + 1.0e-8, 0.5 * math.pi - 1.0e-8),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    beta = float(result.x)
    return beta, float(oblique_theta(m, beta, gamma))


def oblique_beta(mach: float, theta: float, branch: str,
                 gamma: float = GAMMA) -> float:
    """Invert the attached theta-beta-M relation on weak/strong branches."""
    mu = math.asin(1.0 / mach)
    beta_peak, theta_max = oblique_detachment(round(float(mach), 10), gamma)
    if theta < -EPS or theta > theta_max + 1.0e-10:
        raise ValueError("theta is outside the attached-shock domain.")
    if theta <= EPS:
        return mu if branch == "weak" else 0.5 * math.pi
    fn = lambda b: float(oblique_theta(mach, b, gamma) - theta)
    if branch == "weak":
        return brentq(fn, mu + 1.0e-9, beta_peak - 1.0e-9)
    if branch == "strong":
        return brentq(fn, beta_peak + 1.0e-9, 0.5 * math.pi - 1.0e-9)
    raise ValueError("branch must be 'weak' or 'strong'.")


def area_mach(mach: np.ndarray | float, gamma: float = GAMMA) -> np.ndarray:
    m = np.asarray(mach, dtype=float)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    core = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m**2)
    return core**exponent / m


def mach_from_area(area_ratio: float, branch: str, gamma: float = GAMMA) -> float:
    if area_ratio < 1.0:
        raise ValueError("A/A* must be at least one.")
    if abs(area_ratio - 1.0) < 1.0e-13:
        return 1.0
    fn = lambda m: float(area_mach(m, gamma) - area_ratio)
    if branch == "subsonic":
        return brentq(fn, 1.0e-8, 1.0 - 1.0e-10)
    if branch == "supersonic":
        return brentq(fn, 1.0 + 1.0e-10, 50.0)
    raise ValueError("branch must be 'subsonic' or 'supersonic'.")


def normal_shock_m2(m1: float, gamma: float = GAMMA) -> float:
    return math.sqrt((1.0 + 0.5 * (gamma - 1.0) * m1**2) /
                     (gamma * m1**2 - 0.5 * (gamma - 1.0)))


def normal_shock_p02_p01(m1: float, gamma: float = GAMMA) -> float:
    a = ((gamma + 1.0) * m1**2) / ((gamma - 1.0) * m1**2 + 2.0)
    b = (gamma + 1.0) / (2.0 * gamma * m1**2 - (gamma - 1.0))
    return a ** (gamma / (gamma - 1.0)) * b ** (1.0 / (gamma - 1.0))


def isentropic_p_p0(mach: float, gamma: float = GAMMA) -> float:
    return (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (-gamma / (gamma - 1.0))


def nozzle_back_pressure(exit_area_ratio: float, shock_area_ratio: float,
                         gamma: float = GAMMA) -> float:
    """Back pressure ratio for a normal shock inside a C-D nozzle."""
    if not 1.0 <= shock_area_ratio <= exit_area_ratio:
        raise ValueError("Shock area must satisfy 1 <= As/At <= Ae/At.")
    m1 = mach_from_area(shock_area_ratio, "supersonic", gamma)
    m2 = normal_shock_m2(m1, gamma)
    shock_over_downstream_star = float(area_mach(m2, gamma))
    exit_over_downstream_star = (exit_area_ratio / shock_area_ratio) * shock_over_downstream_star
    me = mach_from_area(exit_over_downstream_star, "subsonic", gamma)
    return normal_shock_p02_p01(m1, gamma) * isentropic_p_p0(me, gamma)


def nozzle_shock_area(exit_area_ratio: float, back_pressure: float,
                      gamma: float = GAMMA) -> float:
    """Invert the internal-shock relation with a bracketed root."""
    lo = nozzle_back_pressure(exit_area_ratio, 1.0 + 1.0e-8, gamma)
    hi = nozzle_back_pressure(exit_area_ratio, exit_area_ratio - 1.0e-8, gamma)
    lower, upper = sorted((lo, hi))
    if not lower - 1.0e-9 <= back_pressure <= upper + 1.0e-9:
        raise ValueError("Back pressure is outside the internal-shock domain.")
    fn = lambda a: nozzle_back_pressure(exit_area_ratio, a, gamma) - back_pressure
    return brentq(fn, 1.0 + 1.0e-8, exit_area_ratio - 1.0e-8)


def shock_tube_pressure_ratio(p4_p1: float, t4_t1: float = 1.0,
                              gamma: float = GAMMA) -> float:
    """Solve the ideal-gas shock-tube pressure ratio P2/P1 robustly."""
    if p4_p1 <= 1.0 or t4_t1 <= 0.0:
        raise ValueError("Require P4/P1 > 1 and T4/T1 > 0.")
    a1_a4 = math.sqrt(1.0 / t4_t1)

    def residual(p2_p1: float) -> float:
        term = (gamma - 1.0) * a1_a4 * (p2_p1 - 1.0)
        term /= math.sqrt(2.0 * gamma * (2.0 * gamma + (gamma + 1.0) * (p2_p1 - 1.0)))
        base = 1.0 - term
        if base <= 0.0:
            return math.inf
        predicted = p2_p1 * base ** (-2.0 * gamma / (gamma - 1.0))
        return predicted - p4_p1

    grid = np.geomspace(1.0 + 1.0e-10, p4_p1, 300)
    values = np.array([residual(float(x)) for x in grid])
    finite = np.isfinite(values)
    for left, right, fl, fr in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if np.isfinite(fl) and np.isfinite(fr) and fl * fr <= 0.0:
            return brentq(residual, float(left), float(right), xtol=1.0e-12, rtol=1.0e-12)
    raise RuntimeError(f"No physical shock-tube root found for P4/P1={p4_p1}, T4/T1={t4_t1}; finite={finite.sum()}.")


def shock_tube_residual(p2_p1: np.ndarray | float, p4_p1: np.ndarray | float,
                        t4_t1: np.ndarray | float, gamma: float = GAMMA) -> np.ndarray:
    p2, p4, t4 = np.broadcast_arrays(np.asarray(p2_p1, float), np.asarray(p4_p1, float), np.asarray(t4_t1, float))
    a1_a4 = np.sqrt(1.0 / t4)
    term = (gamma - 1.0) * a1_a4 * (p2 - 1.0)
    term /= np.sqrt(2.0 * gamma * (2.0 * gamma + (gamma + 1.0) * (p2 - 1.0)))
    base = 1.0 - term
    predicted = p2 * np.where(base > 0.0, base ** (-2.0 * gamma / (gamma - 1.0)), np.nan)
    return predicted - p4


def entropy_over_r(temperature_ratio: np.ndarray, pressure_ratio: np.ndarray,
                   gamma: float = GAMMA) -> np.ndarray:
    return gamma / (gamma - 1.0) * np.log(temperature_ratio) - np.log(pressure_ratio)

