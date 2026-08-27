"""Reproducible scaled MLPs, transforms, metrics, and baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class ScaledMLP:
    hidden: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    seed: int = 11
    max_iter: int = 1500

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ScaledMLP":
        x = np.atleast_2d(np.asarray(x, float))
        y = np.asarray(y, float)
        self._one_output = y.ndim == 1
        y2 = y.reshape(-1, 1) if self._one_output else y
        self.x_scaler = StandardScaler().fit(x)
        self.y_scaler = StandardScaler().fit(y2)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation=self.activation,
            solver="lbfgs",
            alpha=1.0e-8,
            max_iter=self.max_iter,
            max_fun=50000,
            tol=1.0e-10,
            random_state=self.seed,
        )
        self.model.fit(self.x_scaler.transform(x), self.y_scaler.transform(y2).ravel() if self._one_output else self.y_scaler.transform(y2))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        pred = self.model.predict(self.x_scaler.transform(x))
        pred2 = np.asarray(pred).reshape(-1, 1) if self._one_output else np.asarray(pred)
        restored = self.y_scaler.inverse_transform(pred2)
        return restored.ravel() if self._one_output else restored

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return the analytical output Jacobian with respect to raw inputs.

        The returned array has shape ``(samples, outputs, inputs)``.  The
        derivative includes both input and output standardization.  The audit
        uses Tanh networks with the identity regression output supported here.
        """
        if self.activation != "tanh" or self.model.out_activation_ != "identity":
            raise NotImplementedError("Analytical Jacobian is implemented for Tanh regression MLPs.")
        x = np.atleast_2d(np.asarray(x, float))
        activation = self.x_scaler.transform(x)
        n_samples, n_inputs = activation.shape
        jac = np.broadcast_to(
            np.diag(1.0 / self.x_scaler.scale_),
            (n_samples, n_inputs, n_inputs),
        ).copy()

        for weights, bias in zip(self.model.coefs_[:-1], self.model.intercepts_[:-1]):
            preactivation = activation @ weights + bias
            activation = np.tanh(preactivation)
            jac = np.einsum("ij,njk->nik", weights.T, jac)
            jac *= (1.0 - activation**2)[:, :, None]

        jac = np.einsum("ij,njk->nik", self.model.coefs_[-1].T, jac)
        jac *= self.y_scaler.scale_[None, :, None]
        return jac


def safe_logit(fraction: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    return logit(np.clip(np.asarray(fraction, float), eps, 1.0 - eps))


def bounded_from_logit(latent: np.ndarray, lower: np.ndarray | float,
                       upper: np.ndarray | float) -> np.ndarray:
    return np.asarray(lower) + expit(np.asarray(latent)) * (np.asarray(upper) - np.asarray(lower))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    err = yp - yt
    denom_l2 = max(float(np.linalg.norm(yt.ravel())), 1.0e-15)
    denom_inf = max(float(np.max(np.abs(yt))), 1.0e-15)
    return {
        "rel_l2": float(np.linalg.norm(err.ravel()) / denom_l2),
        "rel_linf": float(np.max(np.abs(err)) / denom_inf),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
    }
