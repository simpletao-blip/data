"""Interpolation surrogates whose accuracy is tested on held-out Cantera cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["temperature_K", "log10_pressure_bar", "equivalence_ratio", "cracking_ratio"]


def flame_features(frame: pd.DataFrame) -> np.ndarray:
    required = {"temperature_K", "pressure_bar", "equivalence_ratio", "cracking_ratio"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing flame-map features: {sorted(missing)}")
    return np.column_stack([
        frame.temperature_K.to_numpy(float),
        np.log10(frame.pressure_bar.to_numpy(float)),
        frame.equivalence_ratio.to_numpy(float),
        frame.cracking_ratio.to_numpy(float),
    ])


def fit_log_flame_gpr(
    frame: pd.DataFrame,
    seed: int = 20260703,
    optimize: bool = True,
) -> Pipeline:
    y = frame.laminar_burning_velocity_m_per_s.to_numpy(float)
    if np.any(y <= 0.0) or not np.isfinite(y).all():
        raise ValueError("flame speeds must be finite and positive")
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e2), nu=1.5)
              + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1)))
    regressor = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=seed,
        n_restarts_optimizer=2 if optimize else 0,
        optimizer="fmin_l_bfgs_b" if optimize else None,
    )
    model = Pipeline([("scale", StandardScaler()), ("gpr", regressor)])
    model.fit(flame_features(frame), np.log(y))
    return model


def predict_log_flame_gpr(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    mean_log, std_log = model.predict(flame_features(frame), return_std=True)
    return pd.DataFrame({
        "surrogate_m_per_s": np.exp(mean_log),
        "surrogate_90_low_m_per_s": np.exp(mean_log - 1.645 * std_log),
        "surrogate_90_high_m_per_s": np.exp(mean_log + 1.645 * std_log),
        "surrogate_log_standard_deviation": std_log,
    }, index=frame.index)

