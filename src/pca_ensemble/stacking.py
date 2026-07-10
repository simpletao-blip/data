"""Leakage-resistant grouped stacking for deterministic mechanism predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class StackingFit:
    weights: np.ndarray
    objective: float
    success: bool
    message: str


def _validate_arrays(y: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or y.ndim != 1 or predictions.shape[0] != y.size:
        raise ValueError("predictions must have shape (n_samples, n_mechanisms)")
    if predictions.shape[1] < 2:
        raise ValueError("stacking requires at least two mechanisms")
    if not np.isfinite(y).all() or not np.isfinite(predictions).all():
        raise ValueError("stacking input contains missing or non-finite values")
    return y, predictions


def fit_simplex_stacking(
    y: np.ndarray,
    predictions: np.ndarray,
    sample_scale: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> StackingFit:
    """Fit non-negative weights summing to one by scaled squared error."""
    y, predictions = _validate_arrays(y, predictions)
    scale = np.ones_like(y) if sample_scale is None else np.asarray(sample_scale, dtype=float)
    if scale.shape != y.shape or np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("sample_scale must be finite, positive, and match y")
    weight = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weight.shape != y.shape or np.any(weight <= 0.0) or not np.isfinite(weight).all():
        raise ValueError("sample_weight must be finite, positive, and match y")
    weight = weight / weight.sum()
    n_mechanisms = predictions.shape[1]

    def objective(weights: np.ndarray) -> float:
        residual = (predictions @ weights - y) / scale
        return float(np.sum(weight * residual**2))

    result = minimize(
        objective,
        x0=np.full(n_mechanisms, 1.0 / n_mechanisms),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_mechanisms,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    return StackingFit(weights, objective(weights), bool(result.success), str(result.message))


def leave_one_campaign_out(
    y: np.ndarray,
    predictions: np.ndarray,
    campaigns: np.ndarray,
    mechanism_names: list[str] | None = None,
    sample_scale: np.ndarray | None = None,
    campaign_equal_weighted: bool = False,
) -> pd.DataFrame:
    """Evaluate stacking, equal weights, and the training-best mechanism by campaign."""
    y, predictions = _validate_arrays(y, predictions)
    campaigns = np.asarray(campaigns)
    if campaigns.shape != y.shape:
        raise ValueError("campaigns must match y")
    if np.unique(campaigns).size < 2:
        raise ValueError("at least two campaigns are required")
    scale = np.ones_like(y) if sample_scale is None else np.asarray(sample_scale, dtype=float)
    names = mechanism_names or [f"mechanism_{i}" for i in range(predictions.shape[1])]
    rows: list[dict[str, object]] = []
    for held_out in np.unique(campaigns):
        test = campaigns == held_out
        train = ~test
        train_weight = None
        if campaign_equal_weighted:
            _, inverse, counts = np.unique(campaigns[train], return_inverse=True, return_counts=True)
            train_weight = 1.0 / counts[inverse]
        fit = fit_simplex_stacking(
            y[train], predictions[train], scale[train], sample_weight=train_weight
        )
        squared = ((predictions[train] - y[train, None]) / scale[train, None]) ** 2
        training_mse = (
            np.mean(squared, axis=0)
            if train_weight is None
            else np.average(squared, axis=0, weights=train_weight)
        )
        best_index = int(np.argmin(training_mse))
        stacked = predictions[test] @ fit.weights
        equal = predictions[test].mean(axis=1)
        best = predictions[test, best_index]
        for local_index, global_index in enumerate(np.flatnonzero(test)):
            row: dict[str, object] = {
                "sample_index": int(global_index),
                "held_out_campaign": held_out,
                "observed": float(y[global_index]),
                "stacked": float(stacked[local_index]),
                "equal_weight": float(equal[local_index]),
                "best_single": float(best[local_index]),
                "best_single_name": names[best_index],
                "stacking_success": fit.success,
            }
            row.update({f"weight_{name}": float(weight) for name, weight in zip(names, fit.weights, strict=True)})
            rows.append(row)
    return pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)


def nested_grouped_stacking_intervals(
    y: np.ndarray,
    predictions: np.ndarray,
    campaigns: np.ndarray,
    mechanism_names: list[str] | None = None,
    sample_scale: np.ndarray | None = None,
    interval_level: float = 0.90,
    campaign_equal_weighted: bool = False,
) -> pd.DataFrame:
    """Add leakage-resistant empirical intervals using inner held-campaign residuals."""
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must be in (0, 1)")
    y, predictions = _validate_arrays(y, predictions)
    campaigns = np.asarray(campaigns)
    scale = np.ones_like(y) if sample_scale is None else np.asarray(sample_scale, dtype=float)
    outer = leave_one_campaign_out(
        y, predictions, campaigns, mechanism_names, scale,
        campaign_equal_weighted=campaign_equal_weighted,
    )
    outer["residual_interval_level"] = interval_level
    outer["residual_log_half_width"] = np.nan
    outer["residual_interval_low"] = np.nan
    outer["residual_interval_high"] = np.nan
    outer["residual_interval_covered"] = False
    for held_out in np.unique(campaigns):
        train = campaigns != held_out
        if np.unique(campaigns[train]).size < 2:
            continue
        inner = leave_one_campaign_out(
            y[train], predictions[train], campaigns[train], mechanism_names, scale[train],
            campaign_equal_weighted=campaign_equal_weighted,
        )
        scores = np.abs(np.log(inner.stacked.to_numpy() / inner.observed.to_numpy()))
        probability = min(1.0, np.ceil((len(scores) + 1) * interval_level) / len(scores))
        half_width = float(np.quantile(scores, probability, method="higher"))
        rows = outer.held_out_campaign.eq(held_out)
        center = outer.loc[rows, "stacked"]
        low = center * np.exp(-half_width)
        high = center * np.exp(half_width)
        outer.loc[rows, "residual_log_half_width"] = half_width
        outer.loc[rows, "residual_interval_low"] = low
        outer.loc[rows, "residual_interval_high"] = high
        observed = outer.loc[rows, "observed"]
        outer.loc[rows, "residual_interval_covered"] = observed.between(low, high)
    return outer
