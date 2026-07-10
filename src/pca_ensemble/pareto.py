"""Pareto-front utilities that keep unlike emissions as separate objectives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_mask(
    frame: pd.DataFrame,
    minimize: list[str],
    maximize: list[str] | None = None,
) -> np.ndarray:
    """Return True for non-dominated rows; all objectives must be finite."""
    maximize = maximize or []
    if not minimize and not maximize:
        raise ValueError("at least one objective is required")
    columns = minimize + maximize
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives contain missing or non-finite values")
    signs = np.array([1.0] * len(minimize) + [-1.0] * len(maximize))
    costs = values * signs
    efficient = np.ones(len(frame), dtype=bool)
    for index, candidate in enumerate(costs):
        if not efficient[index]:
            continue
        dominated_by_any = np.any(
            np.all(costs <= candidate, axis=1) & np.any(costs < candidate, axis=1)
        )
        if dominated_by_any:
            efficient[index] = False
    return efficient


def mechanism_robust_summary(
    frame: pd.DataFrame,
    condition_columns: list[str],
    minimize: list[str],
    maximize: list[str],
    mechanism_column: str = "mechanism_id",
) -> pd.DataFrame:
    """Aggregate separate objectives without collapsing unlike pollutants."""
    required = set(condition_columns + minimize + maximize + [mechanism_column])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing robust-summary columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(condition_columns, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(condition_columns, keys, strict=True))
        row["mechanism_count"] = group[mechanism_column].nunique()
        for objective in minimize + maximize:
            values = group[objective].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"objective {objective} contains non-finite values")
            row[f"{objective}_median"] = float(np.median(values))
            row[f"{objective}_p05"] = float(np.quantile(values, 0.05))
            row[f"{objective}_p95"] = float(np.quantile(values, 0.95))
            row[f"{objective}_worst"] = (
                float(np.max(values)) if objective in minimize else float(np.min(values))
            )
            absolute_range = float(np.max(values) - np.min(values))
            # Scaling by the largest absolute prediction avoids unstable ratios
            # when an emission median approaches zero. For non-negative
            # objectives this normalized range is bounded by [0, 1].
            scale = max(float(np.max(np.abs(values))), 1e-30)
            row[f"{objective}_absolute_range"] = absolute_range
            row[f"{objective}_relative_range"] = absolute_range / scale
        rows.append(row)
    return pd.DataFrame(rows)


def robust_pareto_mask(
    summary: pd.DataFrame,
    minimize: list[str],
    maximize: list[str],
    dispersion_quantile: float | None = 0.75,
) -> np.ndarray:
    """Pareto-filter worst-case objectives, optionally excluding high dispersion."""
    minimize_columns = [f"{name}_worst" for name in minimize]
    maximize_columns = [f"{name}_worst" for name in maximize]
    eligible = np.ones(len(summary), dtype=bool)
    if dispersion_quantile is not None:
        if not 0.0 < dispersion_quantile <= 1.0:
            raise ValueError("dispersion_quantile must be in (0, 1]")
        for objective in minimize + maximize:
            column = f"{objective}_relative_range"
            eligible &= summary[column].to_numpy() <= summary[column].quantile(dispersion_quantile)
    result = np.zeros(len(summary), dtype=bool)
    if eligible.any():
        result[np.flatnonzero(eligible)] = pareto_mask(
            summary.loc[eligible], minimize_columns, maximize_columns
        )
    return result
