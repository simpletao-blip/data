"""Deterministic, campaign-preserving selection of expensive validation cases."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def farthest_point_indices(features: np.ndarray, count: int) -> list[int]:
    """Select a centroid-near point followed by maximin points."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or not len(features):
        return []
    count = min(int(count), len(features))
    if count <= 0:
        return []
    center = np.nanmean(features, axis=0)
    filled = np.where(np.isfinite(features), features, center)
    first = int(np.argmin(np.linalg.norm(filled - center, axis=1)))
    selected = [first]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(filled[:, None, :] - filled[selected][None, :, :], axis=2), axis=1
        )
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    return selected


def select_lbv_design(data: pd.DataFrame, per_campaign: int = 5) -> pd.DataFrame:
    lbv = data[data.observable.eq("laminar burning velocity")].copy()
    if "initial_composition" in lbv:
        allowed = {"NH3", "H2", "O2", "N2", "AR", "HE"}
        compatible = lbv.initial_composition.map(
            lambda raw: set(json.loads(raw)).issubset(allowed)
            or {name.upper() for name in json.loads(raw)}.issubset(allowed)
        )
        lbv = lbv[compatible].copy()
    if lbv.empty:
        return lbv
    feature_names = ["temperature_K", "pressure_Pa", "equivalence_ratio", "cracking_ratio"]
    values = lbv[feature_names].astype(float).copy()
    values["pressure_Pa"] = np.log10(values["pressure_Pa"])
    span = values.max() - values.min()
    normalized = (values - values.min()) / span.replace(0.0, 1.0)
    lbv[[f"design_{name}" for name in feature_names]] = normalized.to_numpy()
    selected: list[pd.DataFrame] = []
    design_columns = [f"design_{name}" for name in feature_names]
    for _, campaign in lbv.groupby("campaign_id", sort=True):
        indices = farthest_point_indices(campaign[design_columns].to_numpy(), per_campaign)
        selected.append(campaign.iloc[indices])
    result = pd.concat(selected).sort_values(["campaign_id", "dataset_id"]).reset_index(drop=True)
    result.insert(0, "design_id", [f"lbv_{index:04d}" for index in range(1, len(result) + 1)])
    return result.drop(columns=design_columns)


def convex_hull_membership(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return membership in a multi-dimensional experimental convex hull."""
    reference = np.asarray(reference, dtype=float)
    query = np.asarray(query, dtype=float)
    if reference.ndim != 2 or query.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("reference and query must be 2D with matching feature counts")
    if len(reference) <= reference.shape[1]:
        return np.zeros(len(query), dtype=bool)
    low = np.nanmin(reference, axis=0)
    span = np.nanmax(reference, axis=0) - low
    span[span == 0.0] = 1.0
    normalized_reference = (reference - low) / span
    normalized_query = (query - low) / span
    try:
        hull = Delaunay(normalized_reference, qhull_options="QJ")
        return hull.find_simplex(normalized_query) >= 0
    except Exception:
        return np.zeros(len(query), dtype=bool)


def select_operating_design(
    candidates: pd.DataFrame,
    validation_reference: pd.DataFrame,
    train_count: int = 120,
    holdout_count: int = 24,
    train_inside_fraction: float = 0.6,
    holdout_inside_fraction: float = 0.5,
) -> pd.DataFrame:
    features = ["temperature_K", "pressure_bar", "equivalence_ratio", "cracking_ratio"]
    required = set(features)
    if required.difference(candidates) or required.difference(validation_reference):
        raise ValueError(f"both tables require columns: {features}")
    work = candidates.copy().reset_index(drop=True)
    matrix = work[features].to_numpy(float)
    matrix[:, 1] = np.log10(matrix[:, 1])
    low = matrix.min(axis=0)
    span = matrix.max(axis=0) - low
    span[span == 0.0] = 1.0
    normalized = (matrix - low) / span
    reference = validation_reference[features].dropna().to_numpy(float)
    reference[:, 1] = np.log10(reference[:, 1])
    query = matrix.copy()
    membership = convex_hull_membership(reference, query)
    inside = np.flatnonzero(membership)
    outside = np.flatnonzero(~membership)

    def choose(pool: np.ndarray, count: int) -> np.ndarray:
        local = farthest_point_indices(normalized[pool], min(count, len(pool)))
        return pool[local]

    train_inside = min(round(train_count * train_inside_fraction), len(inside))
    train_outside = min(train_count - train_inside, len(outside))
    training = np.concatenate([choose(inside, train_inside), choose(outside, train_outside)])
    if len(training) < train_count:
        unused = np.setdiff1d(np.arange(len(work)), training)
        training = np.concatenate([training, choose(unused, train_count - len(training))])
    remaining = np.setdiff1d(np.arange(len(work)), training)
    remaining_inside = remaining[membership[remaining]]
    remaining_outside = remaining[~membership[remaining]]
    holdout_inside = min(round(holdout_count * holdout_inside_fraction), len(remaining_inside))
    holdout_outside = min(holdout_count - holdout_inside, len(remaining_outside))
    holdout = np.concatenate([
        choose(remaining_inside, holdout_inside), choose(remaining_outside, holdout_outside)
    ])
    if len(holdout) < holdout_count:
        unused = np.setdiff1d(remaining, holdout)
        holdout = np.concatenate([holdout, choose(unused, holdout_count - len(holdout))])
    chosen = np.concatenate([training, holdout])
    result = work.iloc[chosen].copy().reset_index(drop=True)
    result.insert(0, "design_id", [f"map_{index:04d}" for index in range(1, len(result) + 1)])
    result.insert(1, "design_role", ["training"] * len(training) + ["interpolation_holdout"] * len(holdout))
    result["inside_experimental_convex_hull"] = membership[chosen]
    return result
