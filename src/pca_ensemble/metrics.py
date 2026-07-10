"""Observable-specific errors and bootstrap summaries."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def ignition_log_error(simulated: np.ndarray, experimental: np.ndarray) -> np.ndarray:
    simulated = np.asarray(simulated, dtype=float)
    experimental = np.asarray(experimental, dtype=float)
    if np.any(simulated <= 0.0) or np.any(experimental <= 0.0):
        raise ValueError("ignition delays must be positive")
    return np.abs(np.log10(simulated / experimental))


def relative_error(simulated: np.ndarray, experimental: np.ndarray) -> np.ndarray:
    simulated = np.asarray(simulated, dtype=float)
    experimental = np.asarray(experimental, dtype=float)
    if np.any(experimental == 0.0):
        raise ValueError("relative error is undefined for zero experimental values")
    return np.abs(simulated - experimental) / np.abs(experimental)


def standardized_residual(
    simulated: np.ndarray,
    experimental: np.ndarray,
    uncertainty: np.ndarray,
    error_floor: float,
) -> np.ndarray:
    denominator = np.maximum(np.asarray(uncertainty, dtype=float), float(error_floor))
    if np.any(denominator <= 0.0):
        raise ValueError("uncertainty and error floor must define a positive scale")
    return (np.asarray(simulated, dtype=float) - np.asarray(experimental, dtype=float)) / denominator


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    level: float = 0.95,
    replicates: int = 2000,
    seed: int = 20260703,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite values to bootstrap")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        estimates[index] = statistic(rng.choice(values, size=values.size, replace=True))
    tail = (1.0 - level) / 2.0
    return (
        float(statistic(values)),
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    )


def cluster_bootstrap_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    level: float = 0.95,
    replicates: int = 2000,
    seed: int = 20260703,
) -> tuple[float, float, float]:
    """Bootstrap complete campaigns while retaining their within-study rows."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    valid = np.isfinite(values)
    values = values[valid]
    clusters = clusters[valid]
    unique = np.unique(clusters)
    if not len(unique):
        raise ValueError("no finite clustered values to bootstrap")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for index in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        replicate = np.concatenate([values[clusters == cluster] for cluster in sampled])
        estimates[index] = statistic(replicate)
    tail = (1.0 - level) / 2.0
    return (
        float(statistic(values)),
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    )
