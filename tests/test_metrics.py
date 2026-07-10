import numpy as np
import pytest

from pca_ensemble.metrics import cluster_bootstrap_interval, ignition_log_error, relative_error


def test_ignition_log_error():
    error = ignition_log_error(np.array([1.0, 10.0]), np.array([1.0, 1.0]))
    assert error.tolist() == pytest.approx([0.0, 1.0])


def test_relative_error():
    error = relative_error(np.array([2.0, 8.0]), np.array([2.0, 10.0]))
    assert error.tolist() == pytest.approx([0.0, 0.2])


def test_cluster_bootstrap_interval_is_reproducible():
    values = np.array([1.0, 2.0, 10.0, 20.0])
    clusters = np.array(["a", "a", "b", "b"])
    first = cluster_bootstrap_interval(values, clusters, replicates=100, seed=7)
    second = cluster_bootstrap_interval(values, clusters, replicates=100, seed=7)
    assert first == second
    assert first[0] == pytest.approx(8.25)
