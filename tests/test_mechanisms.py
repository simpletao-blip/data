from pathlib import Path

from pca_ensemble.mechanisms import audit_mechanism


def test_mechanism_audit_uses_reactive_benchmark_composition() -> None:
    mechanism = Path("example_data/ammonia-CO-H2-Alzueta-2023.yaml")
    audit = audit_mechanism(mechanism)
    composition = dict(audit.benchmark_composition)
    assert composition["NH3"] > 0.0
    assert composition["O2"] > 0.0
    assert audit.nonfinite_forward_rate_count == 0
    assert audit.negative_equation_aggregate_count == 0
