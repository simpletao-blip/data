import pandas as pd

from pca_ensemble.pareto import mechanism_robust_summary, pareto_mask, robust_pareto_mask


def test_pareto_keeps_tradeoffs_and_rejects_dominated_point():
    frame = pd.DataFrame({"emissions": [1.0, 2.0, 3.0], "speed": [1.0, 3.0, 1.0]})
    mask = pareto_mask(frame, minimize=["emissions"], maximize=["speed"])
    assert mask.tolist() == [True, True, False]


def test_robust_summary_uses_worst_case_by_direction():
    frame = pd.DataFrame({
        "condition": ["a", "a", "b", "b"], "mechanism_id": ["m1", "m2", "m1", "m2"],
        "NO": [1.0, 2.0, 3.0, 4.0], "speed": [2.0, 1.5, 3.0, 2.5],
    })
    summary = mechanism_robust_summary(frame, ["condition"], ["NO"], ["speed"])
    first = summary.set_index("condition").loc["a"]
    assert first.NO_worst == 2.0
    assert first.speed_worst == 1.5
    assert first.NO_absolute_range == 1.0
    assert first.NO_relative_range == 0.5
    assert first.speed_relative_range == 0.25
    mask = robust_pareto_mask(summary, ["NO"], ["speed"], dispersion_quantile=None)
    assert mask.tolist() == [True, True]
