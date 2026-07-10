import numpy as np

from pca_ensemble.stacking import (
    fit_simplex_stacking,
    leave_one_campaign_out,
    nested_grouped_stacking_intervals,
)


def test_simplex_constraints():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.column_stack([y, y + 2.0, y - 1.0])
    fit = fit_simplex_stacking(y, predictions)
    assert fit.success
    assert np.all(fit.weights >= 0.0)
    assert np.isclose(fit.weights.sum(), 1.0)
    assert fit.objective < 1e-8


def test_grouped_holdout_keeps_campaigns_intact():
    y = np.arange(1.0, 7.0)
    predictions = np.column_stack([y, y + 1.0, y - 1.0])
    campaigns = np.array(["a", "a", "b", "b", "c", "c"])
    result = leave_one_campaign_out(y, predictions, campaigns, ["exact", "high", "low"])
    assert len(result) == len(y)
    assert set(result["held_out_campaign"]) == {"a", "b", "c"}
    assert np.allclose(result["stacked"], y, atol=1e-5)


def test_nested_intervals_use_inner_campaign_residuals():
    y = np.arange(1.0, 9.0)
    predictions = np.column_stack([y, y * 1.1])
    campaigns = np.repeat(["a", "b", "c", "d"], 2)
    result = nested_grouped_stacking_intervals(
        y, predictions, campaigns, ["exact", "high"], sample_scale=y
    )
    assert result.residual_interval_low.notna().all()
    assert result.residual_interval_high.notna().all()
    assert (result.residual_interval_low <= result.stacked).all()
    assert (result.residual_interval_high >= result.stacked).all()
    assert 0.0 < result.residual_interval_covered.mean() <= 1.0


def test_campaign_equal_weighting_is_supported():
    y = np.array([1.0, 1.0, 1.0, 1.0, 3.0, 5.0])
    predictions = np.column_stack([np.ones(6), y])
    campaigns = np.array(["large"] * 4 + ["b", "c"])
    result = leave_one_campaign_out(
        y, predictions, campaigns, ["flat", "exact"], sample_scale=y,
        campaign_equal_weighted=True,
    )
    assert len(result) == len(y)
    assert result.stacking_success.all()
