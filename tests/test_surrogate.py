import numpy as np
import pandas as pd

from pca_ensemble.surrogate import fit_log_flame_gpr, predict_log_flame_gpr


def test_log_gpr_returns_positive_predictions():
    frame = pd.DataFrame({
        "temperature_K": [300, 400, 500, 600, 700, 800],
        "pressure_bar": [1, 1, 5, 5, 10, 10],
        "equivalence_ratio": [0.7, 0.9, 1.0, 1.1, 1.2, 1.3],
        "cracking_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "laminar_burning_velocity_m_per_s": [0.05, 0.08, 0.12, 0.16, 0.19, 0.22],
    })
    model = fit_log_flame_gpr(frame, optimize=False)
    prediction = predict_log_flame_gpr(model, frame.iloc[:2])
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction.surrogate_m_per_s > 0.0).all()

