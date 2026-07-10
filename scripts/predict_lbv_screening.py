"""Fit a held-out-tested flame surrogate and predict the common screening grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pca_ensemble.surrogate import fit_log_flame_gpr, predict_log_flame_gpr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-results", type=Path, required=True)
    parser.add_argument("--screening-design", type=Path,
                        default=Path("data/processed/proxy_screening_design.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thermo-limits", type=Path,
                        default=Path("mechanisms/thermo_limits.csv"))
    args = parser.parse_args()

    results = pd.read_csv(args.map_results)
    mechanism_id = results.mechanism_id.dropna().iloc[0]
    limits = pd.read_csv(args.thermo_limits).set_index("mechanism_id")
    if mechanism_id not in limits.index:
        raise ValueError(f"missing thermodynamic limits for {mechanism_id}")
    lower = float(limits.loc[mechanism_id, "min_temperature_K"])
    upper = float(limits.loc[mechanism_id, "max_temperature_K"])
    training = results[(results.design_role == "training") &
                       (results.status == "completed")].copy()
    training = training[
        training.temperature_K.between(lower, upper)
        & training.max_temperature_K.between(lower, upper)
    ].copy()
    if len(training) < 20:
        raise ValueError("at least 20 completed training flames are required")
    model = fit_log_flame_gpr(training)
    screening = pd.read_csv(args.screening_design)
    flame_query = screening.copy()
    flame_query["temperature_K"] = flame_query.flame_temperature_K
    predictions = predict_log_flame_gpr(model, flame_query)
    output = pd.concat([screening.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    output["mechanism_id"] = mechanism_id
    output["prediction_basis"] = "GPR trained on direct Cantera cases"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.groupby("inside_lbv_convex_hull").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
