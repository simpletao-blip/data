"""Fit training-only GPRs and evaluate independent Cantera holdout cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.surrogate import fit_log_flame_gpr, predict_log_flame_gpr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--thermo-limits", type=Path,
                        default=Path("mechanisms/thermo_limits.csv"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    data = data[data.status.eq("completed")].copy()
    limits = pd.read_csv(args.thermo_limits).set_index("mechanism_id")
    missing = sorted(set(data.mechanism_id).difference(limits.index))
    if missing:
        raise ValueError(f"missing thermodynamic limits for: {missing}")
    data["thermo_min_temperature_K"] = data.mechanism_id.map(limits.min_temperature_K)
    data["thermo_max_temperature_K"] = data.mechanism_id.map(limits.max_temperature_K)
    data["within_mechanism_thermo_range"] = (
        data.temperature_K.between(
            data.thermo_min_temperature_K, data.thermo_max_temperature_K
        )
        & data.max_temperature_K.between(
            data.thermo_min_temperature_K, data.thermo_max_temperature_K
        )
    )
    thermo_excluded = data[~data.within_mechanism_thermo_range].copy()
    data = data[data.within_mechanism_thermo_range].copy()
    predictions = []
    summaries = []
    admissibility = []
    for mechanism, frame in data.groupby("mechanism_id"):
        training = frame[frame.design_role.eq("training")]
        holdout = frame[frame.design_role.eq("interpolation_holdout")].copy()
        model = fit_log_flame_gpr(training)
        predicted = predict_log_flame_gpr(model, holdout)
        holdout = holdout.join(predicted)
        holdout["surrogate_relative_error"] = (
            (holdout.surrogate_m_per_s - holdout.laminar_burning_velocity_m_per_s).abs()
            / holdout.laminar_burning_velocity_m_per_s
        )
        holdout["surrogate_log_error"] = np.abs(np.log(
            holdout.surrogate_m_per_s / holdout.laminar_burning_velocity_m_per_s
        ))
        holdout["surrogate_interval_covered"] = holdout.laminar_burning_velocity_m_per_s.between(
            holdout.surrogate_90_low_m_per_s, holdout.surrogate_90_high_m_per_s
        )
        predictions.append(holdout)
        for inside, subset in holdout.groupby("inside_experimental_convex_hull"):
            summaries.append({
                "mechanism_id": mechanism, "inside_experimental_convex_hull": inside,
                "n": len(subset),
                "mean_absolute_relative_error": subset.surrogate_relative_error.mean(),
                "median_absolute_relative_error": subset.surrogate_relative_error.median(),
                "root_mean_squared_log_error": np.sqrt(np.mean(subset.surrogate_log_error**2)),
                "nominal_90_interval_coverage": subset.surrogate_interval_covered.mean(),
            })
        inside = holdout[holdout.inside_experimental_convex_hull].copy()
        mean_error = inside.surrogate_relative_error.mean()
        max_error = inside.surrogate_relative_error.max()
        coverage = inside.surrogate_interval_covered.mean()
        admissibility.append({
            "mechanism_id": mechanism,
            "inside_hull_holdout_n": len(inside),
            "inside_hull_mean_absolute_relative_error": mean_error,
            "inside_hull_max_absolute_relative_error": max_error,
            "inside_hull_nominal_90_interval_coverage": coverage,
            "mean_error_limit": 0.05,
            "max_error_limit": 0.10,
            "coverage_floor": 0.75,
            "surrogate_admissible_inside_hull": bool(
                len(inside) >= 10
                and mean_error <= 0.05
                and max_error <= 0.10
                and coverage >= 0.75
            ),
            "scope_note": (
                "Numerical interpolation gate only; this is not experimental validation "
                "and does not authorize extrapolation outside the LBV convex hull."
            ),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "lbv_surrogate_holdout_predictions.csv", index=False
    )
    thermo_excluded.to_csv(
        args.output_dir / "lbv_surrogate_thermo_excluded.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(args.output_dir / "lbv_surrogate_summary.csv", index=False)
    pd.DataFrame(admissibility).to_csv(
        args.output_dir / "lbv_surrogate_admissibility.csv", index=False
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(admissibility)[
        ["mechanism_id", "surrogate_admissible_inside_hull"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
