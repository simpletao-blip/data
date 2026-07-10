"""Observable-specific grouped stacking for JSR species validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.metrics import cluster_bootstrap_interval
from pca_ensemble.stacking import leave_one_campaign_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    data = data[data.status.eq("completed")].copy()
    pivot = data.pivot(index="dataset_id", columns="mechanism_id", values="simulated_value")
    metadata = (data.sort_values("mechanism_id").drop_duplicates("dataset_id")
                .set_index("dataset_id"))
    mechanism_names = sorted(data.mechanism_id.unique())
    rows = []
    summaries = []
    folds = []
    gate_rows = []
    for species, species_meta in metadata.groupby("observable"):
        ids = species_meta.index.intersection(pivot.dropna(subset=mechanism_names).index)
        if len(ids) < 4:
            continue
        local = metadata.loc[ids]
        campaigns = local.campaign_id.to_numpy()
        if np.unique(campaigns).size < 2:
            continue
        observed = local.value.to_numpy(float)
        uncertainty = np.maximum(local.uncertainty.to_numpy(float), 1e-8)
        predictions = pivot.loc[ids, mechanism_names].to_numpy(float)
        cv = leave_one_campaign_out(
            observed, predictions, campaigns, mechanism_names=mechanism_names,
            sample_scale=uncertainty,
        )
        cv["dataset_id"] = ids[cv.sample_index].to_numpy()
        cv["observable"] = species
        cv["uncertainty"] = uncertainty[cv.sample_index]
        cv["standardized_observed"] = cv.observed / cv.uncertainty
        for method in ("stacked", "equal_weight", "best_single"):
            cv[f"absolute_standardized_error_{method}"] = (
                (cv[method] - cv.observed).abs() / cv.uncertainty
            )
            estimate, low, high = cluster_bootstrap_interval(
                cv[f"absolute_standardized_error_{method}"].to_numpy(),
                cv.held_out_campaign.to_numpy(), replicates=2000,
            )
            summaries.append({
                "observable": species, "method": method, "n": len(cv),
                "campaigns": cv.held_out_campaign.nunique(),
                "mean_absolute_standardized_error": estimate,
                "cluster_bootstrap_95_low": low, "cluster_bootstrap_95_high": high,
                "median_absolute_standardized_error": float(
                    cv[f"absolute_standardized_error_{method}"].median()
                ),
            })
        gate_row = {"observable": species}
        for baseline in ("equal_weight", "best_single"):
            difference = (
                cv["absolute_standardized_error_stacked"]
                - cv[f"absolute_standardized_error_{baseline}"]
            ).to_numpy()
            estimate, low, high = cluster_bootstrap_interval(
                difference, cv.held_out_campaign.to_numpy(), replicates=2000,
            )
            gate_row[f"stacking_minus_{baseline}_mean"] = estimate
            gate_row[f"stacking_minus_{baseline}_95_low"] = low
            gate_row[f"stacking_minus_{baseline}_95_high"] = high
        gate_rows.append(gate_row)
        weight_columns = [name for name in cv if name.startswith("weight_")]
        local_folds = cv[["held_out_campaign", "best_single_name", "stacking_success",
                          *weight_columns]].drop_duplicates().copy()
        local_folds.insert(0, "observable", species)
        folds.append(local_folds)
        rows.append(cv)
    predictions_out = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    wide = summary.pivot(index="observable", columns="method",
                         values="mean_absolute_standardized_error")
    gate_statistics = pd.DataFrame(gate_rows).set_index("observable")
    wide = wide.join(gate_statistics)
    comparator = wide[["equal_weight", "best_single"]].min(axis=1)
    numerical_tolerance = 1e-6 * np.maximum(1.0, comparator.abs())
    wide["stacking_beats_both_point_estimate"] = wide.stacked < (
        comparator - numerical_tolerance
    )
    wide["stacking_beats_both"] = (
        wide.stacking_beats_both_point_estimate
        & (wide.stacking_minus_equal_weight_95_high < 0.0)
        & (wide.stacking_minus_best_single_95_high < 0.0)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_out.to_csv(args.output_dir / "jsr_loco_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "jsr_method_summary.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(args.output_dir / "jsr_fold_weights.csv", index=False)
    wide.reset_index().to_csv(args.output_dir / "jsr_observable_gate.csv", index=False)
    print(wide.to_string())
    print(f"stacking_beats_both={int(wide.stacking_beats_both.sum())}/{len(wide)} observables")


if __name__ == "__main__":
    main()
