"""Compare LBV stacking, equal weights, and training-best single mechanisms."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.metrics import cluster_bootstrap_interval
from pca_ensemble.stacking import nested_grouped_stacking_intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument(
        "--exclude-doi", action="append", default=[],
        help="Exclude complete experimental studies, for example mechanism-development campaigns.",
    )
    parser.add_argument(
        "--campaign-equal-weighted", action="store_true",
        help="Give each training campaign equal total loss weight rather than each point.",
    )
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    excluded = data[data.doi.isin(args.exclude_doi)].copy()
    data = data[~data.doi.isin(args.exclude_doi)].copy()
    data = data[data.status.eq("completed")].copy()
    metadata = (data.drop_duplicates("dataset_id")
                .set_index("dataset_id")[["campaign_id", "doi", "experimental_m_per_s",
                                          "temperature_K", "pressure_Pa", "equivalence_ratio",
                                          "cracking_ratio", "apparatus"]])
    pivot = data.pivot(index="dataset_id", columns="mechanism_id", values="simulated_m_per_s").dropna()
    metadata = metadata.loc[pivot.index]
    names = list(pivot.columns)
    observed = metadata.experimental_m_per_s.to_numpy(float)
    predictions = pivot.to_numpy(float)
    campaigns = metadata.campaign_id.to_numpy()
    cv = nested_grouped_stacking_intervals(
        observed, predictions, campaigns, mechanism_names=names, sample_scale=observed,
        interval_level=0.90, campaign_equal_weighted=args.campaign_equal_weighted,
    )
    cv.insert(1, "dataset_id", pivot.index[cv.sample_index].to_numpy())
    cv = cv.join(metadata.reset_index(drop=True), on="sample_index", rsuffix="_metadata")

    summaries = []
    for method in ["stacked", "equal_weight", "best_single"]:
        residual = (cv[method].to_numpy() - cv.observed.to_numpy()) / cv.observed.to_numpy()
        absolute = np.abs(residual)
        estimate, low, high = cluster_bootstrap_interval(
            absolute, cv.held_out_campaign.to_numpy(), replicates=2000
        )
        summaries.append({
            "method": method, "n": len(cv), "campaigns": cv.held_out_campaign.nunique(),
            "mean_absolute_relative_error": estimate,
            "cluster_bootstrap_95_low": low, "cluster_bootstrap_95_high": high,
            "root_mean_squared_relative_error": float(np.sqrt(np.mean(residual**2))),
            "mean_signed_relative_bias": float(np.mean(residual)),
            "median_absolute_relative_error": float(np.median(absolute)),
        })
        cv[f"relative_error_{method}"] = absolute
        cv[f"signed_relative_residual_{method}"] = residual

    gate = {
        "observable": "LBV",
        "training_loss_weighting": (
            "campaign_equal" if args.campaign_equal_weighted else "point_equal"
        ),
    }
    for baseline in ("equal_weight", "best_single"):
        difference = (
            cv["relative_error_stacked"] - cv[f"relative_error_{baseline}"]
        ).to_numpy()
        estimate, low, high = cluster_bootstrap_interval(
            difference, cv.held_out_campaign.to_numpy(), replicates=2000
        )
        gate[f"stacking_minus_{baseline}_mean"] = estimate
        gate[f"stacking_minus_{baseline}_95_low"] = low
        gate[f"stacking_minus_{baseline}_95_high"] = high
    summary_frame = pd.DataFrame(summaries).set_index("method")
    gate["stacking_beats_both_point_estimate"] = (
        summary_frame.loc["stacked", "mean_absolute_relative_error"]
        < min(summary_frame.loc["equal_weight", "mean_absolute_relative_error"],
              summary_frame.loc["best_single", "mean_absolute_relative_error"])
    )
    gate["stacking_beats_both"] = (
        gate["stacking_beats_both_point_estimate"]
        and gate["stacking_minus_equal_weight_95_high"] < 0.0
        and gate["stacking_minus_best_single_95_high"] < 0.0
    )

    weight_columns = [column for column in cv if column.startswith("weight_")]
    folds = cv[["held_out_campaign", "best_single_name", "stacking_success", *weight_columns]].drop_duplicates()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(args.output_dir / "lbv_loco_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "lbv_method_summary.csv", index=False)
    pd.DataFrame([gate]).to_csv(args.output_dir / "lbv_gate.csv", index=False)
    folds.to_csv(args.output_dir / "lbv_fold_weights.csv", index=False)
    if args.exclude_doi:
        (excluded[["dataset_id", "campaign_id", "doi"]].drop_duplicates()
         .to_csv(args.output_dir / "lbv_excluded_development_campaigns.csv", index=False))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"stacked_interval_coverage={cv.residual_interval_covered.mean():.6g}")
    print(f"common_cases={len(cv)} campaigns={cv.held_out_campaign.nunique()} mechanisms={len(names)}")
    print(f"excluded_development_cases={excluded.dataset_id.nunique()}")
    print(pd.DataFrame([gate]).to_string(index=False))


if __name__ == "__main__":
    main()
