"""Campaign-equal LBV sensitivity to within-study design size and paired errors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.design import farthest_point_indices
from pca_ensemble.stacking import leave_one_campaign_out


def selected_ids(metadata: pd.DataFrame, per_campaign: int) -> list[str]:
    chosen: list[str] = []
    for _, group in metadata.groupby("campaign_id", sort=True):
        x = group[["temperature_K", "pressure_Pa", "equivalence_ratio", "cracking_ratio"]].to_numpy(float).copy()
        x[:, 1] = np.log10(x[:, 1])
        low = np.nanmin(x, axis=0)
        span = np.nanmax(x, axis=0) - low
        span[span == 0] = 1.0
        indices = farthest_point_indices((x - low) / span, min(per_campaign, len(group)))
        chosen.extend(group.iloc[indices].dataset_id.astype(str))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--exclude-doi", action="append", default=[])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/processed/lbv_revision_statistics"))
    args = parser.parse_args()
    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    raw = raw[raw.status.eq("completed") & ~raw.doi.isin(args.exclude_doi)].copy()
    if "laboratory" not in raw:
        raw["laboratory"] = pd.NA
    metadata = raw.sort_values("mechanism_id").drop_duplicates("dataset_id")[
        ["dataset_id", "campaign_id", "doi", "temperature_K", "pressure_Pa",
         "equivalence_ratio", "cracking_ratio", "experimental_m_per_s", "laboratory"]
    ]
    pivot = raw.pivot(index="dataset_id", columns="mechanism_id", values="simulated_m_per_s").dropna()
    metadata = metadata.set_index("dataset_id").loc[pivot.index].reset_index()
    rows = []
    paired_final = None
    for per_campaign in range(1, 6):
        ids = selected_ids(metadata, per_campaign)
        keep = metadata.dataset_id.isin(ids).to_numpy()
        local_meta = metadata.loc[keep].reset_index(drop=True)
        local_pred = pivot.loc[local_meta.dataset_id].to_numpy(float)
        observed = local_meta.experimental_m_per_s.to_numpy(float)
        cv = leave_one_campaign_out(
            observed, local_pred, local_meta.campaign_id.to_numpy(),
            mechanism_names=list(pivot.columns), sample_scale=observed,
            campaign_equal_weighted=True,
        )
        for method in ("stacked", "equal_weight", "best_single"):
            error = np.abs((cv[method] - cv.observed) / cv.observed)
            rows.append({
                "maximum_points_per_campaign": per_campaign,
                "common_points": len(cv),
                "campaigns": cv.held_out_campaign.nunique(),
                "method": method,
                "mean_absolute_relative_error": error.mean(),
                "median_absolute_relative_error": error.median(),
                "root_mean_squared_relative_error": float(np.sqrt(np.mean(error**2))),
            })
        if per_campaign == 5:
            for method in ("stacked", "equal_weight", "best_single"):
                cv[f"absolute_relative_error_{method}"] = np.abs(
                    (cv[method] - cv.observed) / cv.observed
                )
            paired_final = cv.groupby("held_out_campaign", as_index=False).agg(
                points=("observed", "size"),
                stacked_error=("absolute_relative_error_stacked", "mean"),
                equal_weight_error=("absolute_relative_error_equal_weight", "mean"),
                best_single_error=("absolute_relative_error_best_single", "mean"),
            )
            paired_final["stacking_minus_equal_weight"] = (
                paired_final.stacked_error - paired_final.equal_weight_error
            )
            paired_final["stacking_minus_best_single"] = (
                paired_final.stacked_error - paired_final.best_single_error
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "lbv_sampling_count_sensitivity.csv", index=False, lineterminator="\n"
    )
    assert paired_final is not None
    paired_final.to_csv(
        args.output_dir / "lbv_paired_campaign_differences.csv", index=False, lineterminator="\n"
    )
    laboratory_audit = pd.DataFrame([{
        "common_points": len(metadata),
        "campaigns": metadata.campaign_id.nunique(),
        "nonmissing_structured_laboratory_points": int(metadata.laboratory.notna().sum()),
        "leave_one_laboratory_out_feasible": bool(
            metadata.dropna(subset=["laboratory"]).laboratory.nunique() >= 2
        ),
        "reason": "ReSpecTh laboratory field is empty for the common LBV set; curator is not the experimental laboratory.",
    }])
    laboratory_audit.to_csv(
        args.output_dir / "lbv_laboratory_metadata_audit.csv", index=False, lineterminator="\n"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(paired_final.to_string(index=False))
    print(laboratory_audit.to_string(index=False))


if __name__ == "__main__":
    main()
