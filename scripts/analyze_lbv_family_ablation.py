"""Test whether the LBV stacking conclusion survives mechanism-family removal."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.metrics import cluster_bootstrap_interval
from pca_ensemble.stacking import nested_grouped_stacking_intervals


def evaluate(data: pd.DataFrame, label: str) -> dict[str, object]:
    metadata = (data.drop_duplicates("dataset_id").set_index("dataset_id")
                [["campaign_id", "experimental_m_per_s"]])
    pivot = data.pivot(index="dataset_id", columns="mechanism_id",
                       values="simulated_m_per_s").dropna()
    metadata = metadata.loc[pivot.index]
    observed = metadata.experimental_m_per_s.to_numpy(float)
    cv = nested_grouped_stacking_intervals(
        observed, pivot.to_numpy(float), metadata.campaign_id.to_numpy(),
        mechanism_names=list(pivot.columns), sample_scale=observed,
        interval_level=0.90,
    )
    output: dict[str, object] = {
        "ablation": label,
        "mechanisms": pivot.shape[1],
        "cases": len(cv),
        "campaigns": cv.held_out_campaign.nunique(),
        "interval_coverage": float(cv.residual_interval_covered.mean()),
    }
    errors: dict[str, np.ndarray] = {}
    for method in ("stacked", "equal_weight", "best_single"):
        errors[method] = np.abs((cv[method].to_numpy() - cv.observed.to_numpy())
                                / cv.observed.to_numpy())
        output[f"mare_{method}"] = float(errors[method].mean())
    for baseline in ("equal_weight", "best_single"):
        difference = errors["stacked"] - errors[baseline]
        estimate, low, high = cluster_bootstrap_interval(
            difference, cv.held_out_campaign.to_numpy(), replicates=2000
        )
        output[f"stack_minus_{baseline}"] = estimate
        output[f"stack_minus_{baseline}_95_low"] = low
        output[f"stack_minus_{baseline}_95_high"] = high
    output["stacking_beats_both"] = bool(
        output["mare_stacked"] < min(output["mare_equal_weight"], output["mare_best_single"])
        and output["stack_minus_equal_weight_95_high"] < 0
        and output["stack_minus_best_single_95_high"] < 0
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--exclude-doi", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    data = data[data.status.eq("completed") & ~data.doi.isin(args.exclude_doi)].copy()
    families = pd.read_csv(args.families).set_index("mechanism_id").family_group
    missing = sorted(set(data.mechanism_id).difference(families.index))
    if missing:
        raise SystemExit(f"missing mechanism family assignments: {missing}")
    data["family_group"] = data.mechanism_id.map(families)

    rows = [evaluate(data, "none")]
    for family in sorted(data.family_group.unique()):
        reduced = data[~data.family_group.eq(family)].copy()
        if reduced.mechanism_id.nunique() >= 2:
            rows.append(evaluate(reduced, f"remove:{family}"))
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
