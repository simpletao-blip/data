"""Audit local and leave-one-study-out support for LBV screening candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.design import convex_hull_membership


SPECIES = ["NH3", "H2", "O2", "N2", "AR", "HE"]
EXCLUDED_DEVELOPMENT_DOIS = {
    "10.1016/j.combustflame.2021.111472",
    "10.1016/j.combustflame.2019.08.033",
}


def parse(raw: str) -> dict[str, float]:
    values = json.loads(raw) if isinstance(raw, str) and raw else {}
    return {str(key).upper(): float(value) for key, value in values.items()}


def mixture_class(raw: str) -> str:
    oxidizer = parse(raw)
    if oxidizer.get("AR", 0.0) > 0 or oxidizer.get("HE", 0.0) > 0:
        return "inert_diluted"
    denominator = oxidizer.get("O2", 0.0) + oxidizer.get("N2", 0.0)
    if denominator <= 0:
        return "other"
    oxygen_fraction = oxidizer.get("O2", 0.0) / denominator
    if 0.18 <= oxygen_fraction <= 0.23:
        return "air_like"
    if oxygen_fraction > 0.23:
        return "oxygen_enriched"
    return "oxygen_lean_or_other"


def candidate_composition(alpha: float, phi: float) -> dict[str, float]:
    amounts = {
        "NH3": 1.0 - alpha,
        "H2": 1.5 * alpha,
        "N2": 0.5 * alpha + 3.76 * 0.75 / phi,
        "O2": 0.75 / phi,
        "AR": 0.0,
        "HE": 0.0,
    }
    total = sum(amounts.values())
    return {name: amounts[name] / total for name in SPECIES}


def features(frame: pd.DataFrame, composition_column: str) -> np.ndarray:
    base = np.column_stack([
        frame.temperature_K.to_numpy(float),
        np.log10(frame.pressure_Pa.to_numpy(float)),
        frame.equivalence_ratio.to_numpy(float),
        frame.cracking_ratio.to_numpy(float),
    ])
    composition = np.array([
        [parse(raw).get(name, 0.0) for name in SPECIES]
        for raw in frame[composition_column]
    ])
    composition /= composition.sum(axis=1, keepdims=True)
    return np.column_stack([base, composition])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("data/processed/respecth_nh3_long.csv"))
    parser.add_argument("--design", type=Path,
                        default=Path("data/processed/proxy_screening_design.csv"))
    parser.add_argument("--pareto", type=Path,
                        default=Path("results/processed/full_proxy_screening_pareto.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/processed/lbv_candidate_domain_support.csv"))
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    lbv = data[
        data.observable.eq("laminar burning velocity")
        & ~data.doi.isin(EXCLUDED_DEVELOPMENT_DOIS)
    ].copy()
    lbv["mixture_class"] = lbv.oxidizer_composition.map(mixture_class)
    air = lbv[lbv.mixture_class.eq("air_like")].dropna(
        subset=["temperature_K", "pressure_Pa", "equivalence_ratio", "cracking_ratio"]
    ).copy()
    candidate_ids = set(pd.read_csv(args.pareto).design_id)
    candidates = pd.read_csv(args.design)
    candidates = candidates[candidates.design_id.isin(candidate_ids)].copy()
    candidates["temperature_K"] = candidates.flame_temperature_K
    candidates["pressure_Pa"] = candidates.pressure_bar * 1e5
    candidates["initial_composition"] = candidates.apply(
        lambda row: json.dumps(candidate_composition(row.cracking_ratio, row.equivalence_ratio)),
        axis=1,
    )
    reference = features(air, "initial_composition")
    query = features(candidates, "initial_composition")
    low = np.nanmin(reference, axis=0)
    span = np.nanmax(reference, axis=0) - low
    span[span == 0.0] = 1.0
    distances = np.linalg.norm(
        ((query - low) / span)[:, None, :] - ((reference - low) / span)[None, :, :],
        axis=2,
    )
    four_reference = reference[:, :4]
    four_query = query[:, :4]
    full_hull = convex_hull_membership(four_reference, four_query)
    campaigns = sorted(air.campaign_id.unique())
    loso = np.zeros((len(candidates), len(campaigns)), dtype=bool)
    for index, campaign in enumerate(campaigns):
        keep = ~air.campaign_id.eq(campaign).to_numpy()
        loso[:, index] = convex_hull_membership(four_reference[keep], four_query)
    rows = []
    for index, (_, candidate) in enumerate(candidates.iterrows()):
        ordered = np.sort(distances[index])
        rows.append({
            "design_id": candidate.design_id,
            "pressure_bar": candidate.pressure_bar,
            "equivalence_ratio": candidate.equivalence_ratio,
            "cracking_ratio": candidate.cracking_ratio,
            "reference_mixture_class": "air_like_only",
            "reference_points": len(air),
            "reference_campaigns": air.campaign_id.nunique(),
            "inside_air_like_4d_hull": bool(full_hull[index]),
            "nearest_extended_distance": ordered[0],
            "third_nearest_extended_distance": ordered[min(2, len(ordered) - 1)],
            "fifth_nearest_extended_distance": ordered[min(4, len(ordered) - 1)],
            "neighbors_within_0_5": int(np.sum(distances[index] <= 0.5)),
            "loso_hull_pass_count": int(loso[index].sum()),
            "loso_hull_fold_count": len(campaigns),
            "loso_hull_pass_fraction": float(loso[index].mean()),
            "loso_hull_failed_campaigns": ";".join(
                campaign for campaign, passed in zip(campaigns, loso[index], strict=True)
                if not passed
            ),
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, lineterminator="\n")
    class_counts = lbv.groupby("mixture_class").agg(
        points=("dataset_id", "nunique"), campaigns=("campaign_id", "nunique")
    ).reset_index()
    class_counts.to_csv(args.output.with_name("lbv_mixture_class_coverage.csv"), index=False)
    print(result.to_string(index=False))
    print(class_counts.to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
