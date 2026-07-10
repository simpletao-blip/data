"""Major-revision robustness checks requested for Fuel resubmission.

The analysis keeps the Pareto output explicitly at the level of cross-test
fuel-composition screening. It tests whether selections survive alternative
dispersion definitions, GP point estimates, and leave-one-mechanism-out sets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.pareto import mechanism_robust_summary, pareto_mask


MINIMIZE = [
    "ignition_delay_s", "EI_g_per_MJ_NOx_NO2eq",
    "EI_g_per_MJ_N2O", "EI_g_per_MJ_NH3_slip",
]
MAXIMIZE = ["lbv_screen_m_per_s"]
AGGREGATE_MINIMIZE = MINIMIZE + ["idt_shortening_ratio"]
AGGREGATE_MAXIMIZE = MAXIMIZE + ["lbv_enhancement_ratio"]
CONDITION = [
    "design_id", "temperature_K", "flame_temperature_K", "ignition_temperature_K",
    "pressure_bar", "equivalence_ratio", "cracking_ratio", "inside_lbv_convex_hull",
    "inside_all_idt_convex_hull", "inside_exact_criterion_convex_hull",
]


def add_dispersion(summary: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for objective in MINIMIZE + MAXIMIZE:
        grouped = predictions.groupby("design_id")[objective]
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        maximum = grouped.max()
        minimum = grouped.min()
        median = grouped.median()
        positive = predictions.loc[predictions[objective] > 0, objective]
        floor = max(float(positive.quantile(0.01)) * 1e-3, 1e-30) if len(positive) else 1e-30
        ids = result.design_id
        scale = maximum.abs().clip(lower=1e-30)
        result[f"{objective}_iqr_relative"] = ids.map((q75 - q25) / scale)
        result[f"{objective}_max_median_ratio"] = ids.map(maximum / median.abs().clip(lower=floor))
        result[f"{objective}_log_span"] = ids.map(
            np.log10(maximum + floor) - np.log10(minimum.clip(lower=0) + floor)
        )
    return result


def selected_ids(summary: pd.DataFrame, metric_suffix: str, quantile: float) -> set[str]:
    base = (
        summary.mechanism_count.eq(summary.mechanism_count.max())
        & summary.inside_lbv_convex_hull.astype(bool)
        & summary.inside_all_idt_convex_hull.astype(bool)
        & (summary.lbv_enhancement_ratio_worst > 1.0)
        & (summary.idt_shortening_ratio_worst < 1.0)
    )
    supported = summary.loc[base].copy()
    eligible = np.ones(len(supported), dtype=bool)
    for objective in MINIMIZE + MAXIMIZE:
        column = f"{objective}_{metric_suffix}"
        eligible &= supported[column].to_numpy() <= supported[column].quantile(quantile)
    if not eligible.any():
        return set()
    mask = pareto_mask(
        supported.loc[eligible],
        minimize=[f"{name}_worst" for name in MINIMIZE],
        maximize=[f"{name}_worst" for name in MAXIMIZE],
    )
    return set(supported.loc[eligible].loc[mask, "design_id"])


def summarize(predictions: pd.DataFrame, lbv_basis: str) -> pd.DataFrame:
    work = predictions.copy()
    work["lbv_screen_m_per_s"] = work[lbv_basis]
    robust = mechanism_robust_summary(
        work, CONDITION, AGGREGATE_MINIMIZE, AGGREGATE_MAXIMIZE
    )
    return add_dispersion(robust, work)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("results/processed/full_proxy_mechanism_predictions.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/processed/revision_pareto_robustness.csv"))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    mechanisms = sorted(data.mechanism_id.unique())
    rows: list[dict[str, object]] = []
    variants = [
        ("normalized_range", "relative_range", 0.75, "lbv_conservative_m_per_s"),
        ("iqr_relative", "iqr_relative", 0.75, "lbv_conservative_m_per_s"),
        ("max_median_ratio", "max_median_ratio", 0.75, "lbv_conservative_m_per_s"),
        ("log_span", "log_span", 0.75, "lbv_conservative_m_per_s"),
        ("gp_mean_normalized_range", "relative_range", 0.75, "lbv_surrogate_m_per_s"),
    ]
    subsets = [("all", data)] + [
        (f"drop_{mechanism}", data[~data.mechanism_id.eq(mechanism)].copy())
        for mechanism in mechanisms
    ]
    for subset_name, subset in subsets:
        for variant, suffix, quantile, lbv_basis in variants:
            robust = summarize(subset, lbv_basis)
            chosen = selected_ids(robust, suffix, quantile)
            for design_id in sorted(chosen):
                condition = robust.loc[robust.design_id.eq(design_id)].iloc[0]
                rows.append({
                    "mechanism_subset": subset_name,
                    "mechanism_count": subset.mechanism_id.nunique(),
                    "dispersion_variant": variant,
                    "dispersion_quantile": quantile,
                    "lbv_basis": lbv_basis,
                    "design_id": design_id,
                    "pressure_bar": condition.pressure_bar,
                    "equivalence_ratio": condition.equivalence_ratio,
                    "cracking_ratio": condition.cracking_ratio,
                    "selected_count": len(chosen),
                })
            if not chosen:
                rows.append({
                    "mechanism_subset": subset_name,
                    "mechanism_count": subset.mechanism_id.nunique(),
                    "dispersion_variant": variant,
                    "dispersion_quantile": quantile,
                    "lbv_basis": lbv_basis,
                    "design_id": "",
                    "pressure_bar": np.nan,
                    "equivalence_ratio": np.nan,
                    "cracking_ratio": np.nan,
                    "selected_count": 0,
                })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, lineterminator="\n")
    print(result.groupby(["mechanism_subset", "dispersion_variant"]).selected_count.max().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
