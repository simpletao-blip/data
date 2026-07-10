"""Test sensitivity of supported Pareto candidates to the dispersion quantile."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pca_ensemble.pareto import robust_pareto_mask


MINIMIZE = [
    "EI_g_per_MJ_NOx_NO2eq",
    "EI_g_per_MJ_N2O",
    "EI_g_per_MJ_NH3_slip",
    "ignition_delay_s",
]
MAXIMIZE = ["lbv_conservative_m_per_s"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("results/processed/full_proxy_robust_summary.csv")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/processed/pareto_dispersion_quantile_sensitivity.csv"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    eligible = frame[
        frame.reactivity_improved.astype(bool)
        & frame.complete_mechanism_set.astype(bool)
        & frame.support_tier.isin(["strict_criterion_supported", "proxy_supported"])
    ].copy()
    rows = []
    for quantile in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00):
        mask = robust_pareto_mask(
            eligible,
            minimize=MINIMIZE,
            maximize=MAXIMIZE,
            dispersion_quantile=quantile,
        )
        selected = eligible.loc[mask]
        selected_ids = set(selected.design_id)
        for _, row in eligible.iterrows():
            rows.append(
                {
                    "dispersion_quantile": quantile,
                    "design_id": row.design_id,
                    "pressure_bar": row.pressure_bar,
                    "equivalence_ratio": row.equivalence_ratio,
                    "cracking_ratio": row.cracking_ratio,
                    "support_tier": row.support_tier,
                    "selected": row.design_id in selected_ids,
                    "selected_count_at_quantile": len(selected_ids),
                }
            )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, lineterminator="\n")
    summary = (
        result[result.selected]
        .groupby("dispersion_quantile")
        .agg(
            selected_count=("design_id", "nunique"),
            selected_ids=("design_id", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    summary.to_csv(args.output.with_name("pareto_dispersion_quantile_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
