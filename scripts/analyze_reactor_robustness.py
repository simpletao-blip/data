"""Aggregate reactor emissions across mechanisms without combining unlike impacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pca_ensemble.pareto import mechanism_robust_summary, robust_pareto_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    parser.add_argument("--thermo-limits", type=Path,
                        default=Path("mechanisms/thermo_limits.csv"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    data = data[data.status.eq("completed")].copy()
    expected_mechanisms = data.mechanism_id.nunique()
    limits = pd.read_csv(args.thermo_limits).set_index("mechanism_id")
    missing_limits = sorted(set(data.mechanism_id).difference(limits.index))
    if missing_limits:
        raise ValueError(f"missing thermodynamic limits for: {missing_limits}")
    data["thermo_min_temperature_K"] = data.mechanism_id.map(limits.min_temperature_K)
    data["thermo_max_temperature_K"] = data.mechanism_id.map(limits.max_temperature_K)
    data["within_mechanism_thermo_range"] = (
        data.psr_temperature_K.between(
            data.thermo_min_temperature_K, data.thermo_max_temperature_K
        )
        & data.outlet_temperature_K.between(
            data.thermo_min_temperature_K, data.thermo_max_temperature_K
        )
    )
    thermo_excluded = data[~data.within_mechanism_thermo_range].copy()
    data = data[data.within_mechanism_thermo_range].copy()
    # Report NOx on a NO2-equivalent mass basis; N2O and NH3 remain separate.
    # Prefer the mechanism-independent standard-LHV denominator when supplied.
    if "EI_standard_LHV_g_per_MJ_NOx_NO2eq" in data:
        data["NOx_NO2eq_g_per_MJ"] = data.EI_standard_LHV_g_per_MJ_NOx_NO2eq
        data["N2O_g_per_MJ"] = data.EI_standard_LHV_g_per_MJ_N2O
        data["NH3_slip_g_per_MJ"] = data.EI_standard_LHV_g_per_MJ_NH3
        data["emission_heat_basis"] = "fixed_standard_LHV"
    else:
        data["NOx_NO2eq_g_per_MJ"] = (
            data.EI_g_per_MJ_NO * (46.0055 / 30.0061) + data.EI_g_per_MJ_NO2
        )
        data["N2O_g_per_MJ"] = data.EI_g_per_MJ_N2O
        data["NH3_slip_g_per_MJ"] = data.EI_g_per_MJ_NH3
        data["emission_heat_basis"] = "mechanism_thermochemistry"
    condition_columns = [
        "design_id", "design_role", "temperature_K", "pressure_bar", "residence_time_ms",
        "equivalence_ratio", "cracking_ratio", "psr_fraction", "heat_loss_W_per_K",
    ]
    objectives = ["NOx_NO2eq_g_per_MJ", "N2O_g_per_MJ", "NH3_slip_g_per_MJ"]
    summary = mechanism_robust_summary(data, condition_columns, objectives, [])
    summary["complete_mechanism_set"] = summary.mechanism_count.eq(expected_mechanisms)
    summary["emissions_only_robust_pareto"] = False
    eligible = summary.complete_mechanism_set
    if eligible.any():
        summary.loc[eligible, "emissions_only_robust_pareto"] = robust_pareto_mask(
            summary.loc[eligible], objectives, [], dispersion_quantile=0.75
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "reactor_mechanism_predictions.csv", index=False)
    thermo_excluded.to_csv(
        args.output_dir / "reactor_thermo_excluded_predictions.csv", index=False
    )
    summary.to_csv(args.output_dir / "reactor_robust_summary.csv", index=False)
    summary[summary.emissions_only_robust_pareto].to_csv(
        args.output_dir / "reactor_emissions_only_pareto.csv", index=False
    )
    print(f"conditions={len(summary)} mechanisms={data.mechanism_id.nunique()}")
    print(f"emissions_only_robust_pareto={summary.emissions_only_robust_pareto.sum()}")
    print(summary[[f"{name}_relative_range" for name in objectives]].describe().to_string())


if __name__ == "__main__":
    main()
