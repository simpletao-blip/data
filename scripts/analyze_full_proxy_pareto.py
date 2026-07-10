"""Join paired proxies and compute mechanism-robust Pareto screening results.

The resulting front is a screening front.  Any surrogate-selected finalist must
be rerun as a direct Cantera flame before it can support a manuscript claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.pareto import mechanism_robust_summary, robust_pareto_mask


def read_many(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one input file is required")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def add_relative_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    keys = ["mechanism_id", "temperature_K", "pressure_bar", "equivalence_ratio"]
    baseline = work[work.cracking_ratio.eq(0.0)][
        keys + ["lbv_conservative_m_per_s", "ignition_delay_s"]
    ].rename(columns={
        "lbv_conservative_m_per_s": "pure_nh3_lbv_m_per_s",
        "ignition_delay_s": "pure_nh3_ignition_delay_s",
    })
    work = work.merge(baseline, on=keys, how="left", validate="many_to_one")
    work["lbv_enhancement_ratio"] = (
        work.lbv_conservative_m_per_s / work.pure_nh3_lbv_m_per_s
    )
    work["idt_shortening_ratio"] = (
        work.ignition_delay_s / work.pure_nh3_ignition_delay_s
    )
    return work


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lbv", type=Path, nargs="+", required=True)
    parser.add_argument("--idt", type=Path, nargs="+", required=True)
    parser.add_argument("--reactor", type=Path, nargs="+", required=True)
    parser.add_argument("--lbv-admissibility", type=Path, required=True,
                        help="Holdout-tested numerical interpolation gate from evaluate_lbv_surrogate.py")
    parser.add_argument("--thermo-limits", type=Path,
                        default=Path("mechanisms/thermo_limits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/processed"))
    args = parser.parse_args()

    lbv = read_many(args.lbv)
    idt = read_many(args.idt)
    reactor = read_many(args.reactor)
    admissibility = pd.read_csv(args.lbv_admissibility)
    admitted = set(admissibility.loc[
        admissibility.surrogate_admissible_inside_hull, "mechanism_id"
    ])
    if not admitted:
        raise ValueError("no LBV surrogate passed the inside-hull interpolation gate")
    lbv = lbv[lbv.mechanism_id.isin(admitted)].copy()
    idt = idt[idt.mechanism_id.isin(admitted)].copy()
    reactor = reactor[reactor.mechanism_id.isin(admitted)].copy()
    expected_mechanisms = sorted(
        set(lbv.mechanism_id) & set(idt.mechanism_id) & set(reactor.mechanism_id)
    )
    lbv = lbv.rename(columns={
        "surrogate_m_per_s": "lbv_surrogate_m_per_s",
        "surrogate_90_low_m_per_s": "lbv_conservative_m_per_s",
    })
    idt = idt[idt.status.eq("completed")].copy()
    reactor = reactor[reactor.status.eq("completed")].copy()
    limits = pd.read_csv(args.thermo_limits).set_index("mechanism_id")
    missing_limits = sorted(set(expected_mechanisms).difference(limits.index))
    if missing_limits:
        raise ValueError(f"missing thermodynamic limits for: {missing_limits}")
    reactor["thermo_min_temperature_K"] = reactor.mechanism_id.map(
        limits.min_temperature_K
    )
    reactor["thermo_max_temperature_K"] = reactor.mechanism_id.map(
        limits.max_temperature_K
    )
    reactor["within_mechanism_thermo_range"] = (
        reactor.psr_temperature_K.between(
            reactor.thermo_min_temperature_K, reactor.thermo_max_temperature_K
        )
        & reactor.outlet_temperature_K.between(
            reactor.thermo_min_temperature_K, reactor.thermo_max_temperature_K
        )
    )
    thermo_excluded = reactor[~reactor.within_mechanism_thermo_range].copy()
    reactor = reactor[reactor.within_mechanism_thermo_range].copy()
    if "EI_standard_LHV_g_per_MJ_NOx_NO2eq" in reactor:
        reactor["EI_g_per_MJ_NOx_NO2eq"] = reactor["EI_standard_LHV_g_per_MJ_NOx_NO2eq"]
        reactor["EI_g_per_MJ_N2O"] = reactor["EI_standard_LHV_g_per_MJ_N2O"]
        reactor["EI_g_per_MJ_NH3"] = reactor["EI_standard_LHV_g_per_MJ_NH3"]
        reactor["emission_heat_basis"] = "fixed_standard_LHV"
    else:
        reactor["EI_g_per_MJ_NOx_NO2eq"] = (
            reactor["EI_g_per_MJ_NO"].fillna(0.0) * (46.0055 / 30.0061)
            + reactor["EI_g_per_MJ_NO2"].fillna(0.0)
        )
        reactor["emission_heat_basis"] = "mechanism_thermochemistry"
    reactor = reactor.rename(columns={
        "EI_g_per_MJ_N2O": "EI_g_per_MJ_N2O",
        "EI_g_per_MJ_NH3": "EI_g_per_MJ_NH3_slip",
    })
    lbv_columns = [
        "design_id", "mechanism_id", "lbv_surrogate_m_per_s",
        "lbv_conservative_m_per_s", "surrogate_log_standard_deviation",
    ]
    idt_columns = ["design_id", "mechanism_id", "ignition_delay_s"]
    reactor_columns = [
        "design_id", "mechanism_id", "EI_g_per_MJ_NOx_NO2eq", "EI_g_per_MJ_N2O",
        "EI_g_per_MJ_NH3_slip", "outlet_temperature_K",
    ]
    joined = lbv.merge(idt[idt_columns], on=["design_id", "mechanism_id"], validate="one_to_one")
    joined = joined.merge(
        reactor[reactor_columns], on=["design_id", "mechanism_id"], validate="one_to_one"
    )
    joined = add_relative_baselines(joined)

    condition_columns = [
        "design_id", "temperature_K", "flame_temperature_K", "ignition_temperature_K",
        "pressure_bar", "equivalence_ratio", "cracking_ratio", "inside_lbv_convex_hull",
        "inside_all_idt_convex_hull", "inside_exact_criterion_convex_hull",
    ]
    pareto_minimize = [
        "ignition_delay_s", "EI_g_per_MJ_NOx_NO2eq",
        "EI_g_per_MJ_N2O", "EI_g_per_MJ_NH3_slip",
    ]
    pareto_maximize = ["lbv_conservative_m_per_s"]
    aggregate_minimize = [
        "ignition_delay_s", "idt_shortening_ratio", "EI_g_per_MJ_NOx_NO2eq",
        "EI_g_per_MJ_N2O", "EI_g_per_MJ_NH3_slip",
    ]
    aggregate_maximize = ["lbv_conservative_m_per_s", "lbv_enhancement_ratio"]
    robust = mechanism_robust_summary(
        joined, condition_columns, aggregate_minimize, aggregate_maximize
    )
    robust["complete_mechanism_set"] = robust.mechanism_count.eq(len(expected_mechanisms))
    robust["support_tier"] = np.select(
        [
            ~robust.complete_mechanism_set,
            robust.inside_lbv_convex_hull & robust.inside_exact_criterion_convex_hull,
            robust.inside_lbv_convex_hull & robust.inside_all_idt_convex_hull,
        ],
        ["mechanism_thermo_extrapolation", "strict_criterion_supported", "proxy_supported"],
        default="extrapolative",
    )
    robust["reactivity_improved"] = (
        (robust.lbv_enhancement_ratio_worst > 1.0)
        & (robust.idt_shortening_ratio_worst < 1.0)
    )
    # Supported and extrapolative points must not compete on the same front:
    # otherwise an unsupported point can dominate a point inside the data hull.
    robust["screening_pareto"] = False
    supported = robust.support_tier.isin([
        "strict_criterion_supported", "proxy_supported"
    ])
    eligible = robust.reactivity_improved & robust.complete_mechanism_set & supported
    if eligible.any():
        robust.loc[eligible, "screening_pareto"] = robust_pareto_mask(
            robust.loc[eligible],
            minimize=pareto_minimize,
            maximize=pareto_maximize,
            dispersion_quantile=0.75,
        )
    robust["exploratory_extrapolative_pareto"] = False
    exploratory = (
        robust.reactivity_improved
        & robust.complete_mechanism_set
        & robust.support_tier.eq("extrapolative")
    )
    if exploratory.any():
        robust.loc[exploratory, "exploratory_extrapolative_pareto"] = robust_pareto_mask(
            robust.loc[exploratory],
            minimize=pareto_minimize,
            maximize=pareto_maximize,
            dispersion_quantile=0.75,
        )
    robust["requires_direct_flame_confirmation"] = robust.screening_pareto

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.output_dir / "full_proxy_mechanism_predictions.csv", index=False)
    thermo_excluded.to_csv(
        args.output_dir / "full_proxy_thermo_excluded_reactor_rows.csv", index=False
    )
    robust.to_csv(args.output_dir / "full_proxy_robust_summary.csv", index=False)
    robust[robust.screening_pareto].to_csv(
        args.output_dir / "full_proxy_screening_pareto.csv", index=False
    )
    robust[robust.exploratory_extrapolative_pareto].to_csv(
        args.output_dir / "full_proxy_exploratory_extrapolative_pareto.csv", index=False
    )
    print(robust.groupby(["support_tier", "screening_pareto"]).size().to_string())
    print(args.output_dir / "full_proxy_screening_pareto.csv")


if __name__ == "__main__":
    main()
