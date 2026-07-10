"""Recompute key manuscript numbers and flag text/source mismatches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    manuscript = Path("manuscript/manuscript_working.md").read_text(encoding="utf-8")
    method = pd.read_csv("results/processed/lbv_method_summary.csv").set_index("method")
    lbv_gate = pd.read_csv("results/processed/lbv_gate.csv").iloc[0]
    lbv_loco = pd.read_csv("results/processed/lbv_loco_predictions.csv")
    surrogate = pd.read_csv("results/processed/lbv_surrogate_summary.csv")
    admissibility = pd.read_csv("results/processed/lbv_surrogate_admissibility.csv")
    reactor = pd.read_csv("results/processed/reactor_robust_summary.csv")
    reactor_complete = reactor[reactor.complete_mechanism_set]
    robust = pd.read_csv("results/processed/full_proxy_robust_summary.csv")
    confirmation = pd.read_csv("results/processed/pareto_flame_confirmation.csv")
    quantile_summary = pd.read_csv(
        "results/processed/pareto_dispersion_quantile_summary.csv"
    )
    jsr_gate = pd.read_csv("results/processed/jsr_observable_gate.csv")
    rmg_map = pd.read_csv("results/raw/RMG_2026_Burke_lbv_map.csv")
    inside = surrogate[surrogate.inside_experimental_convex_hull.astype(bool)]
    outside = surrogate[~surrogate.inside_experimental_convex_hull.astype(bool)]

    checks = [
        ("record_count", "5,011", "data/processed/respecth_nh3_long.csv"),
        ("source_files", "306", "data/raw/respecth_nh3_v2_3"),
        ("campaigns", "38", "figures/source_data/figure2_coverage_summary.csv"),
        ("stacking_mare", f"{100 * method.loc['stacked', 'mean_absolute_relative_error']:.2f}%",
         "results/processed/lbv_method_summary.csv"),
        ("equal_mare", f"{100 * method.loc['equal_weight', 'mean_absolute_relative_error']:.2f}%",
         "results/processed/lbv_method_summary.csv"),
        ("best_single_mare", f"{100 * method.loc['best_single', 'mean_absolute_relative_error']:.2f}%",
         "results/processed/lbv_method_summary.csv"),
        ("interval_coverage", f"{100 * lbv_loco.residual_interval_covered.mean():.2f}%",
         "results/processed/lbv_loco_predictions.csv"),
        ("inside_surrogate_low", f"{100 * inside.mean_absolute_relative_error.min():.3f}%",
         "results/processed/lbv_surrogate_summary.csv"),
        ("inside_surrogate_high", f"{100 * inside.mean_absolute_relative_error.max():.3f}%",
         "results/processed/lbv_surrogate_summary.csv"),
        ("inside_surrogate_max_case",
         f"{100 * admissibility.inside_hull_max_absolute_relative_error.max():.2f}%",
         "results/processed/lbv_surrogate_admissibility.csv"),
        ("outside_coverage_low", f"{100 * outside.nominal_90_interval_coverage.min():.1f}%",
         "results/processed/lbv_surrogate_summary.csv"),
        ("rmg_map_completion", f"{int(rmg_map.status.eq('completed').sum())}/144",
         "results/raw/RMG_2026_Burke_lbv_map.csv"),
        ("nox_median_range",
         f"{100 * reactor_complete.NOx_NO2eq_g_per_MJ_relative_range.median():.2f}%",
         "results/processed/reactor_robust_summary.csv"),
        ("nox_max_range",
         f"{100 * reactor_complete.NOx_NO2eq_g_per_MJ_relative_range.max():.2f}%",
         "results/processed/reactor_robust_summary.csv"),
        ("n2o_max_range", f"{100 * reactor_complete.N2O_g_per_MJ_relative_range.max():.2f}%",
         "results/processed/reactor_robust_summary.csv"),
        ("nh3_max_range", f"{100 * reactor_complete.NH3_slip_g_per_MJ_relative_range.max():.2f}%",
         "results/processed/reactor_robust_summary.csv"),
        ("proxy_pareto", str(int((robust.screening_pareto & robust.support_tier.eq(
            'proxy_supported')).sum())), "results/processed/full_proxy_robust_summary.csv"),
        ("exploratory_pareto", str(int(robust.exploratory_extrapolative_pareto.sum())),
         "results/processed/full_proxy_robust_summary.csv"),
        ("thermo_incomplete", str(int(robust.support_tier.eq(
            'mechanism_thermo_extrapolation').sum())),
         "results/processed/full_proxy_robust_summary.csv"),
        ("direct_count", str(len(confirmation)),
         "results/processed/pareto_flame_confirmation.csv"),
        ("direct_error_median", f"{100 * confirmation.surrogate_relative_error.median():.2f}%",
         "results/processed/pareto_flame_confirmation.csv"),
        ("direct_error_max", f"{100 * confirmation.surrogate_relative_error.max():.2f}%",
         "results/processed/pareto_flame_confirmation.csv"),
        ("dispersion_quantile_counts",
         ", ".join(str(int(value)) for value in quantile_summary.selected_count.iloc[:-1])
         + f" and {int(quantile_summary.selected_count.iloc[-1])}",
         "results/processed/pareto_dispersion_quantile_summary.csv"),
        ("jsr_gate", f"{int(jsr_gate.stacking_beats_both.sum())}/8",
         "results/processed/jsr_observable_gate.csv"),
        ("idt_count", "46", "results/processed/idt_model_summary.csv"),
    ]
    rows = [
        {"claim_id": claim, "expected_text": value, "present": value in manuscript,
         "source": source}
        for claim, value, source in checks
    ]
    audit = pd.DataFrame(rows)
    output = Path("results/logs/manuscript_number_audit.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    print(audit.to_string(index=False))
    if not audit.present.all():
        missing = audit.loc[~audit.present, "claim_id"].tolist()
        raise SystemExit(f"manuscript number audit failed: {missing}")
    print(output)


if __name__ == "__main__":
    main()
