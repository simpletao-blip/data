"""Generate Supplementary Figs. S1-S11 from existing audited results."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPORTS = Path("figures/supplementary")
SOURCE = Path("figures/source_data")
BLUE = "#4477AA"
ORANGE = "#EE9944"
GREEN = "#228833"
RED = "#CC6677"
CYAN = "#66CCEE"
PURPLE = "#AA3377"
GRAY = "#68757D"


def setup() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def panel_labels(axes) -> None:
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
        ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")


def save(fig, stem: str) -> None:
    base = EXPORTS / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_s1() -> None:
    data = pd.read_csv("literature/evidence_matrix.csv").fillna("")
    data.to_csv(SOURCE / "figureS1_literature_evidence.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    use = data.primary_use.replace("", "unclassified").value_counts().sort_values()
    axes[0].barh(use.index, use.values, color=BLUE)
    axes[0].set_xlabel("Records")
    axes[0].set_title("Evidence role")
    years = pd.to_numeric(data.year, errors="coerce").dropna().astype(int)
    axes[1].hist(years, bins=np.arange(years.min() - 0.5, years.max() + 1.5),
                 color=ORANGE, edgecolor="white")
    axes[1].set_xlabel("Publication year")
    axes[1].set_ylabel("Records")
    axes[1].set_title("Search-era coverage")
    panel_labels(axes)
    fig.suptitle("Literature evidence was screened by role and year", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS1_literature_evidence")


def observable_class(value: str, experiment_type: str) -> str:
    if value == "ignition delay":
        return "IDT"
    if value == "laminar burning velocity":
        return "LBV"
    return "species"


def fig_s2() -> None:
    data = pd.read_csv("data/processed/respecth_nh3_long.csv")
    data["observable_class"] = [observable_class(o, e) for o, e in zip(data.observable, data.experiment_type)]
    cols = ["dataset_id", "campaign_id", "observable_class", "observable", "temperature_K",
            "pressure_Pa", "equivalence_ratio", "cracking_ratio", "apparatus"]
    data[cols].to_csv(SOURCE / "figureS2_data_coverage.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    palette = {"IDT": RED, "LBV": BLUE, "species": GREEN}
    for label, group in data.dropna(subset=["temperature_K", "pressure_Pa"]).groupby("observable_class"):
        axes[0].scatter(group.temperature_K, group.pressure_Pa / 1e5, s=8, alpha=0.45,
                        label=label, color=palette[label], edgecolors="none")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Pressure (bar)")
    axes[0].set_title("Thermodynamic coverage")
    axes[0].legend()
    app = data.groupby("apparatus").campaign_id.nunique().sort_values(ascending=False).head(9).sort_values()
    labels = [text if len(text) <= 34 else text[:31] + "..." for text in app.index]
    axes[1].barh(labels, app.values, color=CYAN)
    axes[1].set_xlabel("Unique campaigns")
    axes[1].set_title("Most represented apparatuses")
    panel_labels(axes)
    fig.suptitle("Public records span heterogeneous apparatus and operating domains", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS2_data_coverage")


def idt_long() -> pd.DataFrame:
    data = pd.read_csv("results/processed/idt_common_predictions.csv")
    mechanisms = [c for c in data.columns if c not in {
        "dataset_id", "campaign_id", "doi", "experimental_s", "temperature_K", "pressure_Pa",
        "definition", "equal_log_ensemble_s", "equal_log_ensemble_abs_log10_error"
    }]
    long = data.melt(
        id_vars=["dataset_id", "temperature_K", "pressure_Pa", "experimental_s"],
        value_vars=mechanisms, var_name="mechanism_id", value_name="predicted_s",
    ).dropna()
    long["signed_log10_error"] = np.log10(long.predicted_s / long.experimental_s)
    long["absolute_log10_error"] = long.signed_log10_error.abs()
    return long


def fig_s3() -> None:
    long = idt_long()
    long.to_csv(SOURCE / "figureS3_idt_criterion_errors.csv", index=False)
    summary = long.groupby("mechanism_id").absolute_log10_error.mean().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axes[0].barh(summary.index, summary.values, color=BLUE)
    axes[0].set_xlabel("Mean |log10 error|")
    axes[0].set_title("Exact-criterion mechanism error")
    equal = pd.read_csv("results/processed/idt_common_predictions.csv")
    sc = axes[1].scatter(equal.temperature_K, equal.equal_log_ensemble_abs_log10_error,
                         c=equal.pressure_Pa / 1e5, cmap="viridis", s=20)
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Equal-ensemble |log10 error|")
    axes[1].set_title("One-study criterion-matched cases")
    fig.colorbar(sc, ax=axes[1], label="Pressure (bar)")
    panel_labels(axes)
    fig.suptitle("Criterion-matched ignition evidence ranks mechanisms but cannot test a new study", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS3_idt_criterion")


def mechanism_name(path: str) -> str:
    text = path.replace("\\", "/").lower()
    mapping = [
        ("polimi", "POLIMI_2023"), ("konnov", "Konnov_2026"),
        ("nuigmech1.1", "NUIGMech1.1_HT"), ("shrestha", "Shrestha_2018"),
        ("mei_2019", "MEI_2019"), ("c3-v3.5", "C3_v3_5_C0"),
        ("chem_linear_burke", "RMG_2026_Burke"), ("nuig_2024", "NUIG_2024"),
        ("mei_2021", "MEI_2021"), ("otomo", "Otomo_2018"),
        ("kaust", "KAUST_2023"),
    ]
    return next((name for token, name in mapping if token in text), "")


def fig_s4() -> None:
    frames = [pd.read_csv("results/logs/mechanism_audit.csv"),
              pd.read_csv("results/logs/new_mechanisms_audit.csv")]
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["mechanism_id"] = data.mechanism_file.astype(str).map(mechanism_name)
    data = data[data.mechanism_id.ne("") & data.species_count.notna()].copy()
    data["priority"] = data.status.eq("passed").astype(int)
    data = data.sort_values("priority", ascending=False).drop_duplicates("mechanism_id")
    data[["mechanism_id", "species_count", "reaction_count", "status", "warning_count",
          "duplicate_equation_count"]].to_csv(SOURCE / "figureS4_mechanism_sizes.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    colors = np.where(data.status.eq("passed"), GREEN, ORANGE)
    axes[0].scatter(data.species_count, data.reaction_count, c=colors, s=30)
    for _, row in data.iterrows():
        axes[0].annotate(row.mechanism_id.replace("_", " "), (row.species_count, row.reaction_count),
                         xytext=(3, 2), textcoords="offset points", fontsize=5.5)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Species count")
    axes[0].set_ylabel("Reaction count")
    axes[0].set_title("Mechanism size")
    ordered = data.sort_values("warning_count")
    axes[1].barh(ordered.mechanism_id, ordered.warning_count.fillna(0), color=colors[data.index.get_indexer(ordered.index)])
    axes[1].set_xlabel("Recorded audit warnings")
    axes[1].set_title("Executable warnings remain visible")
    panel_labels(axes)
    fig.suptitle("Formal mechanisms span two orders of magnitude in size and audit burden", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS4_mechanism_sizes")


def fig_s5() -> None:
    data = pd.read_csv("results/processed/lbv_loco_predictions.csv")
    data.to_csv(SOURCE / "figureS5_lbv_campaign_residuals.csv", index=False)
    campaign = data.groupby("held_out_campaign").agg(
        stacked=("relative_error_stacked", "mean"), equal=("relative_error_equal_weight", "mean"),
        best=("relative_error_best_single", "mean"), n=("sample_index", "size"),
    ).sort_values("stacked")
    y = np.arange(len(campaign))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    axes[0].scatter(campaign["equal"] * 100, y, color=ORANGE, label="Equal", s=18)
    axes[0].scatter(campaign["best"] * 100, y, color=GREEN, label="Training-best", s=18)
    axes[0].scatter(campaign["stacked"] * 100, y, color=BLUE, label="Stacking", s=18)
    axes[0].set_yticks(y, [v.replace("campaign_", "")[:8] for v in campaign.index])
    axes[0].set_xlabel("Campaign MARE (%)")
    axes[0].set_title("Held-study error")
    axes[0].legend(ncol=3, fontsize=6)
    diff = (campaign.stacked - campaign.best) * 100
    axes[1].barh(y, diff, color=np.where(diff <= 0, GREEN, RED))
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Stacking minus best-single MARE (points)")
    axes[1].set_title("Study-specific gain or loss")
    panel_labels(axes)
    fig.suptitle("Stacking improves on equal averaging but not consistently on model selection", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS5_lbv_campaign_residuals")


def fig_s6() -> None:
    long = idt_long()
    long["temperature_bin_K"] = pd.cut(long.temperature_K, bins=6, duplicates="drop")
    heat = long.pivot_table(index="mechanism_id", columns="temperature_bin_K",
                            values="absolute_log10_error", aggfunc="mean", observed=True)
    source = heat.reset_index()
    source.columns = ["mechanism_id"] + [str(c) for c in heat.columns]
    source.to_csv(SOURCE / "figureS6_idt_temperature_pressure.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    im = axes[0].imshow(heat.to_numpy(), aspect="auto", cmap="magma_r", vmin=0)
    axes[0].set_yticks(range(len(heat)), heat.index)
    axes[0].set_xticks(range(len(heat.columns)), [f"{i.left:.0f}-{i.right:.0f}" for i in heat.columns], rotation=40, ha="right")
    axes[0].set_xlabel("Temperature bin (K)")
    axes[0].set_title("Mean |log10 error|")
    fig.colorbar(im, ax=axes[0], label="Error")
    pressure = long.assign(pressure_bar=long.pressure_Pa / 1e5).groupby("mechanism_id").apply(
        lambda g: np.corrcoef(np.log10(g.pressure_bar), g.signed_log10_error)[0, 1],
        include_groups=False,
    ).sort_values()
    axes[1].barh(pressure.index, pressure.values, color=np.where(pressure.values < 0, BLUE, ORANGE))
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Correlation: log pressure vs signed error")
    axes[1].set_title("Pressure-dependent bias within one study")
    panel_labels(axes)
    fig.suptitle("Ignition errors vary with temperature and pressure even under one criterion", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS6_idt_residual_structure")


def fig_s7() -> None:
    data = pd.read_csv("results/logs/POLIMI_2023_flame_grid_check.csv")
    data.to_csv(SOURCE / "figureS7_grid_independence.csv", index=False)
    order = ["coarse", "base", "fine"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for case, group in data.groupby("case"):
        group = group.set_index("grid").loc[order]
        axes[0].plot(order, group.laminar_burning_velocity_m_per_s, marker="o", label=case.replace("_", " "))
        axes[1].plot(order, group.runtime_s, marker="o", label=case.replace("_", " "))
    axes[0].set_ylabel("LBV (m s$^{-1}$)")
    axes[0].set_title("Solution stability")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("Numerical cost")
    axes[1].legend(fontsize=6)
    panel_labels(axes)
    fig.suptitle("Baseline flame grids differ from fine grids by at most 1.32%", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS7_grid_independence")


def fig_s8() -> None:
    data = pd.read_csv("results/processed/lbv_surrogate_holdout_predictions.csv")
    data.to_csv(SOURCE / "figureS8_surrogate_calibration.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    inside = data.inside_experimental_convex_hull.astype(bool)
    for mask, label, color in ((inside, "Inside hull", BLUE), (~inside, "Outside hull", ORANGE)):
        part = data[mask]
        axes[0].scatter(part.laminar_burning_velocity_m_per_s, part.surrogate_m_per_s,
                        s=12, alpha=0.65, label=label, color=color)
    limits = [data[["laminar_burning_velocity_m_per_s", "surrogate_m_per_s"]].min().min(),
              data[["laminar_burning_velocity_m_per_s", "surrogate_m_per_s"]].max().max()]
    axes[0].plot(limits, limits, color="black", lw=0.8)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Direct Cantera LBV (m s$^{-1}$)")
    axes[0].set_ylabel("Surrogate LBV (m s$^{-1}$)")
    axes[0].set_title("Holdout prediction")
    axes[0].legend()
    box = [100 * data.loc[inside & data.mechanism_id.eq(m), "surrogate_relative_error"].dropna().values
           for m in sorted(data.mechanism_id.unique())]
    axes[1].boxplot(box, tick_labels=[m.replace("_", "\n") for m in sorted(data.mechanism_id.unique())],
                    showfliers=False)
    axes[1].tick_params(axis="x", labelsize=5.5, rotation=35)
    axes[1].set_ylabel("Inside-hull relative error (%)")
    axes[1].set_title("Mechanism-specific calibration")
    panel_labels(axes)
    fig.suptitle("All eight LBV interpolators pass the inside-hull holdout gate", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS8_surrogate_calibration")


def reactor_data() -> pd.DataFrame:
    files = sorted(Path("results/raw").glob("*_reactor_map.csv"))
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def add_nox(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["EI_g_per_MJ_NOx_NO2eq"] = data.EI_g_per_MJ_NO.fillna(0) * (46.0055 / 30.0061) + data.EI_g_per_MJ_NO2.fillna(0)
    return data


def fig_s9() -> None:
    data = add_nox(reactor_data())
    anchors = {(0.1, 0.7), (0.3, 1.0), (0.5, 1.2), (0.7, 1.0)}
    matched = data[
        data.status.eq("completed") & data.temperature_K.eq(750) & data.pressure_bar.eq(10)
        & data.residence_time_ms.eq(10) & data.heat_loss_W_per_K.eq(0)
        & pd.Series(list(zip(data.cracking_ratio, data.equivalence_ratio)), index=data.index).isin(anchors)
        & data.psr_fraction.isin([0.3, 0.5, 0.7])
    ].copy()
    matched.to_csv(SOURCE / "figureS9_psr_fraction_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for alpha, group in matched.groupby("cracking_ratio"):
        summary = group.groupby("psr_fraction").EI_g_per_MJ_NOx_NO2eq.median()
        axes[0].plot(summary.index, summary.values, marker="o", label=f"alpha={alpha:g}")
        slip = group.groupby("psr_fraction").EI_g_per_MJ_NH3.median()
        axes[1].plot(slip.index, slip.values, marker="o", label=f"alpha={alpha:g}")
    axes[0].set_xlabel("PSR residence-time fraction")
    axes[0].set_ylabel("Median NOx EI (g MJ$^{-1}$)")
    axes[0].set_title("NOx response")
    axes[1].set_xlabel("PSR residence-time fraction")
    axes[1].set_ylabel("Median NH3-slip EI (g MJ$^{-1}$)")
    axes[1].set_yscale("log")
    axes[1].set_title("NH3-slip response")
    axes[1].legend(fontsize=6, ncol=2)
    panel_labels(axes)
    fig.suptitle("PSR-fraction effects are conditional on cracking ratio and equivalence ratio", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS9_psr_fraction_sensitivity")


def fig_s10() -> None:
    data = add_nox(reactor_data())
    part = data[
        data.status.eq("completed")
        & data.cracking_ratio.eq(0.3) & data.equivalence_ratio.eq(1.0)
        & data.pressure_bar.eq(5) & data.psr_fraction.eq(0.5)
        & data.temperature_K.isin([600, 750])
        & data.residence_time_ms.isin([5, 20])
        & data.heat_loss_W_per_K.isin([0, 0.5, 2.0])
    ].copy()
    part.to_csv(SOURCE / "figureS10_heat_loss_residence.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for residence, group in part.groupby("residence_time_ms"):
        summary = group.groupby("heat_loss_W_per_K").agg(
            nox=("EI_g_per_MJ_NOx_NO2eq", "median"), n2o=("EI_g_per_MJ_N2O", "median"),
            temperature=("outlet_temperature_K", "median")
        )
        axes[0].plot(summary.index, summary.nox, marker="o", label=f"{residence:g} ms")
        axes[1].plot(summary.index, summary.temperature, marker="o", label=f"{residence:g} ms")
    axes[0].set_xlabel("UA sensitivity parameter (W K$^{-1}$)")
    axes[0].set_ylabel("Median NOx EI (g MJ$^{-1}$)")
    axes[0].set_title("Emission response")
    axes[1].set_xlabel("UA sensitivity parameter (W K$^{-1}$)")
    axes[1].set_ylabel("Median outlet temperature (K)")
    axes[1].set_title("Thermal response")
    axes[1].legend()
    panel_labels(axes)
    fig.suptitle("Heat-loss and residence-time effects remain conditional ideal-reactor sensitivities", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS10_heat_loss_sensitivity")


def fig_s11() -> None:
    data = pd.read_csv("results/processed/full_proxy_robust_summary.csv")
    quant = pd.read_csv("results/processed/pareto_dispersion_quantile_summary.csv")
    source = data[["design_id", "pressure_bar", "equivalence_ratio", "cracking_ratio", "support_tier",
                   "screening_pareto", "exploratory_extrapolative_pareto", "reactivity_improved"]].copy()
    source.to_csv(SOURCE / "figureS11_pareto_support.csv", index=False)
    quant.to_csv(SOURCE / "figureS11_dispersion_quantile.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    slice_ = data[data.pressure_bar.eq(5)]
    colors = {"proxy_supported": GREEN, "strict_criterion_supported": BLUE,
              "extrapolative": "#BBBBBB", "mechanism_thermo_extrapolation": RED}
    for tier, group in slice_.groupby("support_tier"):
        axes[0].scatter(group.equivalence_ratio, group.cracking_ratio, s=14,
                        color=colors.get(tier, GRAY), alpha=0.65, label=tier.replace("_", " "))
    selected = slice_[slice_.screening_pareto.astype(bool)]
    axes[0].scatter(selected.equivalence_ratio, selected.cracking_ratio, marker="*", s=90,
                    facecolors="none", edgecolors="black", linewidths=0.9, label="q=0.75 candidates")
    axes[0].set_xlabel("Equivalence ratio")
    axes[0].set_ylabel("Cracking ratio")
    axes[0].set_title("5 bar support tiers")
    axes[0].legend(fontsize=5.5)
    axes[1].plot(quant.dispersion_quantile, quant.selected_count, marker="o", color=PURPLE)
    axes[1].axvline(0.75, color=GRAY, lw=0.8, ls="--")
    axes[1].set_xlabel("Dispersion quantile")
    axes[1].set_ylabel("Supported Pareto candidates")
    axes[1].set_title("Threshold sensitivity")
    axes[1].set_xticks(quant.dispersion_quantile)
    panel_labels(axes)
    fig.suptitle("Only one proxy candidate persists across all tested robustness gates", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "figureS11_pareto_support")


def main() -> None:
    setup()
    for function in (fig_s1, fig_s2, fig_s3, fig_s4, fig_s5, fig_s6,
                     fig_s7, fig_s8, fig_s9, fig_s10, fig_s11):
        function()
        print(function.__name__)
    fig, axes = plt.subplots(3, 4, figsize=(14.4, 8.1))
    for number, ax in enumerate(axes.flat, start=1):
        matches = sorted(EXPORTS.glob(f"figureS{number}_*.png"))
        if matches:
            ax.imshow(plt.imread(matches[0]))
            ax.set_title(f"Figure S{number}", fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(EXPORTS / "qa_contact_sheet.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
