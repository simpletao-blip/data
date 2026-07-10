"""Figure 7: kinetic, thermal and cracking-energy sensitivity evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POSITIVE = "#4477AA"
NEGATIVE = "#CC6677"
COLORS = {"NOx": "#4477AA", "N2O": "#EE9944", "NH3 slip": "#228833"}


def short_equation(text: str, limit: int = 32) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def sensitivity_panel(ax, frame, title):
    top = frame.nsmallest(8, "absolute_rank").sort_values(
        "normalized_flame_speed_sensitivity"
    )
    values = top.normalized_flame_speed_sensitivity.to_numpy(float)
    ax.barh(
        range(len(top)), values,
        color=[POSITIVE if value >= 0 else NEGATIVE for value in values], height=0.68,
    )
    ax.set_yticks(range(len(top)), [short_equation(x) for x in top.reaction_equation])
    ax.axvline(0, color="#444444", linewidth=0.7)
    ax.set_xlabel(r"Normalized sensitivity, $\partial\ln S_u/\partial\ln k_j$")
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
    ax.grid(False)


def main() -> None:
    polimi = pd.read_csv("results/processed/POLIMI_2023_flame_sensitivity.csv")
    konnov = pd.read_csv("results/processed/Konnov_2026_flame_sensitivity.csv")
    reactor = pd.read_csv("results/processed/reactor_robust_summary.csv")
    energy = pd.read_csv("results/processed/cracking_energy_sensitivity.csv")
    thermal = reactor[
        np.isclose(reactor.temperature_K, 750)
        & np.isclose(reactor.pressure_bar, 20)
        & np.isclose(reactor.residence_time_ms, 20)
        & np.isclose(reactor.equivalence_ratio, 1.0)
        & np.isclose(reactor.cracking_ratio, 0.3)
        & np.isclose(reactor.psr_fraction, 0.5)
    ].sort_values("heat_loss_W_per_K").copy()
    if len(thermal) != 3:
        raise ValueError(f"expected three matched thermal cases, found {len(thermal)}")
    pollutant_columns = {
        "NOx": "NOx_NO2eq_g_per_MJ_median",
        "N2O": "N2O_g_per_MJ_median",
        "NH3 slip": "NH3_slip_g_per_MJ_median",
    }
    for label, column in pollutant_columns.items():
        baseline = float(
            thermal.loc[np.isclose(thermal.heat_loss_W_per_K, 0), column].iloc[0]
        )
        thermal[f"{label}_relative_change_percent"] = 100 * (
            thermal[column] / baseline - 1
        )
    reference = float(
        energy.loc[
            np.isclose(energy.cracking_ratio, 0),
            "net_after_external_heat_J_per_mol_initial_NH3",
        ].iloc[0]
    )
    energy["net_energy_percent_reference"] = (
        100 * energy.net_after_external_heat_J_per_mol_initial_NH3 / reference
    )

    source = Path("figures/source_data")
    exports = Path("figures/exports")
    source.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    pd.concat([
        polimi.nsmallest(12, "absolute_rank"),
        konnov.nsmallest(12, "absolute_rank"),
    ]).to_csv(source / "figure7_flame_sensitivity.csv", index=False)
    thermal.to_csv(source / "figure7_thermal_sensitivity.csv", index=False)
    energy.to_csv(source / "figure7_cracking_energy.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), constrained_layout=True)
    sensitivity_panel(axes[0, 0], polimi, "a  POLIMI flame-speed sensitivity")
    sensitivity_panel(axes[0, 1], konnov, "b  Konnov flame-speed sensitivity")

    ax = axes[1, 0]
    for label, color in COLORS.items():
        ax.plot(
            thermal.heat_loss_W_per_K,
            thermal[f"{label}_relative_change_percent"],
            marker="o", linewidth=1.2, color=color, label=label,
        )
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xlabel(r"Lumped heat-loss coefficient, $UA$ (W K$^{-1}$)")
    ax.set_ylabel("Change from adiabatic case (%)")
    ax.set_title(
        "c  Emission response to bounded heat loss", loc="left",
        fontsize=7.5, fontweight="bold",
    )
    ax.legend(fontsize=5.8)

    ax = axes[1, 1]
    for recovery, group in energy.groupby("heat_recovery_fraction"):
        ax.plot(
            group.cracking_ratio, group.net_energy_percent_reference, marker="o",
            linewidth=1.2, label=f"{100 * recovery:.0f}% heat recovery",
        )
    ax.axhline(100, color="#777777", linewidth=0.7)
    ax.set_xlabel(r"Cracking ratio, $\alpha$")
    ax.set_ylabel("Fuel-side net thermochemical balance (%)")
    ax.set_title(
        "d  Cracking-heat recovery sensitivity", loc="left",
        fontsize=7.5, fontweight="bold",
    )
    ax.legend(fontsize=5.8)

    fig.suptitle(
        "Shared H/O chain branching and mechanism-specific NHx termination shape the candidate flame",
        fontsize=9, fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        r"Sensitivity flame: 300 K, 5 bar, $\phi=1.0$, $\alpha=0.4$. "
        r"Thermal panel: 750 K, 20 bar, 20 ms, $\alpha=0.3$.",
        ha="center", fontsize=6,
    )
    base = exports / "figure7_sensitivity"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
