"""Figure 3: mechanism-specific pilot errors and evidence asymmetry."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MECHANISM_ORDER = [
    "POLIMI_2023", "Konnov_2026", "NUIGMech1.1_HT", "Shrestha_2018",
    "MEI_2019", "C3_v3_5_C0", "RMG_2026_Burke", "NUIG_2024",
    "MEI_2021", "Otomo_2018", "KAUST_2023",
]
COLORS = dict(zip(MECHANISM_ORDER, mpl.colormaps["tab20"].colors))


def main() -> None:
    lbv_paths = sorted(Path("results/raw").glob("*_lbv_validation_staged.csv"))
    lbv = pd.concat([pd.read_csv(path) for path in lbv_paths if path.exists()], ignore_index=True)
    lbv = lbv[lbv.status.eq("completed")].copy()
    development_dois = {
        "10.1016/j.combustflame.2019.08.033",
        "10.1016/j.combustflame.2021.111472",
    }
    lbv_excluded = lbv[lbv.doi.isin(development_dois)].copy()
    lbv = lbv[~lbv.doi.isin(development_dois)].copy()
    campaign = (lbv.groupby(["campaign_id", "mechanism_id"], as_index=False)
                .agg(median_relative_error=("relative_error", "median"),
                     mean_relative_error=("relative_error", "mean"), n=("dataset_id", "size")))
    idt = pd.read_csv("results/processed/idt_common_predictions.csv")
    idt_mechanisms = [name for name in MECHANISM_ORDER if name in idt.columns]
    idt_long = idt.melt(
        id_vars=["dataset_id", "campaign_id", "experimental_s", "temperature_K", "pressure_Pa"],
        value_vars=idt_mechanisms,
        var_name="mechanism_id", value_name="simulated_s",
    )
    idt_long["absolute_log10_error"] = np.abs(
        np.log10(idt_long.simulated_s / idt_long.experimental_s)
    )
    jsr = pd.read_csv("results/processed/jsr_method_summary.csv")
    jsr_gate = pd.read_csv("results/processed/jsr_observable_gate.csv").set_index("observable")

    source = Path("figures/source_data")
    exports = Path("figures/exports")
    source.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    lbv.to_csv(source / "figure3_lbv_predictions.csv", index=False)
    lbv_excluded.to_csv(source / "figure3_lbv_excluded_development_predictions.csv", index=False)
    campaign.to_csv(source / "figure3_lbv_campaign_errors.csv", index=False)
    idt_long.to_csv(source / "figure3_idt_predictions.csv", index=False)
    jsr.to_csv(source / "figure3_jsr_method_summary.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)

    ax = axes[0, 0]
    for mechanism in [name for name in MECHANISM_ORDER if name in set(lbv.mechanism_id)]:
        group = lbv[lbv.mechanism_id.eq(mechanism)]
        ax.scatter(group.experimental_m_per_s, group.simulated_m_per_s, s=13, alpha=0.62,
                   color=COLORS[mechanism], edgecolors="none", label=mechanism.replace("_", " "))
    limit = 1.05 * max(lbv.experimental_m_per_s.max(), lbv.simulated_m_per_s.max())
    ax.plot([0, limit], [0, limit], "--", color="#333333", linewidth=0.8)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel(r"Measured $S_u$ (m s$^{-1}$)")
    ax.set_ylabel(r"Simulated $S_u$ (m s$^{-1}$)")
    ax.set_title("a  Externally tested LBV predictions", loc="left", fontsize=7.5,
                 fontweight="bold")
    ax.legend(fontsize=5.0, ncol=2)

    ax = axes[0, 1]
    campaigns = sorted(campaign.campaign_id.unique())
    mechanisms = sorted(campaign.mechanism_id.unique())
    heat = (campaign.pivot(index="campaign_id", columns="mechanism_id",
                           values="median_relative_error")
            .reindex(index=campaigns, columns=mechanisms).to_numpy(float))
    image = ax.imshow(heat, origin="upper", aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(len(mechanisms)), [name.replace("_", " ") for name in mechanisms], rotation=20)
    ax.set_yticks(range(len(campaigns)), [str(i) for i in range(1, len(campaigns) + 1)])
    ax.set_ylabel("Experimental study")
    ax.set_title("b  Study-median LBV error", loc="left", fontsize=7.5, fontweight="bold")
    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Median absolute relative error", fontsize=6)

    ax = axes[1, 0]
    jsr_methods = ["stacked", "equal_weight", "best_single"]
    jsr_species = sorted(jsr.observable.unique())
    jsr_heat = (jsr.pivot(index="observable", columns="method",
                          values="mean_absolute_standardized_error")
                .reindex(index=jsr_species, columns=jsr_methods).to_numpy(float))
    image = ax.imshow(np.log10(jsr_heat), origin="upper", aspect="auto", cmap="viridis")
    ax.set_xticks(range(3), ["Stacking", "Equal", "Best single"], rotation=20)
    ax.set_yticks(range(len(jsr_species)), jsr_species)
    ax.set_title("c  JSR external-validation gate", loc="left", fontsize=7.5,
                 fontweight="bold")
    for row, species in enumerate(jsr_species):
        if bool(jsr_gate.loc[species, "stacking_beats_both"]):
            ax.text(0, row, "*", ha="center", va="center", color="white", fontsize=9,
                    fontweight="bold")
    cb = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_{10}$ mean absolute standardized error", fontsize=6)

    ax = axes[1, 1]
    for mechanism in idt_mechanisms:
        group = idt_long[idt_long.mechanism_id.eq(mechanism)]
        ax.scatter(group.experimental_s, group.simulated_s, s=14, alpha=0.65,
                   color=COLORS[mechanism], edgecolors="none", label=mechanism.replace("_", " "))
    low = min(idt_long.experimental_s.min(), idt_long.simulated_s.min())
    high = max(idt_long.experimental_s.max(), idt_long.simulated_s.max())
    ax.plot([low, high], [low, high], "--", color="#333333", linewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Measured IDT (s)"); ax.set_ylabel("Simulated IDT (s)")
    ax.set_title("d  Criterion-matched IDT (46 points; one study)", loc="left",
                 fontsize=7.5, fontweight="bold")
    ax.legend(fontsize=4.6, ncol=2)

    fig.suptitle("Mechanism errors remain observable dependent and ignition evidence is study limited",
                 fontsize=9, fontweight="bold")
    base = exports / "figure3_mechanism_errors"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
