"""Figure 6: separate reactive-nitrogen objectives and Pareto support status."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def matrix(frame, column, phis, alphas):
    return (frame.pivot(index="equivalence_ratio", columns="cracking_ratio", values=column)
            .reindex(index=phis, columns=alphas).to_numpy(float))


def heat(ax, values, title, label, phis, alphas, log=False):
    shown = np.log10(np.maximum(values, 1e-12)) if log else values
    image = ax.imshow(shown, origin="lower", aspect="auto", cmap="cividis")
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(len(alphas)), [f"{x:g}" for x in alphas], rotation=45, ha="right")
    ax.set_yticks(range(len(phis)), [f"{x:g}" for x in phis])
    ax.set_xlabel(r"Cracking ratio, $\alpha$")
    ax.set_ylabel(r"Equivalence ratio, $\phi$")
    cb = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(("log$_{10}$ " if log else "") + label, fontsize=6)
    cb.ax.tick_params(labelsize=5.5)


def main() -> None:
    data = pd.read_csv("results/processed/full_proxy_robust_summary.csv")
    selected = data[(data.temperature_K == 600) & (data.flame_temperature_K == 300)
                    & (data.pressure_bar == 5)].copy()
    phis = sorted(selected.equivalence_ratio.unique())
    alphas = sorted(selected.cracking_ratio.unique())
    source = Path("figures/source_data")
    exports = Path("figures/exports")
    source.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    selected.to_csv(source / "figure6_emission_slice.csv", index=False)
    data[data.screening_pareto].to_csv(source / "figure6_pareto_candidates.csv", index=False)
    data[data.exploratory_extrapolative_pareto].to_csv(
        source / "figure6_exploratory_pareto.csv", index=False
    )

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
        "axes.linewidth": 0.7, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    heat(axes[0, 0], matrix(selected, "EI_g_per_MJ_NOx_NO2eq_worst", phis, alphas),
         "a  Worst-model NOx", "g NO$_2$-eq MJ$^{-1}$", phis, alphas)
    heat(axes[0, 1], matrix(selected, "EI_g_per_MJ_N2O_worst", phis, alphas),
         "b  Worst-model N$_2$O", "g MJ$^{-1}$", phis, alphas, log=True)
    heat(axes[1, 0], matrix(selected, "EI_g_per_MJ_NH3_slip_worst", phis, alphas),
         "c  Worst-model NH$_3$ slip", "g MJ$^{-1}$", phis, alphas, log=True)

    ax = axes[1, 1]
    ax.scatter(selected.cracking_ratio, selected.equivalence_ratio, s=14,
               color="#C9CED1", label="Screened condition", zorder=1)
    exploratory = selected[selected.exploratory_extrapolative_pareto]
    supported = selected[
        selected.screening_pareto
        & selected.support_tier.isin(["proxy_supported", "strict_criterion_supported"])
    ]
    thermo_limited = selected[~selected.complete_mechanism_set]
    ax.scatter(exploratory.cracking_ratio, exploratory.equivalence_ratio, s=30, marker="x",
               color="#CC6677", linewidths=1.1, label="Extrapolative front", zorder=2)
    ax.scatter(supported.cracking_ratio, supported.equivalence_ratio, s=65, marker="*",
               color="#228833", edgecolor="white", linewidth=0.6,
               label="Declared cross-test points", zorder=3)
    ax.scatter(thermo_limited.cracking_ratio, thermo_limited.equivalence_ratio, s=32, marker="s",
               facecolor="none", edgecolor="#AA4499", linewidth=0.9,
               label="Incomplete thermo domain", zorder=4)
    ax.set_title("d  Screening status is evidence limited", loc="left", fontsize=7.5, fontweight="bold")
    ax.set_xlabel(r"Cracking ratio, $\alpha$")
    ax.set_ylabel(r"Equivalence ratio, $\phi$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.6)
    ax.grid(False)
    fig.suptitle("Reactive-nitrogen objectives remain separate; cross-test points are not operating states",
                 fontsize=9, fontweight="bold")
    fig.text(0.5, -0.015,
             "Paired reference slice: reactor 600 K, 10 ms, adiabatic; LBV 300 K; IDT 1200 K; 5 bar.",
             ha="center", fontsize=6)
    base = exports / "figure6_emissions_pareto"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
