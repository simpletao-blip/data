"""Figure 5: paired reactivity proxies and model-form disagreement."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def matrix(frame: pd.DataFrame, column: str, phis: list[float], alphas: list[float]) -> np.ndarray:
    return (frame.pivot(index="equivalence_ratio", columns="cracking_ratio", values=column)
            .reindex(index=phis, columns=alphas).to_numpy(float))


def panel(ax, values, title, label, cmap, phis, alphas, supported, vmin=None, vmax=None):
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(len(alphas)), [f"{x:g}" for x in alphas], rotation=45, ha="right")
    ax.set_yticks(range(len(phis)), [f"{x:g}" for x in phis])
    ax.set_xlabel(r"Cracking ratio, $\alpha$")
    ax.set_ylabel(r"Equivalence ratio, $\phi$")
    for row in supported.itertuples(index=False):
        ax.scatter(alphas.index(row.cracking_ratio), phis.index(row.equivalence_ratio),
                   s=34, facecolors="none", edgecolors="white", linewidths=0.9)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(label, fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5)


def main() -> None:
    data = pd.read_csv("results/processed/full_proxy_robust_summary.csv")
    selected = data[(data.temperature_K == 600) & (data.flame_temperature_K == 300)
                    & (data.pressure_bar == 5)].copy()
    phis = sorted(selected.equivalence_ratio.unique())
    alphas = sorted(selected.cracking_ratio.unique())
    supported = selected[selected.support_tier.eq("proxy_supported")]
    selected["idt_shortening_orders"] = -np.log10(selected.idt_shortening_ratio_worst)
    selected["lbv_model_range_percent"] = 100 * selected.lbv_conservative_m_per_s_relative_range
    selected["idt_model_range_percent"] = 100 * selected.ignition_delay_s_relative_range

    source = Path("figures/source_data")
    exports = Path("figures/exports")
    source.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    selected.to_csv(source / "figure5_reactivity_maps.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    panel(axes[0, 0], matrix(selected, "lbv_enhancement_ratio_worst", phis, alphas),
          "a  Conservative LBV enhancement", "Worst-model ratio to pure NH$_3$", "viridis",
          phis, alphas, supported, vmin=1)
    panel(axes[0, 1], matrix(selected, "idt_shortening_orders", phis, alphas),
          "b  Conservative IDT shortening", r"$-\log_{10}(\tau/\tau_{NH_3})$", "viridis",
          phis, alphas, supported, vmin=0)
    panel(axes[1, 0], matrix(selected, "lbv_model_range_percent", phis, alphas),
          "c  LBV model-form spread", "Inter-mechanism relative range (%)", "magma",
          phis, alphas, supported, vmin=0)
    panel(axes[1, 1], matrix(selected, "idt_model_range_percent", phis, alphas),
          "d  IDT model-form spread", "Inter-mechanism relative range (%)", "magma",
          phis, alphas, supported, vmin=0)
    fig.suptitle("Cracking improves both reactivity proxies, but support and model spread remain condition dependent",
                 fontsize=9, fontweight="bold")
    fig.text(0.5, -0.015,
             "Paired reference slice: LBV 300 K; IDT 1200 K; reactor inlet 600 K; 5 bar. White rings: joint proxy support.",
             ha="center", fontsize=6)
    base = exports / "figure5_reactivity_maps"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
