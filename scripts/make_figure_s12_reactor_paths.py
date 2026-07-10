"""Supplementary Figure S12: cross-mechanism N2O and NH3 reaction contributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MECHANISMS = [
    "POLIMI_2023", "Konnov_2026", "MEI_2019", "C3_v3_5_C0",
    "Otomo_2018", "NUIG_2024", "MEI_2021", "RMG_2026_Burke",
]


def contribution_matrix(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = frame[
        frame.location.eq("pfr_outlet")
        & frame.target_species.eq(target)
        & frame.absolute_rank.le(3)
    ].copy()
    subset["signed_normalized_contribution"] = np.where(
        subset.direction.eq("production"),
        subset.normalized_absolute_contribution,
        -subset.normalized_absolute_contribution,
    )
    reactions = subset.groupby("reaction_equation").normalized_absolute_contribution.max().sort_values(
        ascending=False
    ).index
    matrix = subset.pivot_table(
        index="reaction_equation", columns="mechanism_id",
        values="signed_normalized_contribution", fill_value=0.0,
    ).reindex(index=reactions, columns=MECHANISMS, fill_value=0.0)
    return matrix, subset


def heatmap(ax, matrix: pd.DataFrame, title: str):
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.9, vmax=0.9)
    ax.set_xticks(range(len(matrix.columns)), [x.replace("_", "-") for x in matrix.columns],
                  rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            if abs(value) >= 0.05:
                ax.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(value) > 0.48 else "black")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Signed fraction of total absolute contribution", fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5)


def main() -> None:
    data = pd.read_csv("results/processed/reactor_path_analysis/species_reaction_contributions.csv")
    n2o, n2o_source = contribution_matrix(data, "N2O")
    nh3, nh3_source = contribution_matrix(data, "NH3")
    source_dir = Path("figures/source_data")
    export_dir = Path("figures/supplementary")
    source_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([n2o_source, nh3_source], ignore_index=True).to_csv(
        source_dir / "figureS12_reactor_path_contributions.csv", index=False
    )
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.2,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.3), constrained_layout=True)
    heatmap(axes[0], n2o, "a  Dominant net N$_2$O reaction contributions at the PFR outlet")
    heatmap(axes[1], nh3, "b  Dominant net NH$_3$ reaction contributions at the PFR outlet")
    fig.suptitle(
        "Mechanisms share major nitrogen pathways but assign different net contributions",
        fontsize=9, fontweight="bold",
    )
    fig.text(
        0.5, -0.01,
        r"Instantaneous rates at 600 K inlet, 5 bar, 10 ms, $\phi=1.0$, $\alpha=0.4$; "
        "blue is net consumption and red is net production.",
        ha="center", fontsize=6,
    )
    base = export_dir / "figureS12_reactor_paths"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
