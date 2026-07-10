"""Supplementary Fig. S13: screening robustness and study-deletion support."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    robust = pd.read_csv(root / "results/processed/revision_pareto_robustness.csv")
    domain = pd.read_csv(root / "results/processed/lbv_candidate_domain_support.csv")
    declared = set(
        robust.loc[
            robust.mechanism_subset.eq("all")
            & robust.dispersion_variant.eq("normalized_range")
            & robust.design_id.fillna("").ne(""),
            "design_id",
        ]
    )
    retained = (
        robust[robust.design_id.fillna("").ne("")]
        .groupby(["mechanism_subset", "dispersion_variant"])
        .design_id.agg(lambda values: len(set(values) & declared))
        .rename("retained_declared_points")
        .reset_index()
    )
    matrix = retained.pivot(
        index="mechanism_subset", columns="dispersion_variant",
        values="retained_declared_points",
    )
    order = [
        "normalized_range", "iqr_relative", "max_median_ratio", "log_span",
        "gp_mean_normalized_range",
    ]
    matrix = matrix.reindex(columns=order)
    rows = ["all"] + sorted(index for index in matrix.index if index != "all")
    matrix = matrix.reindex(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1.55, 1]})
    image = axes[0].imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=3, aspect="auto")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            value = int(matrix.iloc[i, j])
            axes[0].text(j, i, str(value), ha="center", va="center",
                         color="white" if value >= 2 else "#222222", fontsize=6.5)
    axes[0].set_xticks(range(len(order)), ["normalized\nrange", "relative\nIQR", "max /\nmedian",
                                                "log\nspan", "GP mean +\nnormalized"], rotation=0)
    axes[0].set_yticks(range(len(rows)), [row.replace("drop_", "− ").replace("_", " ") for row in rows])
    axes[0].set_title("a  Declared-point membership depends on definition", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    colorbar.set_label("Declared points retained")

    labels = [
        f"{row.equivalence_ratio:.1f}, {row.cracking_ratio:.1f}"
        for row in domain.itertuples()
    ]
    x = np.arange(len(domain))
    axes[1].bar(x, domain.loso_hull_pass_fraction, color="#88CCEE", width=0.62,
                label="LOSO hull pass fraction")
    axes[1].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("LBV hull pass fraction")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel(r"$(\phi, \alpha)$ at 5 bar")
    twin = axes[1].twinx()
    twin.scatter(x, domain.neighbors_within_0_5, color="#CC6677", marker="D", s=24,
                 label="Local neighbors")
    twin.set_ylabel("Neighbors within distance 0.5")
    twin.set_ylim(0, max(domain.neighbors_within_0_5) * 1.25)
    twin.spines["top"].set_visible(False)
    axes[1].set_title("b  One study controls global hull membership", loc="left", fontweight="bold")
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="lower left", fontsize=6)

    fig.suptitle(
        "Cross-test screening is reproducible under the declared rule but not definition invariant",
        fontsize=9.5, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = root / "figures/supplementary/figureS13_revision_robustness"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    source = root / "figures/source_data"
    source.mkdir(parents=True, exist_ok=True)
    matrix.reset_index().to_csv(source / "figureS13_candidate_count_matrix.csv", index=False)
    domain.to_csv(source / "figureS13_domain_support.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
