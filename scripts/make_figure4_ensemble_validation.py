"""Figure 4: grouped external LBV comparison and stacking-gate audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["stacked", "equal_weight", "best_single"]
LABELS = {"stacked": "Stacking", "equal_weight": "Equal weight", "best_single": "Training-best single"}
COLORS = {"stacked": "#4477AA", "equal_weight": "#999999", "best_single": "#EE9944"}


def main() -> None:
    predictions = pd.read_csv("results/processed/lbv_loco_predictions.csv")
    summary = pd.read_csv("results/processed/lbv_method_summary.csv")
    weights = pd.read_csv("results/processed/lbv_fold_weights.csv")
    excluded_path = Path("results/processed/lbv_excluded_development_campaigns.csv")
    source_dir = Path("figures/source_data")
    export_dir = Path("figures/exports")
    source_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(source_dir / "figure4_loco_predictions.csv", index=False)
    summary.to_csv(source_dir / "figure4_method_summary.csv", index=False)
    weights.to_csv(source_dir / "figure4_fold_weights.csv", index=False)
    if excluded_path.exists():
        pd.read_csv(excluded_path).to_csv(
            source_dir / "figure4_excluded_development_campaigns.csv", index=False
        )

    campaign = (predictions.groupby(["held_out_campaign", "doi"])
                .agg(**{method: (f"relative_error_{method}", "mean") for method in METHODS})
                .reset_index().sort_values("stacked", ascending=False).reset_index(drop=True))
    campaign.insert(0, "campaign_number", np.arange(1, len(campaign) + 1))
    campaign.to_csv(source_dir / "figure4_campaign_errors.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.7, "legend.frameon": False,
    })
    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], height_ratios=[1.05, 0.95])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    low = predictions.residual_interval_low
    high = predictions.residual_interval_high
    center = predictions.stacked
    x = predictions.observed
    ax_a.vlines(x, low, high, color="#A9BCD0", alpha=0.30, linewidth=0.6)
    ax_a.scatter(x, center, s=12, color=COLORS["stacked"], alpha=0.75,
                 edgecolors="white", linewidths=0.2)
    limit = max(float(x.max()), float(high.max())) * 1.04
    ax_a.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=0.8)
    ax_a.set_xlim(0, limit)
    ax_a.set_ylim(0, limit)
    ax_a.set_xlabel(r"Measured $S_\mathrm{u}$ (m s$^{-1}$)")
    ax_a.set_ylabel(r"Held-study stacking $S_\mathrm{u}$ (m s$^{-1}$)")
    coverage = predictions.residual_interval_covered.mean()
    ax_a.text(0.04, 0.94, f"90% residual interval coverage: {coverage:.1%}",
              transform=ax_a.transAxes, va="top")

    ordered = summary.set_index("method").loc[METHODS]
    means = ordered.mean_absolute_relative_error.to_numpy()
    lower = means - ordered.cluster_bootstrap_95_low.to_numpy()
    upper = ordered.cluster_bootstrap_95_high.to_numpy() - means
    bars = ax_b.bar(np.arange(3), means, color=[COLORS[item] for item in METHODS], width=0.65)
    ax_b.errorbar(np.arange(3), means, yerr=np.vstack([lower, upper]), fmt="none",
                  ecolor="black", elinewidth=0.8, capsize=2.5)
    ax_b.set_xticks(np.arange(3), ["Stacking", "Equal\nweight", "Training-best\nsingle"], rotation=0)
    ax_b.set_ylabel("Mean absolute relative error")
    ax_b.set_ylim(0, ordered.cluster_bootstrap_95_high.max() * 1.18)
    for bar, value in zip(bars, means, strict=True):
        ax_b.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                  ha="center", va="bottom", fontsize=6)

    for method in METHODS:
        ax_c.plot(campaign.campaign_number, campaign[method], marker="o", markersize=3,
                  linewidth=1.0, color=COLORS[method], label=LABELS[method])
    ax_c.set_xlabel("Held-out experimental study")
    ax_c.set_ylabel("Mean absolute relative error")
    ax_c.set_xticks(campaign.campaign_number)
    ax_c.set_xticklabels(campaign.campaign_number, fontsize=5.5)
    ax_c.legend(ncol=3, loc="upper right", fontsize=5.7)

    weight_columns = [column for column in weights if column.startswith("weight_")]
    weight_matrix = weights[weight_columns].to_numpy()
    image = ax_d.imshow(weight_matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    names = [column.removeprefix("weight_").replace("_", " ") for column in weight_columns]
    ax_d.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    ax_d.set_yticks(np.arange(len(weights)), np.arange(1, len(weights) + 1), fontsize=5.5)
    ax_d.set_ylabel("Held-out study")
    ax_d.set_title("Outer-fold stacking weights", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.05, pad=0.03)
    colorbar.set_label("Weight")

    for label, axis in zip("abcd", [ax_a, ax_b, ax_c, ax_d], strict=True):
        axis.text(-0.16, 1.07, label, transform=axis.transAxes, fontsize=9,
                  fontweight="bold", va="top")
        axis.grid(False)
    base = export_dir / "figure4_ensemble_validation"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
