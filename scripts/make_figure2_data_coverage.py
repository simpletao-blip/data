"""Figure 2: public evidence coverage and validation-admission gate.

Core conclusion: the public evidence base is broad in record count but strongly
heterogeneous in operating coverage and criterion match, requiring campaign-
level curation before mechanism weighting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"LBV": "#4477AA", "IDT": "#EE9944", "Species": "#6C8E78"}
LABELS = {"LBV": "Laminar burning velocity", "IDT": "Ignition delay", "Species": "Species profiles"}


def measurement_class(experiment_type: str) -> str:
    if experiment_type == "laminar burning velocity measurement":
        return "LBV"
    if experiment_type == "ignition delay measurement":
        return "IDT"
    return "Species"


def main() -> None:
    data = pd.read_csv("data/processed/respecth_nh3_long.csv")
    registry = pd.read_csv("data/processed/respecth_campaign_registry.csv")
    data["measurement_class"] = data.experiment_type.map(measurement_class)
    registry["measurement_class"] = registry.experiment_type.map(measurement_class)
    conditions = data.drop_duplicates([
        "measurement_class", "campaign_id", "source_file", "temperature_K", "pressure_Pa",
        "equivalence_ratio", "cracking_ratio", "initial_composition",
    ]).copy()
    conditions["pressure_bar"] = conditions.pressure_Pa / 1e5
    summary = (data.groupby("measurement_class")
               .agg(records=("dataset_id", "size"), conditions=("source_file", "size"),
                    campaigns=("campaign_id", "nunique"), files=("source_file", "nunique"))
               .reset_index())
    summary["conditions"] = summary.measurement_class.map(
        conditions.groupby("measurement_class").size()
    )
    gate = (registry.groupby(["measurement_class", "preliminary_eligibility"], dropna=False)
            .rows.sum().unstack(fill_value=0)
            .reindex(["LBV", "IDT", "Species"]).fillna(0))

    source_dir = Path("figures/source_data")
    export_dir = Path("figures/exports")
    source_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(source_dir / "figure2_coverage_points.csv", index=False)
    summary.to_csv(source_dir / "figure2_coverage_summary.csv", index=False)
    gate.reset_index().to_csv(source_dir / "figure2_admission_gate.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False,
    })
    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    for category in ["Species", "LBV", "IDT"]:
        subset = conditions[conditions.measurement_class.eq(category)]
        ax_a.scatter(subset.temperature_K, subset.pressure_bar, s=10, alpha=0.45,
                     color=COLORS[category], edgecolors="none", label=LABELS[category])
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Temperature (K)")
    ax_a.set_ylabel("Pressure (bar)")
    ax_a.legend(loc="upper right", fontsize=6, markerscale=1.2)

    for category in ["Species", "LBV", "IDT"]:
        subset = conditions[conditions.measurement_class.eq(category)].dropna(
            subset=["equivalence_ratio", "cracking_ratio"]
        )
        ax_b.scatter(subset.cracking_ratio, subset.equivalence_ratio, s=10, alpha=0.40,
                     color=COLORS[category], edgecolors="none")
    ax_b.set_xlabel(r"Equivalent cracking ratio, $\alpha$")
    ax_b.set_ylabel(r"Equivalence ratio, $\phi$")
    ax_b.set_xlim(-0.03, 1.03)

    order = ["LBV", "IDT", "Species"]
    y = np.arange(len(order))
    ordered = summary.set_index("measurement_class").loc[order]
    ax_c.barh(y - 0.16, ordered.conditions, height=0.30,
              color=[COLORS[item] for item in order], alpha=0.9, label="Unique conditions")
    ax_c.barh(y + 0.16, ordered.campaigns, height=0.30,
              color="white", edgecolor=[COLORS[item] for item in order], hatch="///",
              linewidth=0.8, label="Campaigns")
    ax_c.set_yticks(y, [LABELS[item] for item in order])
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Count (log scale)")
    ax_c.legend(loc="upper right", fontsize=6)

    gate = gate.loc[order]
    eligibility_order = ["exact_candidate", "pending_manual_review", "unsupported_exact"]
    eligibility_colors = ["#4477AA", "#BBBBBB", "#CC6677"]
    left = np.zeros(len(order))
    for status, color in zip(eligibility_order, eligibility_colors, strict=True):
        values = gate[status].to_numpy() if status in gate else np.zeros(len(order))
        ax_d.barh(y, values, left=left, color=color, height=0.55,
                  label=status.replace("_", " "))
        left += values
    ax_d.set_yticks(y, [LABELS[item] for item in order])
    ax_d.set_xlabel("Records at preliminary admission gate")
    ax_d.legend(loc="center right", fontsize=5.8)

    for label, axis in zip("abcd", [ax_a, ax_b, ax_c, ax_d], strict=True):
        axis.text(-0.16, 1.06, label, transform=axis.transAxes, fontsize=9,
                  fontweight="bold", va="top")
        axis.grid(False)

    base = export_dir / "figure2_data_coverage"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
