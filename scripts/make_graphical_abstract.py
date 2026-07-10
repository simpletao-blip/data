"""Evidence-compatible graphical abstract for the current working manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


def main() -> None:
    method = pd.read_csv("results/processed/lbv_method_summary.csv").set_index("method")
    stacking_mare = 100 * method.loc["stacked", "mean_absolute_relative_error"]
    best_single_mare = 100 * method.loc["best_single", "mean_absolute_relative_error"]
    robust = pd.read_csv("results/processed/full_proxy_robust_summary.csv")
    proxy_count = int((robust.screening_pareto & robust.support_tier.eq("proxy_supported")).sum())
    strict_count = int((robust.screening_pareto & robust.support_tier.eq(
        "strict_criterion_supported"
    )).sum())
    exploratory_count = int(robust.exploratory_extrapolative_pareto.sum())
    revision = pd.read_csv("results/processed/revision_pareto_robustness.csv")
    revision = revision[revision.design_id.fillna("").ne("")]
    complete = revision[revision.mechanism_subset.eq("all")]
    complete_sets = [set(group.design_id) for _, group in complete.groupby("dispersion_variant")]
    metric_invariant_count = len(set.intersection(*complete_sets))
    all_sets = [
        set(group.design_id)
        for _, group in revision.groupby(["mechanism_subset", "dispersion_variant"])
    ]
    all_stress_invariant_count = len(set.intersection(*all_sets))
    summary = pd.DataFrame([
        ("public_records", 5011), ("campaigns", 38),
        ("stacking_MARE_percent", stacking_mare),
        ("best_single_MARE_percent", best_single_mare),
        ("screened_conditions", 432), ("strict_supported_pareto", strict_count),
        ("proxy_supported_pareto", proxy_count),
        ("dispersion_metric_invariant_points", metric_invariant_count),
        ("all_stress_invariant_points", all_stress_invariant_count),
        ("exploratory_extrapolative_pareto", exploratory_count),
    ], columns=["metric", "value"])
    source = Path("figures/source_data")
    exports = Path("submission")
    source.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source / "graphical_abstract_summary.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panels = [(0.03, "Public evidence", "#4477AA"), (0.36, "External ensemble gate", "#EE9944"),
              (0.69, "Robustness stress test", "#228833")]
    for x, title, color in panels:
        ax.add_patch(FancyBboxPatch((x, 0.25), 0.27, 0.59,
                    boxstyle="round,pad=0.012,rounding_size=0.025",
                    edgecolor=color, facecolor="white", linewidth=1.5))
        ax.text(x + 0.135, 0.76, title, ha="center", fontsize=8, fontweight="bold", color=color)
    for x0, x1 in ((0.30, 0.36), (0.63, 0.69)):
        ax.add_patch(FancyArrowPatch((x0, 0.55), (x1, 0.55), arrowstyle="-|>",
                                    mutation_scale=12, linewidth=1.2, color="#68757D"))

    ax.text(0.165, 0.61, "5,011", ha="center", fontsize=22, fontweight="bold", color="#263238")
    ax.text(0.165, 0.52, "records", ha="center", fontsize=8, color="#263238")
    ax.text(0.165, 0.40, "38 complete campaigns", ha="center", fontsize=7, color="#263238")
    ax.text(0.165, 0.30, "IDT  |  LBV  |  species", ha="center", fontsize=6.5, color="#68757D")
    ax.text(0.33, 0.63, "whole-study\nholdout", ha="center", va="center", fontsize=5.8, color="#68757D")

    ax.text(0.495, 0.62, f"{stacking_mare:.3f}%", ha="center", fontsize=15, fontweight="bold", color="#4477AA")
    ax.text(0.495, 0.55, "stacking MARE", ha="center", fontsize=6.5)
    ax.text(0.495, 0.45, f"{best_single_mare:.3f}%", ha="center", fontsize=15, fontweight="bold", color="#EE9944")
    ax.text(0.495, 0.38, "training-best single", ha="center", fontsize=6.5)
    ax.text(0.495, 0.29, "GATE NOT PASSED", ha="center", fontsize=7.3, fontweight="bold", color="#CC6677")
    ax.text(0.66, 0.63, "propagate\nuncertainty", ha="center", va="center", fontsize=5.8, color="#68757D")

    ax.text(0.825, 0.64, "432 screened", ha="center", fontsize=11, fontweight="bold", color="#263238")
    ax.text(0.735, 0.48, str(proxy_count), ha="center", fontsize=18, fontweight="bold", color="#228833")
    ax.text(0.735, 0.38, "declared\nrule", ha="center", va="center", fontsize=5.8)
    ax.text(0.825, 0.48, str(metric_invariant_count), ha="center", fontsize=18, fontweight="bold", color="#4477AA")
    ax.text(0.825, 0.38, "all full-set\nmetrics", ha="center", va="center", fontsize=5.8,
            linespacing=0.95)
    ax.text(0.915, 0.48, str(all_stress_invariant_count), ha="center", fontsize=18, fontweight="bold", color="#CC6677")
    ax.text(0.915, 0.38, "all metric ×\nmechanism tests", ha="center", va="center", fontsize=5.8,
            linespacing=0.95)
    ax.text(0.825, 0.29, f"{exploratory_count} extrapolative points kept separate",
            ha="center", fontsize=5.8, color="#68757D")

    ax.text(0.5, 0.10,
            "Study-held-out validation and stress tests expose false robustness in apparent optima",
            ha="center", fontsize=8.2, fontweight="bold", color="#263238")
    base = exports / "graphical_abstract_working"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
