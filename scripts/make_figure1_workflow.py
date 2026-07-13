"""Figure 1: evidence-gated multi-mechanism workflow.

Core conclusion: ensemble and screening claims are admitted only to the tier
that survives study-held-out validation, baseline comparison, domain
reconstruction and robustness stress tests.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


BLUE = "#4477AA"
ORANGE = "#EE9944"
GREEN = "#228833"
RED = "#CC6677"
GREY = "#66727A"
LIGHT = "#F4F6F7"
TEXT = "#263238"


def box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.055 * w, y + 0.73 * h, title,
        color=color, fontsize=8.0, fontweight="bold", va="center",
    )
    ax.text(
        x + 0.055 * w, y + 0.49 * h, body,
        color=TEXT, fontsize=6.5, va="top", linespacing=1.30,
    )


def main() -> None:
    nodes = pd.DataFrame([
        (1, "Evidence", "Curated experiments and immutable mechanisms"),
        (2, "External validation", "Whole-campaign outer folds"),
        (3, "Ensemble gate", "Observable-specific non-negative stacking"),
        (4, "Cross-test screen", "Separate 0D, 1D and PSR-PFR tests"),
        (5, "Stress tests", "Domain, metric and mechanism deletion"),
    ], columns=["stage", "title", "description"])
    edges = pd.DataFrame([
        (1, 2, "criterion + mechanism QA"),
        (2, 3, "external residuals"),
        (3, 4, "only if stacking beats baselines"),
        (4, 5, "domain tags + solver status"),
    ], columns=["from_stage", "to_stage", "gate"])
    source = Path("figures/source_data")
    exports = Path("figures/exports")
    source.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(source / "figure1_workflow_nodes.csv", index=False)
    edges.to_csv(source / "figure1_workflow_edges.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    })
    # The two-row serpentine layout remains readable at final manuscript width.
    fig, ax = plt.subplots(figsize=(7.2, 4.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.025, 0.68, BLUE, "1  Evidence",
         "IDT, LBV and species\nStudy-level metadata\n8–12 mechanisms"),
        (0.36, 0.68, BLUE, "2  External validation",
         "Leave one study out\nMatch criterion/reactor\nError, bias and failures"),
        (0.695, 0.68, ORANGE, "3  Ensemble gate",
         "Non-negative weights\nSeparate observables\nBeat both baselines?"),
        (0.695, 0.43, GREEN, "4  Cross-test screen",
         "Supported IDT range\nSoret 1D flames\nBounded PSR–PFR"),
        (0.36, 0.43, GREEN, "5  Stress tests",
         "Separate pollutants\nAlternative dispersion\nMechanism deletion"),
    ]
    for x, y, color, title, body in stages:
        box(ax, x, y, 0.28, 0.17, title, body, color)

    arrows = [
        ((0.307, 0.765), (0.357, 0.765)),
        ((0.642, 0.765), (0.692, 0.765)),
        ((0.835, 0.675), (0.835, 0.605)),
        ((0.692, 0.515), (0.642, 0.515)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            linewidth=1.0, color=GREY,
        ))

    ax.add_patch(FancyArrowPatch(
        (0.50, 0.43), (0.50, 0.365), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.0, color=GREY,
    ))
    ax.text(0.025, 0.355, "Claim admission", fontsize=8.2,
            fontweight="bold", color=TEXT, va="center")

    tiers = [
        (0.025, GREEN, "Supported",
         "Exact criterion and\nexperimental-domain overlap"),
        (0.355, ORANGE, "Proxy-supported",
         "Related criterion; limitation\ncarried into text"),
        (0.685, RED, "Exploratory",
         "Outside support or not invariant\nto stress tests"),
    ]
    for x, color, title, body in tiers:
        ax.add_patch(FancyBboxPatch(
            (x, 0.185), 0.29, 0.13,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor=LIGHT, edgecolor=color, linewidth=1.0,
        ))
        ax.text(x + 0.015, 0.265, title, fontsize=7.4,
                fontweight="bold", color=color)
        ax.text(x + 0.015, 0.222, body, fontsize=5.5,
                color=TEXT, va="top", linespacing=1.20)

    ax.plot([0.18, 0.82], [0.13, 0.13], color=GREY, linewidth=0.8)
    ax.text(
        0.5, 0.09,
        "Claim strength is limited by the weakest validation, domain or robustness test",
        ha="center", va="center", fontsize=7.2, color=TEXT,
    )
    ax.text(0.025, 0.94, "Evidence-gated workflow for claim admission",
            fontsize=10, fontweight="bold", color=TEXT)

    base = exports / "figure1_workflow"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
