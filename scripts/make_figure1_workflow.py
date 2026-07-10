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


def box(ax, x, y, w, h, title, body, color):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1, edgecolor=color, facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(x + 0.04 * w, y + 0.76 * h, title, color=color, fontsize=6.8,
            fontweight="bold", va="center")
    ax.text(x + 0.04 * w, y + 0.52 * h, body, color="#263238", fontsize=5.5,
            va="top", linespacing=1.35)


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
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.015, 0.212, 0.409, 0.606, 0.803]
    colors = [BLUE, BLUE, ORANGE, GREEN, GREEN]
    titles = ["1  Evidence", "2  External validation", "3  Ensemble gate",
              "4  Cross-test screen", "5  Stress tests"]
    bodies = [
        "IDT, LBV, species\nStudy-level metadata\n8–12 mechanisms",
        "Leave-one-study-out\nMatch criterion/reactor\nError, bias, failures",
        "Non-negative weights\nSeparate observables\nBeat both baselines?",
        "Supported IDT range\nSoret 1D flames\nBounded PSR–PFR",
        "Separate pollutants\nAlternative dispersion\nMechanism deletion",
    ]
    for x, title, body, color in zip(xs, titles, bodies, colors, strict=True):
        box(ax, x, 0.55, 0.17, 0.29, title, body, color)
    for i in range(4):
        ax.add_patch(FancyArrowPatch(
            (xs[i] + 0.172, 0.695), (xs[i + 1] - 0.003, 0.695),
            arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=GREY
        ))
    gate_labels = ["QA", "held-out\nresiduals", "performance\ngate", "support +\nconvergence"]
    for i, label in enumerate(gate_labels):
        ax.text((xs[i] + xs[i + 1] + 0.17) / 2, 0.525, label, ha="center",
                va="top", fontsize=5.1, color=GREY, linespacing=1.0)

    ax.text(0.025, 0.43, "Claim admission", fontsize=8, fontweight="bold", color="#263238")
    tiers = [
        (0.025, GREEN, "Supported", "Exact criterion and experimental-domain overlap"),
        (0.355, ORANGE, "Proxy-supported", "Related criterion; limitation carried into text"),
        (0.685, RED, "Exploratory", "Outside support or not invariant to stress tests"),
    ]
    for x, color, title, body in tiers:
        ax.add_patch(FancyBboxPatch(
            (x, 0.25), 0.29, 0.115, boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor=LIGHT, edgecolor=color, linewidth=1.0
        ))
        ax.text(x + 0.015, 0.325, title, fontsize=7.2, fontweight="bold", color=color)
        ax.text(x + 0.015, 0.28, body, fontsize=5.2, color="#263238")

    ax.add_patch(FancyBboxPatch(
        (0.18, 0.075), 0.64, 0.085, boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor="white", edgecolor=GREY, linewidth=0.9
    ))
    ax.text(0.5, 0.117,
            "Manuscript claim strength is limited by the weakest validation, domain and robustness test",
            ha="center", va="center", fontsize=6.7, color="#263238")
    ax.text(0.018, 0.94, "Evidence gates expose when mechanism ensembles and apparent optima are not credible",
            fontsize=9, fontweight="bold", color="#263238")

    base = exports / "figure1_workflow"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
