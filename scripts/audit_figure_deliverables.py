"""Check that every main figure has exports, Source Data and documentation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    exports = Path("figures/exports")
    source = Path("figures/source_data")
    rows = []
    stems = {
        1: "figure1_workflow", 2: "figure2_data_coverage", 3: "figure3_mechanism_errors",
        4: "figure4_ensemble_validation", 5: "figure5_reactivity_maps",
        6: "figure6_emissions_pareto", 7: "figure7_sensitivity",
    }
    for number, stem in stems.items():
        source_files = list(source.glob(f"figure{number}_*.csv"))
        for suffix in ("svg", "pdf", "tiff", "png"):
            path = exports / f"{stem}.{suffix}"
            rows.append({"figure": number, "artifact": suffix, "path": str(path),
                         "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
        for kind in ("contract", "legend", "qa"):
            path = Path("figures") / f"figure{number}_{kind}.md"
            rows.append({"figure": number, "artifact": kind, "path": str(path),
                         "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
        rows.append({"figure": number, "artifact": "source_data", "path": str(source),
                     "exists": bool(source_files), "bytes": sum(p.stat().st_size for p in source_files)})
    supplementary_stems = {
        1: "figureS1_literature_evidence", 2: "figureS2_data_coverage",
        3: "figureS3_idt_criterion", 4: "figureS4_mechanism_sizes",
        5: "figureS5_lbv_campaign_residuals", 6: "figureS6_idt_residual_structure",
        7: "figureS7_grid_independence", 8: "figureS8_surrogate_calibration",
        9: "figureS9_psr_fraction_sensitivity", 10: "figureS10_heat_loss_sensitivity",
        11: "figureS11_pareto_support", 12: "figureS12_reactor_paths",
        13: "figureS13_revision_robustness",
    }
    supplementary = Path("figures/supplementary")
    for number, stem in supplementary_stems.items():
        label = f"S{number}"
        for suffix in ("svg", "pdf", "tiff", "png"):
            path = supplementary / f"{stem}.{suffix}"
            rows.append({"figure": label, "artifact": suffix, "path": str(path),
                         "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
        contract = Path("figures") / f"figureS{number}_contract.md"
        rows.append({"figure": label, "artifact": "contract", "path": str(contract),
                     "exists": contract.exists(), "bytes": contract.stat().st_size if contract.exists() else 0})
        source_files = list(source.glob(f"figureS{number}_*.csv"))
        rows.append({"figure": label, "artifact": "source_data", "path": str(source),
                     "exists": bool(source_files), "bytes": sum(p.stat().st_size for p in source_files)})
    audit = pd.DataFrame(rows)
    output = Path("results/logs/figure_deliverable_audit.csv")
    audit.to_csv(output, index=False)
    print(audit.groupby(["figure", "exists"]).size().to_string())
    if not audit.exists.all() or (audit.bytes <= 0).any():
        raise SystemExit("figure deliverable audit failed")
    print(output)


if __name__ == "__main__":
    main()
