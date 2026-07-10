# Partially Cracked Ammonia Multi-Mechanism Ensemble

Reproducible Python/Cantera pipeline for cross-validated kinetic-mechanism
ensembles and combustion-side Pareto maps for partially cracked ammonia.

## Scientific status

This repository is a pre-submission research evidence package. The principal
Cantera calculations, complete-study cross-validation, cross-test screening, figures
and manuscript draft are complete. Submission still requires author metadata,
persistent data/code identifiers and mechanism-by-mechanism redistribution
checks. Synthetic or smoke-test outputs must never be reported as experimental
or publication results.

## Environment

```powershell
conda activate cantera
python -m pip install -e .
pytest -q
python scripts/run_smoke.py
python scripts/parse_respecth.py data/raw/respecth_nh3_v2_3/extracted
python scripts/audit_respecth.py data/processed/respecth_nh3_long.csv
python scripts/select_lbv_design.py data/processed/respecth_nh3_long.csv data/processed/lbv_validation_design.csv
python scripts/build_operating_design.py
python scripts/build_idt_operating_design.py
python scripts/build_proxy_screening_design.py
python scripts/make_figure1_workflow.py
python scripts/assemble_manuscript.py
python scripts/audit_manuscript_numbers.py
python scripts/audit_figure_deliverables.py
python scripts/audit_supplementary_references.py
python scripts/extract_respecth_bibliography.py
python scripts/build_manuscript_references.py
python scripts/build_repository_manifest.py
```

The research environment uses Python 3.12 and Cantera 3.2.0. Recreate it with
`environment.yml` when needed.

## Reproducible workflow

1. Register each source in the evidence ledger and each mechanism in
   `mechanisms/registry.csv`.
2. Retain immutable source artifacts under `data/raw/` or `mechanisms/raw/`.
3. Transform records into the canonical schema documented in `data/README.md`.
4. Run mechanism QA before simulation.
5. Generate predictions with explicit reactor, criterion, transport and
   convergence metadata.
6. Fit observable-specific ensemble weights only with complete-study grouped
   cross-validation.
7. Export every figure with matching Source Data.

Long-running 1D scripts checkpoint after every case. Failed, timed-out or
unsupported cases remain in the output ledger rather than disappearing from
the batch.

## Current evidence gates

- ReSpecTh NH3 v2.3 contains 306 source XML files and 5,011 normalized records.
- All eleven admitted mechanisms completed 46 criterion-matched IDT points from
  one experimental study. This supports comparison, not external IDT weighting.
- Seven mechanisms completed all 80 staged multicomponent/Soret LBV flames;
  RMG-Burke completed 78/80 and retained two audited timeouts.
- The primary LBV gate excludes the complete MEI 2019 and Mei 2021 development
  studies, leaving 68 common points from 14 held-out studies. In the final
  eight-mechanism result, stacking improves on equal weighting but not on the
  training-best single mechanism. No family-removal ablation passes both gates.
- JSR validation contains 48 conditions, 161 observations and six studies
  evaluated with eleven mechanisms. No species passes both paired-bootstrap
  baselines (0/8).
- Shrestha completed only 3/21 LBV feasibility cases within 300 s. KAUST did
  not finish its first staged flame within 900.3 s and used approximately
  0.58 GB. Both remain admitted for supported lower-dimensional observables but
  are excluded from formal LBV stacking.
- UCSD 2018 is retained as a screened baseline after 22/80 LBV convergence,
  0.842 mean absolute log10 IDT error and missing-He JSR coverage.
- Conditions outside an observable's experimental convex hull remain explicit
  extrapolations and are excluded from a trusted operating window.
- No condition lies in the intersection of the exact-criterion IDT and LBV
  support domains; no strictly validated Pareto window is currently admitted.
- Seven mechanisms completed all 144 direct LBV operating-map flames and
  RMG-Burke completed 142/144. All eight inside-hull interpolation models passed
  the numerical holdout gate.
- The declared eight-mechanism screen retained three cross-test fuel-composition points;
  all 24 candidate-mechanism direct flames fell inside their nominal surrogate
  intervals, with a maximum interpolation error of 2.02%. Only the
  (phi, alpha) = (0.9, 0.4) candidate persisted across all tested 50th-100th
  percentile dispersion gates.
- All seven main figures have SVG, PDF, 600 dpi TIFF, PNG review files, Source
  Data, contracts and QA records; Supplementary Figs. S1–S13 add eight-mechanism
  nitrogen-path contribution evidence.
- The manuscript now contains 35 DOI-linked references, including every formal
  LBV, exact-criterion IDT and JSR source used in validation. Local XML metadata
  and a citation-provenance registry support the bibliography.
- Submission remains blocked only by author/affiliation/funding metadata,
  repository DOI assignment and third-party redistribution decisions.

## Boundary of inference

Zero-dimensional ignition, freely propagating one-dimensional flames and ideal
reactor networks provide combustion-side operability proxies. They do not
directly establish blow-off, flashback, thermoacoustic stability or full
gas-turbine performance.
