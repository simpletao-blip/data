# Partially cracked ammonia multi-mechanism study

This repository contains the author-generated data, analysis code and figure Source Data supporting the manuscript:

> Jiachao Tang, *Study-held-out validation exposes evidence gaps in multi-mechanism screening of partially cracked ammonia*.

Author: Jiachao Tang, College of Automotive Engineering, Jilin University, Changchun 130022, China. ORCID: [0009-0000-5472-5902](https://orcid.org/0009-0000-5472-5902).

## Repository contents

- `data/processed/`: normalized validation tables, campaign registry and simulation designs.
- `results/raw/`: condition-level Cantera outputs, including recorded failures.
- `results/processed/`: cross-validation, stacking, surrogate, uncertainty and screening results.
- `results/logs/`: numerical, provenance and manuscript consistency audits.
- `figures/source_data/`: machine-readable Source Data for all main and supplementary figures.
- `figures/exports/`: PNG, PDF and SVG versions of the seven main figures.
- `figures/supplementary/`: PNG, PDF and SVG versions of Supplementary Figs. S1-S13.
- `src/`, `scripts/`, `tests/`: reusable Python package, workflow scripts and automated tests.
- `mechanism_metadata/`: mechanism sources, versions, family groups and declared thermodynamic limits. Third-party mechanism files are not redistributed here.
- `metadata/`: repository inventory and file manifest.

The 600 dpi TIFF files are omitted from GitHub because they are reproducible from the included scripts and exceed a practical source-repository size. The submission TIFF files remain available from the corresponding author.

## Reproducible environment

```bash
conda env create -f environment.yml
conda activate cracked-ammonia
pip install -e .
pytest
```

The project-level `README.md`, scripts and configuration files document individual analysis and figure-generation commands. Results used in the manuscript are provided directly so that numerical claims can be checked without rerunning every one-dimensional flame.

## Data provenance and rights

No new experiments were performed. The validation database was normalized from the public ReSpecTh NH3 v2.3 collection. Kinetic mechanisms and literature supplements remain third-party works and are not relicensed or redistributed in this repository. See `RIGHTS.md`, `mechanism_metadata/registry.csv` and `metadata/repository_file_manifest.csv` for provenance and packaging decisions.

## Citation

Until a versioned DOI is issued, cite this repository as:

> Tang, J. (2026). Data and code for “Study-held-out validation exposes evidence gaps in multi-mechanism screening of partially cracked ammonia”, version 1.0.0. GitHub. https://github.com/simpletao-blip/data

## Contact

Jiachao Tang: tangjc2224@mails.jlu.edu.cn
