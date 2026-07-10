"""Generate a reproducible quality-control report for the parsed RKD table."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


COMPOSITION_OBSERVABLES = {"NH3", "H2", "O2", "NO", "NO2", "N2O", "N2", "H2O", "CO"}


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional pandas dependencies."""
    if frame.empty:
        return "(none)"
    display = frame.fillna("").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, default=Path("literature/respecth_data_audit.md"))
    parser.add_argument("--flags", type=Path, default=Path("data/processed/respecth_nh3_quality_flags.csv"))
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    flags: list[dict[str, object]] = []

    def flag(mask: pd.Series, code: str, detail: str) -> None:
        for row in data.loc[mask, ["dataset_id", "source_file"]].itertuples(index=False):
            flags.append({"dataset_id": row.dataset_id, "source_file": row.source_file,
                          "flag": code, "detail": detail})

    flag(data.dataset_id.duplicated(keep=False), "duplicate_dataset_id", "Dataset identifier is not unique.")
    flag(data.temperature_K.isna() | (data.temperature_K <= 0), "invalid_temperature", "Temperature is missing or non-positive.")
    flag(data.pressure_Pa.isna() | (data.pressure_Pa <= 0), "invalid_pressure", "Pressure is missing or non-positive.")
    flag(data.value.isna() | (data.value < 0), "invalid_value", "Measurement is missing or negative.")
    is_composition = data.observable.isin(COMPOSITION_OBSERVABLES)
    flag(is_composition & (data.value > 1), "composition_above_one", "SI-normalized mole fraction exceeds one.")
    flag(data.doi.isna() | data.doi.eq(""), "missing_doi", "Source article DOI is missing.")
    flag(data.uncertainty.isna(), "missing_uncertainty", "No uncertainty value was parsed.")

    flag_frame = pd.DataFrame(flags, columns=["dataset_id", "source_file", "flag", "detail"])
    args.flags.parent.mkdir(parents=True, exist_ok=True)
    flag_frame.to_csv(args.flags, index=False)

    missing = (data[["doi", "temperature_K", "pressure_Pa", "equivalence_ratio",
                     "cracking_ratio", "uncertainty", "laboratory"]]
               .isna().replace({False: 0, True: 1}).mean().mul(100))
    missing["laboratory"] = data.laboratory.fillna("").eq("").mean() * 100
    counts = (data.groupby(["experiment_type", "observable"])
              .agg(rows=("dataset_id", "size"), campaigns=("campaign_id", "nunique"),
                   files=("source_file", "nunique"))
              .reset_index())
    unit_counts = data.groupby(["observable", "unit"]).size().rename("rows").reset_index()
    flag_counts = (flag_frame.groupby("flag").size().rename("rows").reset_index()
                   if not flag_frame.empty else pd.DataFrame(columns=["flag", "rows"]))

    lines = [
        "# ReSpecTh NH3 v2.3 parsed-data audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Provenance and status",
        "",
        f"- Input: `{args.input.as_posix()}`",
        f"- Parsed records: {len(data):,}",
        f"- RKD source files: {data.source_file.nunique():,}",
        f"- Source-article campaigns (DOI-grouped): {data.campaign_id.nunique():,}",
        f"- Unique source DOIs: {data.doi.nunique():,}",
        "- Admission status: all records remain `pending_manual_review`.",
        "- Cracking ratio is an equivalent fuel-side descriptor inferred from H2/NH3; it does not prove physical cracking.",
        "",
        "## Coverage",
        "",
        markdown_table(counts),
        "",
        "## Canonical units",
        "",
        markdown_table(unit_counts),
        "",
        "## Missingness (%)",
        "",
        markdown_table(missing.rename("missing_percent").round(2).rename_axis("field").reset_index()),
        "",
        "Equivalence ratio is not universally applicable to all reactor records. Laboratory affiliation is not encoded as a structured RKD field and must be curated from each paper; the RKD file curator is stored separately.",
        "",
        "## Automated flags",
        "",
        markdown_table(flag_counts) if not flag_counts.empty else "No automated hard-failure flags.",
        "",
        "## Admission requirements before validation",
        "",
        "1. Verify apparatus, ignition criterion, flame-speed definition and correction method against the article.",
        "2. Confirm mixture basis, diluent identity and whether the H2/NH3 blend represents cracked ammonia.",
        "3. Resolve laboratory affiliation and define one campaign per independent experimental study.",
        "4. Check uncertainty semantics and detection limits for each observable.",
        "5. Mark records as included or excluded with an explicit reason; do not train on pending records.",
        "",
        "## Interpretation",
        "",
        "The archive is suitable as a discovery and curation backbone, not as an automatically admissible validation dataset. DOI-level grouping prevents individual points from the same paper from leaking across cross-validation folds.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(args.report)
    print(args.flags)
    print(f"flags={len(flag_frame)}")


if __name__ == "__main__":
    main()
