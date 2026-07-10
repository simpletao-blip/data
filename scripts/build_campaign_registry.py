"""Build a study-level curation registry without auto-admitting experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def preliminary_decision(row: pd.Series) -> tuple[str, str]:
    experiment = row["experiment_type"]
    definition = str(row["definition"])
    if experiment == "ignition delay measurement":
        if "relative concentration" in definition:
            return "exact_candidate", "Reported species-relative criterion can be reproduced."
        if "OH*" in definition:
            return "unsupported_exact", "OH* optical criterion is absent from ground-state mechanisms."
        return "pending_manual_review", "Ignition definition requires source-paper review."
    if experiment == "laminar burning velocity measurement":
        return "pending_manual_review", "Verify stretch/radiation correction and reported flame-speed definition."
    return "pending_manual_review", "Verify reactor thermal boundary, residence time, detection limits and sampling basis."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    keys = ["campaign_id", "doi", "experiment_type", "apparatus", "observable", "definition"]
    registry = (data.groupby(keys, dropna=False)
                .agg(rows=("dataset_id", "size"), source_files=("source_file", "nunique"),
                     min_temperature_K=("temperature_K", "min"), max_temperature_K=("temperature_K", "max"),
                     min_pressure_Pa=("pressure_Pa", "min"), max_pressure_Pa=("pressure_Pa", "max"),
                     min_cracking_ratio=("cracking_ratio", "min"), max_cracking_ratio=("cracking_ratio", "max"))
                .reset_index())
    decisions = registry.apply(preliminary_decision, axis=1)
    registry["preliminary_eligibility"] = [item[0] for item in decisions]
    registry["eligibility_reason"] = [item[1] for item in decisions]
    registry["final_role"] = "pending"
    registry["manual_notes"] = ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.output, index=False)
    print(registry.groupby("preliminary_eligibility").rows.sum().to_string())
    print(args.output)


if __name__ == "__main__":
    main()

