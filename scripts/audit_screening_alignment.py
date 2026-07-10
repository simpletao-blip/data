"""Audit that screening outputs exactly match the declared physical design."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_MAPPINGS = {
    "pressure_bar": "pressure_bar",
    "equivalence_ratio": "equivalence_ratio",
    "cracking_ratio": "cracking_ratio",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["idt", "lbv", "reactor"], required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    design = pd.read_csv(args.design)
    if design.design_id.duplicated().any():
        raise ValueError("design contains duplicate design_id values")
    expected = set(design.design_id)
    reference = design.set_index("design_id")
    rows = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        duplicate_count = int(frame.design_id.duplicated().sum())
        observed = set(frame.design_id)
        missing = expected - observed
        unknown = observed - expected
        condition_mismatches = 0
        common = sorted(expected & observed)
        mappings = dict(BASE_MAPPINGS)
        if args.mode == "idt":
            mappings.update({
                "temperature_K": "ignition_temperature_K",
                "flame_reactor_temperature_K": "temperature_K",
            })
        elif args.mode == "lbv":
            mappings.update({
                "flame_temperature_K": "flame_temperature_K",
                "temperature_K": "temperature_K",
                "ignition_temperature_K": "ignition_temperature_K",
            })
        else:
            mappings.update({
                "temperature_K": "temperature_K",
                "residence_time_ms": "residence_time_ms",
                "psr_fraction": "psr_fraction",
                "heat_loss_W_per_K": "heat_loss_W_per_K",
            })
        for output_column, design_column in mappings.items():
            if output_column not in frame.columns:
                continue
            values = frame.set_index("design_id").loc[common, output_column]
            target = reference.loc[common, design_column]
            if pd.api.types.is_numeric_dtype(target):
                condition_mismatches += int((~np.isclose(
                    values.astype(float), target.astype(float), rtol=0, atol=1e-10,
                    equal_nan=True,
                )).sum())
            else:
                condition_mismatches += int((values.astype(str) != target.astype(str)).sum())
        failed = int((~frame.status.eq("completed")).sum()) if "status" in frame else 0
        passed = not (duplicate_count or missing or unknown or condition_mismatches or failed)
        rows.append({
            "file": str(path),
            "row_count": len(frame),
            "duplicate_design_ids": duplicate_count,
            "missing_design_ids": len(missing),
            "unknown_design_ids": len(unknown),
            "condition_mismatches": condition_mismatches,
            "noncompleted_rows": failed,
            "passed": passed,
        })
    audit = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    print(audit.to_string(index=False))
    if not audit.passed.all():
        raise SystemExit("screening alignment audit failed")


if __name__ == "__main__":
    main()
