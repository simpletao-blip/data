"""Audit Cantera mechanism files and write a reproducibility ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

from pca_ensemble.io import sha256
from pca_ensemble.mechanisms import audit_mechanism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mechanisms", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/logs/mechanism_audit.csv"))
    args = parser.parse_args()
    rows = []
    for path in args.mechanisms:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                audit = audit_mechanism(path)
            warning_text = " | ".join(str(item.message).strip() for item in caught)
            rows.append({"mechanism_file": str(path), "sha256": sha256(path),
                         "status": "passed_with_warnings" if caught else "passed",
                         "warning_count": len(caught), "warnings": warning_text,
                         "error": "", **audit.to_dict()})
        except Exception as exc:
            rows.append({"mechanism_file": str(path), "sha256": sha256(path),
                         "status": "failed", "warning_count": 0, "warnings": "",
                         "error": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    columns = [column for column in ["mechanism_file", "species_count", "reaction_count",
                                      "multicomponent_transport_available", "status"]
               if column in frame]
    print(frame[columns].to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
