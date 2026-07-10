"""Run criterion-matched IDT validation for one registered mechanism."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pca_ensemble.validation import validate_idt_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/respecth_nh3_long.csv"))
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    result = validate_idt_table(data, str(args.mechanism), args.mechanism_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.groupby("status").size().to_string())
    completed = result[result.status.eq("completed")]
    if not completed.empty:
        print(f"median_abs_log10_error={completed.absolute_log10_error.median():.6g}")
        print(f"mean_abs_log10_error={completed.absolute_log10_error.mean():.6g}")
    print(args.output)


if __name__ == "__main__":
    main()

