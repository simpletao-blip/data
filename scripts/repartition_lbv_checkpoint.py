"""Repartition completed LBV rows when changing the worker count."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    design = pd.read_csv(args.design)
    previous = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    completed = previous[previous.status.eq("completed")].drop_duplicates("design_id")
    unknown = set(completed.design_id).difference(design.design_id)
    if unknown:
        raise SystemExit(f"unknown completed design IDs: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.workers):
        ids = set(design.iloc[index::args.workers].design_id)
        shard = completed[completed.design_id.isin(ids)].copy()
        output = args.output_dir / f"shard_{index}.csv"
        shard.to_csv(output, index=False)
        print(f"{output}: retained {len(shard)} completed cases")


if __name__ == "__main__":
    main()
