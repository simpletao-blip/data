"""Merge checkpoint shards with duplicate and design-completeness checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--key", default="design_id")
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    if frame[args.key].duplicated().any():
        duplicates = frame.loc[frame[args.key].duplicated(False), args.key].unique().tolist()
        raise ValueError(f"duplicate result keys across shards: {duplicates[:10]}")
    if args.design:
        design = pd.read_csv(args.design)
        expected = design[args.key].astype(str).tolist()
        observed = set(frame[args.key].astype(str))
        missing = [key for key in expected if key not in observed]
        extra = sorted(observed.difference(expected))
        if missing or extra:
            raise ValueError(f"design mismatch: missing={missing[:10]}, extra={extra[:10]}")
        order = {key: index for index, key in enumerate(expected)}
        frame["_design_order"] = frame[args.key].astype(str).map(order)
        frame = frame.sort_values("_design_order").drop(columns="_design_order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.groupby("status").size().to_string() if "status" in frame else len(frame))
    print(args.output)


if __name__ == "__main__":
    main()
