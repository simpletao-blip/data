"""Merge deterministic LBV worker shards with completeness checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()

    design = pd.read_csv(args.design)
    result = pd.concat([pd.read_csv(path) for path in args.shards], ignore_index=True)
    duplicated = result.design_id[result.design_id.duplicated()].tolist()
    if duplicated:
        raise SystemExit(f"duplicate design IDs: {duplicated}")
    unknown = sorted(set(result.design_id).difference(design.design_id))
    if unknown:
        raise SystemExit(f"unknown design IDs: {unknown}")
    order = {design_id: index for index, design_id in enumerate(design.design_id)}
    result["_order"] = result.design_id.map(order)
    result = result.sort_values("_order").drop(columns="_order")
    missing = sorted(set(design.design_id).difference(result.design_id))
    if missing:
        raise SystemExit(f"incomplete shards; missing {len(missing)} cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.groupby("status").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
