"""Split a validation design deterministically into interleaved worker shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    design = pd.read_csv(args.design)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index in range(args.shards):
        shard = design.iloc[index::args.shards].copy()
        output = args.output_dir / f"{args.prefix}_{index}.csv"
        shard.to_csv(output, index=False)
        outputs.append(output)
        print(f"{output}: {len(shard)} cases")

    merged_ids = pd.concat([pd.read_csv(path) for path in outputs]).design_id
    if len(merged_ids) != len(design) or merged_ids.duplicated().any():
        raise SystemExit("shard completeness check failed")


if __name__ == "__main__":
    main()
