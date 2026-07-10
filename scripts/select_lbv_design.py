"""Select a campaign-balanced, maximin LBV validation design."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pca_ensemble.design import select_lbv_design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-campaign", type=int, default=5)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    design = select_lbv_design(data, args.per_campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output, index=False)
    print(f"cases={len(design)} campaigns={design.campaign_id.nunique()}")
    print(design.groupby("apparatus").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()

