"""Compare legacy direct-start and staged-transport LBV solutions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("pairs", nargs="+", help="mechanism_id:legacy.csv:staged.csv")
    args = parser.parse_args()
    rows = []
    for item in args.pairs:
        mechanism, legacy_path, staged_path = item.split(":", 2)
        legacy = pd.read_csv(legacy_path).set_index("design_id")
        staged = pd.read_csv(staged_path).set_index("design_id")
        common = legacy.join(staged, lsuffix="_legacy", rsuffix="_staged", how="inner")
        valid = common[
            common.status_legacy.eq("completed") & common.status_staged.eq("completed")
        ].copy()
        relative = np.abs(
            valid.simulated_m_per_s_staged - valid.simulated_m_per_s_legacy
        ) / valid.simulated_m_per_s_legacy
        rows.append({
            "mechanism_id": mechanism,
            "n_legacy": len(legacy),
            "n_staged": len(staged),
            "n_paired_completed": len(valid),
            "median_relative_difference": float(relative.median()),
            "maximum_relative_difference": float(relative.max()),
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
