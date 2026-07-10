"""Select supported surrogate Pareto candidates for direct 1D confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto", type=Path,
                        default=Path("results/processed/full_proxy_screening_pareto.csv"))
    parser.add_argument("--support-tier", default="proxy_supported")
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/pareto_flame_confirmation_design.csv"))
    args = parser.parse_args()
    pareto = pd.read_csv(args.pareto)
    chosen = pareto[pareto.support_tier.eq(args.support_tier)].copy()
    if chosen.empty:
        raise ValueError(f"no Pareto candidates at support tier {args.support_tier!r}")
    design = chosen[[
        "design_id", "flame_temperature_K", "pressure_bar", "equivalence_ratio",
        "cracking_ratio", "inside_lbv_convex_hull",
    ]].copy()
    design = design.rename(columns={
        "flame_temperature_K": "temperature_K",
        "inside_lbv_convex_hull": "inside_experimental_convex_hull",
    })
    design.insert(1, "design_role", "direct_pareto_confirmation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output, index=False)
    print(design.to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
