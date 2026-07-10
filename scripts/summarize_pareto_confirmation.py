"""Compare direct Pareto-finalist flames with their screening predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, nargs="+", required=True)
    parser.add_argument("--screening", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/processed/pareto_flame_confirmation.csv"))
    args = parser.parse_args()
    direct = pd.concat([pd.read_csv(path) for path in args.direct], ignore_index=True)
    screening = pd.concat([pd.read_csv(path) for path in args.screening], ignore_index=True)
    direct = direct[direct.status.eq("completed")].copy()
    columns = [
        "design_id", "mechanism_id", "surrogate_m_per_s", "surrogate_90_low_m_per_s",
        "surrogate_90_high_m_per_s",
    ]
    result = direct.merge(screening[columns], on=["design_id", "mechanism_id"],
                          validate="one_to_one")
    result["surrogate_relative_error"] = (
        (result.surrogate_m_per_s - result.laminar_burning_velocity_m_per_s).abs()
        / result.laminar_burning_velocity_m_per_s
    )
    result["direct_inside_surrogate_90_interval"] = result.laminar_burning_velocity_m_per_s.between(
        result.surrogate_90_low_m_per_s, result.surrogate_90_high_m_per_s
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[["design_id", "mechanism_id", "surrogate_relative_error",
                  "direct_inside_surrogate_90_interval"]].to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
