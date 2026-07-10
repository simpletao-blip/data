"""Check 1D flame sensitivity to domain width and refinement criteria."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cantera as ct
import pandas as pd

from pca_ensemble.reactors import simulate_free_flame


CASES = [
    {"case": "lean_low_crack", "temperature_K": 300.0, "pressure_bar": 1.0,
     "equivalence_ratio": 0.7, "cracking_ratio": 0.1},
    {"case": "stoich_mid_crack", "temperature_K": 300.0, "pressure_bar": 1.0,
     "equivalence_ratio": 1.0, "cracking_ratio": 0.3},
    {"case": "rich_high_pressure", "temperature_K": 450.0, "pressure_bar": 10.0,
     "equivalence_ratio": 1.2, "cracking_ratio": 0.5},
]

GRIDS = [
    {"grid": "coarse", "width_m": 0.03, "refine": {"ratio": 4.0, "slope": 0.10, "curve": 0.15}},
    {"grid": "base", "width_m": 0.05, "refine": {"ratio": 3.0, "slope": 0.06, "curve": 0.10}},
    {"grid": "fine", "width_m": 0.08, "refine": {"ratio": 2.0, "slope": 0.03, "curve": 0.05}},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gas = ct.Solution(str(args.mechanism))
    rows = []
    for case in CASES:
        for grid in GRIDS:
            start = perf_counter()
            result = simulate_free_flame(
                gas, case["temperature_K"], case["pressure_bar"] * 1e5,
                case["equivalence_ratio"], case["cracking_ratio"],
                width_m=grid["width_m"], refine=grid["refine"],
            )
            rows.append({**case, **grid, "mechanism_id": args.mechanism_id,
                         **result.to_dict(), "runtime_s": perf_counter() - start})
            print(case["case"], grid["grid"], result.converged, flush=True)
    frame = pd.DataFrame(rows)
    base = frame[frame.grid.eq("base")].set_index("case").laminar_burning_velocity_m_per_s
    fine = frame[frame.grid.eq("fine")].set_index("case").laminar_burning_velocity_m_per_s
    differences = ((fine - base).abs() / fine).rename("base_to_fine_relative_difference")
    frame = frame.join(differences, on="case")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(differences.to_string())
    print(args.output)


if __name__ == "__main__":
    main()

