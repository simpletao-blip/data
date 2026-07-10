"""Run checkpointed partially cracked-ammonia 1D operating-map cases."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cantera as ct
import numpy as np
import pandas as pd

from pca_ensemble.reactors import simulate_free_flame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    design = pd.read_csv(args.design)
    if args.max_cases is not None:
        design = design.head(args.max_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output)
        rows = previous.to_dict("records")
        design = design[~design.design_id.isin(set(previous.design_id))]
    else:
        rows = []
    gas = ct.Solution(str(args.mechanism))
    for record in design.itertuples(index=False):
        start = perf_counter()
        base = {
            "design_id": record.design_id, "design_role": record.design_role,
            "inside_experimental_convex_hull": record.inside_experimental_convex_hull,
            "mechanism_id": args.mechanism_id, "temperature_K": record.temperature_K,
            "pressure_bar": record.pressure_bar, "equivalence_ratio": record.equivalence_ratio,
            "cracking_ratio": record.cracking_ratio,
        }
        try:
            result = simulate_free_flame(
                gas, float(record.temperature_K), float(record.pressure_bar) * 1e5,
                float(record.equivalence_ratio), float(record.cracking_ratio),
            )
            rows.append({
                **base, "status": "completed" if result.converged else "failed",
                "laminar_burning_velocity_m_per_s": result.laminar_burning_velocity_m_per_s,
                "max_temperature_K": result.max_temperature_K,
                "grid_points": result.grid_points, "transport_model": result.transport_model,
                "soret_enabled": result.soret_enabled, "failure_reason": result.failure_reason,
                "runtime_s": perf_counter() - start,
            })
        except Exception as exc:
            rows.append({
                **base, "status": "failed", "laminar_burning_velocity_m_per_s": np.nan,
                "max_temperature_K": np.nan, "grid_points": 0, "transport_model": "",
                "soret_enabled": True, "failure_reason": f"{type(exc).__name__}: {exc}",
                "runtime_s": perf_counter() - start,
            })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(record.design_id, rows[-1]["status"], f"{rows[-1]['runtime_s']:.2f}s", flush=True)
    result_frame = pd.DataFrame(rows)
    print(result_frame.groupby("status").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()

