"""Run checkpointed 0D ignition-proxy map cases with a fixed criterion."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cantera as ct
import numpy as np
import pandas as pd

from pca_ensemble.reactors import simulate_ignition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-time-s", type=float, default=5.0)
    parser.add_argument("--temperature-column", default="temperature_K")
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
        simulation_temperature_K = float(getattr(record, args.temperature_column))
        base = {
            "design_id": record.design_id, "design_role": record.design_role,
            "inside_all_idt_convex_hull": record.inside_all_idt_convex_hull,
            "inside_exact_criterion_convex_hull": record.inside_exact_criterion_convex_hull,
            "mechanism_id": args.mechanism_id, "temperature_K": simulation_temperature_K,
            "pressure_bar": record.pressure_bar, "equivalence_ratio": record.equivalence_ratio,
            "cracking_ratio": record.cracking_ratio, "criterion": "max_dTdt",
            "reactor": "constant_volume",
        }
        if args.temperature_column != "temperature_K":
            base["flame_reactor_temperature_K"] = float(record.temperature_K)
            base["temperature_source_column"] = args.temperature_column
        try:
            result = simulate_ignition(
                gas, simulation_temperature_K, float(record.pressure_bar) * 1e5,
                float(record.equivalence_ratio), float(record.cracking_ratio),
                reactor="constant_volume", criterion="max_dTdt", max_time_s=args.max_time_s,
            )
            rows.append({
                **base, "status": "completed" if result.converged else "no_ignition",
                **result.to_dict(), "runtime_s": perf_counter() - start,
            })
        except Exception as exc:
            rows.append({
                **base, "status": "failed", "ignition_delay_s": np.nan,
                "final_temperature_K": np.nan, "max_temperature_K": np.nan,
                "integration_steps": 0, "converged": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "runtime_s": perf_counter() - start,
            })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(record.design_id, rows[-1]["status"], f"{rows[-1]['runtime_s']:.3f}s", flush=True)
    result_frame = pd.DataFrame(rows)
    print(result_frame.groupby("status").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
