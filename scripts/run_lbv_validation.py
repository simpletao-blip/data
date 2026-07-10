"""Run direct-mixture LBV validation for a deterministic design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cantera as ct
import numpy as np
import pandas as pd

from pca_ensemble.reactors import simulate_reported_free_flame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--loglevel", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    design = pd.read_csv(args.design)
    if args.max_cases is not None:
        design = design.head(args.max_cases)
    gas = ct.Solution(str(args.mechanism))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output)
        rows = previous.to_dict("records")
        done = set(previous.design_id)
        design = design[~design.design_id.isin(done)]
    else:
        rows = []
    for record in design.itertuples(index=False):
        start = perf_counter()
        base = {
            "design_id": record.design_id, "dataset_id": record.dataset_id,
            "campaign_id": record.campaign_id, "doi": record.doi,
            "mechanism_id": args.mechanism_id, "experimental_m_per_s": record.value,
            "temperature_K": record.temperature_K, "pressure_Pa": record.pressure_Pa,
            "equivalence_ratio": record.equivalence_ratio,
            "cracking_ratio": record.cracking_ratio, "apparatus": record.apparatus,
        }
        try:
            result = simulate_reported_free_flame(
                gas, float(record.temperature_K), float(record.pressure_Pa),
                json.loads(record.initial_composition), loglevel=args.loglevel,
            )
            runtime = perf_counter() - start
            simulated = result.laminar_burning_velocity_m_per_s
            relative_error = (abs(simulated - float(record.value)) / float(record.value)
                              if result.converged and float(record.value) > 0 else np.nan)
            rows.append({
                **base, "status": "completed" if result.converged else "failed",
                "simulated_m_per_s": simulated, "relative_error": relative_error,
                "converged": result.converged, "failure_reason": result.failure_reason,
                "grid_points": result.grid_points, "transport_model": result.transport_model,
                "soret_enabled": result.soret_enabled, "runtime_s": runtime,
            })
            print(record.design_id, "ok" if result.converged else "failed", f"{runtime:.2f}s", flush=True)
        except Exception as exc:
            runtime = perf_counter() - start
            rows.append({
                **base, "status": "skipped", "simulated_m_per_s": np.nan,
                "relative_error": np.nan, "converged": False,
                "failure_reason": f"{type(exc).__name__}: {exc}", "grid_points": 0,
                "transport_model": "", "soret_enabled": True, "runtime_s": runtime,
            })
            print(record.design_id, "skipped", f"{runtime:.2f}s", flush=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
    output = pd.DataFrame(rows)
    output.to_csv(args.output, index=False)
    print(output.groupby("status").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
