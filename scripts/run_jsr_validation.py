"""Run checkpointed JSR species validation for one mechanism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cantera as ct
import numpy as np
import pandas as pd

from pca_ensemble.species_validation import simulate_isothermal_jsr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclude-condition", action="append", default=[])
    args = parser.parse_args()
    design = pd.read_csv(args.design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.output).to_dict("records") if args.resume and args.output.exists() else []
    terminal = {row["condition_id"] for row in rows
                if row.get("status") in {"completed", "excluded"}}
    if rows:
        rows = [row for row in rows if row["condition_id"] in terminal]
    gas = ct.Solution(str(args.mechanism))
    for condition_id, group in design.groupby("condition_id", sort=False):
        if condition_id in terminal:
            continue
        first = group.iloc[0]
        if condition_id in set(args.exclude_condition):
            for record in group.to_dict("records"):
                rows.append({**record, "mechanism_id": args.mechanism_id,
                             "status": "excluded", "simulated_value": np.nan,
                             "simulated_unit": "mole fraction", "runtime_s_per_condition": np.nan,
                             "failure_reason": "predeclared exclusion after repeated steady-state timeout",
                             "solver_path": "excluded"})
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(condition_id, "excluded", flush=True)
            continue
        start = perf_counter()
        try:
            result = simulate_isothermal_jsr(
                gas, float(first.temperature_K), float(first.pressure_Pa),
                json.loads(first.initial_composition), float(first.residence_time_s),
            )
            runtime = perf_counter() - start
            for record in group.to_dict("records"):
                species = str(record["observable"])
                rows.append({
                    **record, "mechanism_id": args.mechanism_id,
                    "status": "completed" if result.converged else "failed",
                    "simulated_value": result.mole_fractions.get(species, np.nan),
                    "simulated_unit": "mole fraction", "runtime_s_per_condition": runtime,
                    "reactor_temperature_K": result.temperature_K,
                    "reactor_pressure_Pa": result.pressure_Pa, "integration_steps": result.steps,
                    "failure_reason": result.failure_reason, "solver_path": result.solver_path,
                })
        except Exception as exc:
            runtime = perf_counter() - start
            for record in group.to_dict("records"):
                rows.append({**record, "mechanism_id": args.mechanism_id, "status": "failed",
                             "simulated_value": np.nan, "simulated_unit": "mole fraction",
                             "runtime_s_per_condition": runtime,
                             "failure_reason": f"{type(exc).__name__}: {exc}"})
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(condition_id, rows[-1]["status"], f"{runtime:.3f}s", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.drop_duplicates("condition_id").groupby("status").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
