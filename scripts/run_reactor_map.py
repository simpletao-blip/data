"""Run checkpointed PSR-PFR emission-map cases for one mechanism."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cantera as ct
import numpy as np
import pandas as pd

from pca_ensemble.io import load_yaml
from pca_ensemble.reactor_network import oxygen_corrected_dry_fraction, simulate_psr_pfr


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
    config = load_yaml("config/study.yaml")["reactor_network"]
    oxygen_reference = float(config["oxygen_reference_dry"])
    volume = float(config["reactor_volume_m3"])
    gas = ct.Solution(str(args.mechanism))
    for record in design.itertuples(index=False):
        start = perf_counter()
        base = record._asdict()
        base["mechanism_id"] = args.mechanism_id
        try:
            result = simulate_psr_pfr(
                gas, float(record.temperature_K), float(record.pressure_bar) * 1e5,
                float(record.equivalence_ratio), float(record.cracking_ratio),
                float(record.residence_time_ms) * 1e-3, float(record.psr_fraction),
                float(record.heat_loss_W_per_K), reactor_volume_m3=volume,
            )
            row = {**base, "status": "completed" if result.converged else "failed",
                   "runtime_s": perf_counter() - start,
                   **{key: value for key, value in result.to_dict().items()
                      if not isinstance(value, dict)}}
            row.update({f"X_{key}": value for key, value in result.outlet_mole_fractions.items()})
            row.update({f"X_dry_{key}": value for key, value in result.outlet_dry_mole_fractions.items()})
            row.update({f"Y_{key}": value for key, value in result.outlet_mass_fractions.items()})
            row.update({f"EI_g_per_MJ_{key}": value for key, value in result.emission_indices_g_per_MJ.items()})
            measured_o2 = result.outlet_dry_mole_fractions.get("O2", np.nan)
            if np.isfinite(measured_o2) and measured_o2 < 0.209:
                for species in ("NO", "NO2", "N2O", "NH3"):
                    value = result.outlet_dry_mole_fractions.get(species, np.nan)
                    row[f"X_dry_O2corr15_{species}"] = oxygen_corrected_dry_fraction(
                        value, measured_o2, oxygen_reference
                    ) if np.isfinite(value) else np.nan
            rows.append(row)
        except Exception as exc:
            rows.append({**base, "mechanism_id": args.mechanism_id, "status": "failed",
                         "failure_reason": f"{type(exc).__name__}: {exc}",
                         "runtime_s": perf_counter() - start})
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(record.design_id, rows[-1]["status"], f"{rows[-1]['runtime_s']:.2f}s", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby("status").size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()

