"""Reuse exact direct-map flames and emit only missing Pareto confirmation designs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto", type=Path, required=True)
    parser.add_argument("--map", action="append", type=Path, required=True)
    parser.add_argument("--thermo-limits", type=Path,
                        default=Path("mechanisms/thermo_limits.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = pd.read_csv(args.pareto)
    candidates = candidates[candidates.support_tier.isin([
        "strict_criterion_supported", "proxy_supported"
    ])].copy()
    if candidates.empty:
        raise ValueError("no supported Pareto candidates")
    limits = pd.read_csv(args.thermo_limits).set_index("mechanism_id")
    reused: list[dict[str, object]] = []
    missing_by_mechanism: dict[str, list[dict[str, object]]] = {}
    for map_path in args.map:
        direct = pd.read_csv(map_path)
        mechanism_id = str(direct.mechanism_id.dropna().iloc[0])
        lower = float(limits.loc[mechanism_id, "min_temperature_K"])
        upper = float(limits.loc[mechanism_id, "max_temperature_K"])
        for candidate in candidates.itertuples(index=False):
            matched = direct[
                np.isclose(direct.temperature_K, candidate.flame_temperature_K)
                & np.isclose(direct.pressure_bar, candidate.pressure_bar)
                & np.isclose(direct.equivalence_ratio, candidate.equivalence_ratio)
                & np.isclose(direct.cracking_ratio, candidate.cracking_ratio)
            ]
            valid = matched[
                matched.status.eq("completed")
                & matched.temperature_K.between(lower, upper)
                & matched.max_temperature_K.between(lower, upper)
            ]
            if len(valid) > 1:
                raise ValueError(f"multiple exact direct flames for {mechanism_id} {candidate.design_id}")
            if len(valid) == 1:
                row = valid.iloc[0].to_dict()
                row["source_design_id"] = row["design_id"]
                row["design_id"] = candidate.design_id
                row["design_role"] = "direct_pareto_confirmation_reused"
                reused.append(row)
            else:
                missing_by_mechanism.setdefault(mechanism_id, []).append({
                    "design_id": candidate.design_id,
                    "design_role": "direct_pareto_confirmation_new",
                    "inside_experimental_convex_hull": candidate.inside_lbv_convex_hull,
                    "temperature_K": candidate.flame_temperature_K,
                    "pressure_bar": candidate.pressure_bar,
                    "equivalence_ratio": candidate.equivalence_ratio,
                    "cracking_ratio": candidate.cracking_ratio,
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reused).to_csv(args.output_dir / "reused_direct_flames.csv", index=False)
    manifest = []
    for mechanism_id in sorted(set(
        pd.read_csv(path).mechanism_id.dropna().iloc[0] for path in args.map
    )):
        rows = missing_by_mechanism.get(mechanism_id, [])
        path = args.output_dir / f"{mechanism_id}_missing_design.csv"
        pd.DataFrame(rows, columns=[
            "design_id", "design_role", "inside_experimental_convex_hull",
            "temperature_K", "pressure_bar", "equivalence_ratio", "cracking_ratio",
        ]).to_csv(path, index=False)
        manifest.append({
            "mechanism_id": mechanism_id,
            "candidate_count": len(candidates),
            "reused_count": sum(row["mechanism_id"] == mechanism_id for row in reused),
            "missing_count": len(rows),
            "missing_design_file": str(path),
        })
    pd.DataFrame(manifest).to_csv(args.output_dir / "confirmation_manifest.csv", index=False)
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()
