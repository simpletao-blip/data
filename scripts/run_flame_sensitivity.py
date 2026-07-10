"""Compute direct normalized flame-speed reaction sensitivities for one condition."""

from __future__ import annotations

import argparse
from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd

from pca_ensemble.reactors import set_cracked_ammonia_state


def reaction_family(equation: str) -> str:
    text = equation.upper()
    if any(token in text for token in ("NO", "N2O", "HNO", "H2NO", "HONO")):
        return "NOx/N2O chemistry"
    if any(token in text for token in ("NH3", "NH2", "NNH", "HNNH", " NH ")):
        return "NHx conversion"
    if any(token in text for token in ("H2", "O2", " OH", " H ", " O ", "H2O")):
        return "H/O radical chemistry"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=5.0)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    gas = ct.Solution(str(args.mechanism))
    set_cracked_ammonia_state(
        gas, args.temperature_K, args.pressure_bar * 1e5, args.phi, args.alpha
    )
    flame = ct.FreeFlame(gas, width=0.05)
    flame.set_refine_criteria(ratio=3.0, slope=0.06, curve=0.10)
    flame.transport_model = "multicomponent"
    flame.soret_enabled = True
    flame.solve(loglevel=0, auto=True)
    sensitivities = np.asarray(flame.get_flame_speed_reaction_sensitivities(), dtype=float)
    frame = pd.DataFrame({
        "mechanism_id": args.mechanism_id,
        "reaction_index": np.arange(gas.n_reactions),
        "reaction_equation": gas.reaction_equations(),
        "normalized_flame_speed_sensitivity": sensitivities,
    })
    frame["absolute_sensitivity"] = frame.normalized_flame_speed_sensitivity.abs()
    frame["absolute_rank"] = frame.absolute_sensitivity.rank(method="first", ascending=False).astype(int)
    frame["reaction_family"] = frame.reaction_equation.map(reaction_family)
    frame["temperature_K"] = args.temperature_K
    frame["pressure_bar"] = args.pressure_bar
    frame["equivalence_ratio"] = args.phi
    frame["cracking_ratio"] = args.alpha
    frame["flame_speed_m_per_s"] = float(flame.velocity[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("absolute_rank").to_csv(args.output, index=False)
    print(frame.nsmallest(12, "absolute_rank")[[
        "absolute_rank", "reaction_equation", "normalized_flame_speed_sensitivity"
    ]].to_string(index=False))
    print(args.output)


if __name__ == "__main__":
    main()
