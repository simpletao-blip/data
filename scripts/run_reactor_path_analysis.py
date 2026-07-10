"""Export nitrogen reaction paths and species production contributions at bounded reactor states."""

from __future__ import annotations

import argparse
from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd

from pca_ensemble.io import load_yaml
from pca_ensemble.reactors import set_cracked_ammonia_state


TARGETS = ("NO", "N2O", "NH3", "NH2", "HNO")


def parse_mechanism(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mechanism must be MECHANISM_ID=PATH")
    mechanism_id, path = value.split("=", 1)
    return mechanism_id, Path(path)


def bounded_states(
    mechanism: Path,
    temperature_K: float,
    pressure_bar: float,
    phi: float,
    alpha: float,
    residence_time_ms: float,
    psr_fraction: float,
) -> dict[str, ct.Solution]:
    config = load_yaml("config/study.yaml")["reactor_network"]
    volume = float(config["reactor_volume_m3"])
    pressure = pressure_bar * 1e5
    gas = ct.Solution(str(mechanism))
    set_cracked_ammonia_state(gas, temperature_K, pressure, phi, alpha)
    inlet = ct.Reservoir(gas, clone=True)
    gas.equilibrate("HP")
    psr = ct.IdealGasConstPressureReactor(gas, energy="on", volume=volume, clone=True)
    exhaust = ct.Reservoir(gas, clone=True)
    psr_tau = residence_time_ms * 1e-3 * psr_fraction
    pfr_tau = residence_time_ms * 1e-3 * (1.0 - psr_fraction)
    mass_flow = ct.MassFlowController(inlet, psr, mdot=lambda _: psr.mass / psr_tau)
    ct.PressureController(psr, exhaust, primary=mass_flow, K=1e-5)
    network = ct.ReactorNet([psr])
    network.advance_to_steady_state(max_steps=100000)

    psr_state = ct.Solution(str(mechanism))
    psr_state.TPX = psr.T, pressure, psr.phase.X
    pfr = ct.IdealGasConstPressureReactor(psr.phase, energy="on", volume=volume, clone=True)
    ct.ReactorNet([pfr]).advance(pfr_tau)
    outlet_state = ct.Solution(str(mechanism))
    outlet_state.TPX = pfr.T, pressure, pfr.phase.X
    return {"psr_steady": psr_state, "pfr_outlet": outlet_state}


def contribution_rows(
    phase: ct.Solution, mechanism_id: str, location: str, top_n: int
) -> list[dict[str, object]]:
    net_stoich = phase.product_stoich_coeffs - phase.reactant_stoich_coeffs
    rates = np.asarray(phase.net_rates_of_progress, dtype=float)
    equations = phase.reaction_equations()
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        if target not in phase.species_names:
            continue
        index = phase.species_index(target)
        contributions = net_stoich[index, :] * rates
        scale = float(np.sum(np.abs(contributions)))
        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        for rank, reaction_index in enumerate(order, start=1):
            value = float(contributions[reaction_index])
            if value == 0:
                continue
            rows.append({
                "mechanism_id": mechanism_id,
                "location": location,
                "temperature_K": float(phase.T),
                "pressure_bar": float(phase.P / 1e5),
                "target_species": target,
                "absolute_rank": rank,
                "reaction_index": int(reaction_index),
                "reaction_equation": equations[reaction_index],
                "production_contribution_kmol_m3_s": value,
                "normalized_absolute_contribution": abs(value) / max(scale, 1e-300),
                "direction": "production" if value > 0 else "consumption",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", action="append", type=parse_mechanism, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/processed/reactor_path_analysis"))
    parser.add_argument("--temperature-K", type=float, default=600.0)
    parser.add_argument("--pressure-bar", type=float, default=5.0)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--residence-time-ms", type=float, default=10.0)
    parser.add_argument("--psr-fraction", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    states = []
    for mechanism_id, mechanism_path in args.mechanism:
        for location, phase in bounded_states(
            mechanism_path, args.temperature_K, args.pressure_bar, args.phi,
            args.alpha, args.residence_time_ms, args.psr_fraction,
        ).items():
            all_rows.extend(contribution_rows(phase, mechanism_id, location, args.top_n))
            states.append({
                "mechanism_id": mechanism_id,
                "location": location,
                "temperature_K": phase.T,
                "pressure_bar": phase.P / 1e5,
                "equivalence_ratio": args.phi,
                "cracking_ratio": args.alpha,
                "residence_time_ms": args.residence_time_ms,
                "psr_fraction": args.psr_fraction,
            })
            diagram = ct.ReactionPathDiagram(phase, "N")
            diagram.threshold = 0.01
            diagram.label_threshold = 0.01
            diagram.title = f"{mechanism_id}: {location} nitrogen paths"
            diagram.write_dot(str(args.output_dir / f"{mechanism_id}_{location}_N_paths.dot"))
    pd.DataFrame(all_rows).to_csv(
        args.output_dir / "species_reaction_contributions.csv", index=False
    )
    pd.DataFrame(states).to_csv(args.output_dir / "state_manifest.csv", index=False)
    print(pd.DataFrame(states).to_string(index=False))
    print(args.output_dir)


if __name__ == "__main__":
    main()
