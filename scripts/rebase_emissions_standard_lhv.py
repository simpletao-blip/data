"""Re-express reactor emission indices using one mechanism-independent LHV basis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MW = {"NH3": 17.03052, "H2": 2.01588, "N2": 28.0134, "O2": 31.9988}
LHV_J_PER_KG = {"NH3": 18.6e6, "H2": 120.0e6}


def standard_heat_input(row: pd.Series) -> float:
    alpha = float(row.cracking_ratio)
    phi = float(row.equivalence_ratio)
    amounts = {
        "NH3": 1.0 - alpha,
        "H2": 1.5 * alpha,
        "N2": 0.5 * alpha + 3.76 * 0.75 / phi,
        "O2": 0.75 / phi,
    }
    total_moles = sum(amounts.values())
    mean_mw = sum(amounts[name] * MW[name] for name in amounts) / total_moles
    heat_per_kmol_initial_nh3 = (
        amounts["NH3"] * MW["NH3"] * LHV_J_PER_KG["NH3"]
        + amounts["H2"] * MW["H2"] * LHV_J_PER_KG["H2"]
    )
    heat_per_kmol_mixture = heat_per_kmol_initial_nh3 / total_moles
    return float(row.steady_mass_flow_kg_per_s) / mean_mw * heat_per_kmol_mixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+",
                        default=[Path("results/processed/reactor_mechanism_predictions.csv")])
    parser.add_argument("--output", type=Path,
                        default=Path("results/processed/reactor_emissions_standard_lhv.csv"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.input], ignore_index=True)
    completed = data.status.eq("completed") & data.steady_mass_flow_kg_per_s.notna()
    data["standard_fuel_heat_input_W"] = pd.NA
    data.loc[completed, "standard_fuel_heat_input_W"] = data.loc[completed].apply(
        standard_heat_input, axis=1
    )
    data["mechanism_to_standard_heat_ratio"] = (
        data.fuel_heat_input_W / pd.to_numeric(data.standard_fuel_heat_input_W)
    )
    for species in ("NO", "NO2", "N2O", "NH3"):
        data[f"EI_standard_LHV_g_per_MJ_{species}"] = (
            data[f"EI_g_per_MJ_{species}"] * data.mechanism_to_standard_heat_ratio
        )
    data["EI_standard_LHV_g_per_MJ_NOx_NO2eq"] = (
        data.EI_standard_LHV_g_per_MJ_NO * (46.0055 / 30.0061)
        + data.EI_standard_LHV_g_per_MJ_NO2
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, lineterminator="\n")
    print(data.loc[completed, "mechanism_to_standard_heat_ratio"].describe().to_string())
    print(args.output)


if __name__ == "__main__":
    main()
