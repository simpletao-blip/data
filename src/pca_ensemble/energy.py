"""Thermochemistry-based partial-cracking energy bookkeeping."""

from __future__ import annotations

import cantera as ct
import pandas as pd


def species_enthalpy_J_per_mol(
    gas: ct.Solution, species: str, temperature_K: float = 298.15
) -> float:
    gas.TPX = temperature_K, ct.one_atm, {species: 1.0}
    return float(gas.enthalpy_mole / 1000.0)


def cracking_energy_sensitivity(
    gas: ct.Solution,
    cracking_ratio: float,
    recovery_fractions: tuple[float, ...] = (0.0, 0.5, 0.8),
    reference_temperature_K: float = 298.15,
) -> pd.DataFrame:
    """Return energy terms per mole of NH3 before ideal partial cracking.

    The combustion product water is gaseous, so the heat release is an LHV-like
    reference quantity. Recovered heat is treated only as a sensitivity credit;
    no balance-of-plant efficiency is implied.
    """
    if not 0.0 <= cracking_ratio <= 1.0:
        raise ValueError("cracking_ratio must be in [0, 1]")
    if any(not 0.0 <= fraction <= 1.0 for fraction in recovery_fractions):
        raise ValueError("recovery fractions must be in [0, 1]")
    h = {name: species_enthalpy_J_per_mol(gas, name, reference_temperature_K)
         for name in ("NH3", "H2", "N2", "O2", "H2O")}
    alpha = cracking_ratio
    cracking_heat = alpha * (1.5 * h["H2"] + 0.5 * h["N2"] - h["NH3"])
    cracked_reactants = (
        (1.0 - alpha) * h["NH3"] + 1.5 * alpha * h["H2"]
        + 0.5 * alpha * h["N2"] + 0.75 * h["O2"]
    )
    products = 0.5 * h["N2"] + 1.5 * h["H2O"]
    gross_lhv = cracked_reactants - products
    rows = []
    for recovery in recovery_fractions:
        external_heat = (1.0 - recovery) * cracking_heat
        rows.append({
            "cracking_ratio": alpha,
            "heat_recovery_fraction": recovery,
            "cracking_heat_J_per_mol_initial_NH3": cracking_heat,
            "gross_LHV_like_J_per_mol_initial_NH3": gross_lhv,
            "external_cracking_heat_J_per_mol_initial_NH3": external_heat,
            "net_after_external_heat_J_per_mol_initial_NH3": gross_lhv - external_heat,
            "reference_temperature_K": reference_temperature_K,
            "water_state": "gas",
        })
    return pd.DataFrame(rows)

