"""Explicitly bounded PSR-to-PFR surrogate for combustion-side emissions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cantera as ct
import numpy as np

from .reactors import set_cracked_ammonia_state
from .energy import species_enthalpy_J_per_mol


TRACKED_SPECIES = ("NO", "NO2", "N2O", "NH3", "H2", "O2", "H2O")


@dataclass(frozen=True)
class ReactorNetworkResult:
    total_residence_time_s: float
    psr_fraction: float
    heat_loss_W_per_K: float
    ambient_temperature_K: float
    psr_temperature_K: float
    outlet_temperature_K: float
    psr_steps: int
    pfr_steps: int
    converged: bool
    failure_reason: str
    outlet_mole_fractions: dict[str, float]
    outlet_dry_mole_fractions: dict[str, float]
    outlet_mass_fractions: dict[str, float]
    steady_mass_flow_kg_per_s: float
    fuel_heat_input_W: float
    emission_indices_g_per_MJ: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dry_mole_fractions(mole_fractions: dict[str, float]) -> dict[str, float]:
    water = float(mole_fractions.get("H2O", 0.0))
    denominator = 1.0 - water
    if denominator <= 0.0:
        raise ValueError("dry-basis conversion requires H2O mole fraction below one")
    return {name: float(value) / denominator for name, value in mole_fractions.items() if name != "H2O"}


def oxygen_corrected_dry_fraction(
    dry_fraction: float,
    measured_o2_dry: float,
    reference_o2_dry: float,
    atmospheric_o2: float = 0.209,
) -> float:
    """Apply an explicitly parameterized oxygen correction on a dry basis."""
    if not 0.0 <= reference_o2_dry < atmospheric_o2:
        raise ValueError("reference O2 must be in [0, atmospheric O2)")
    if not 0.0 <= measured_o2_dry < atmospheric_o2:
        raise ValueError("measured dry O2 must be in [0, atmospheric O2)")
    return float(dry_fraction) * (atmospheric_o2 - reference_o2_dry) / (
        atmospheric_o2 - measured_o2_dry
    )


def _tracked(phase: ct.Solution) -> dict[str, float]:
    return {
        species: float(phase[species].X[0]) if species in phase.species_names else np.nan
        for species in TRACKED_SPECIES
    }


def _tracked_mass(phase: ct.Solution) -> dict[str, float]:
    return {
        species: float(phase[species].Y[0]) if species in phase.species_names else np.nan
        for species in TRACKED_SPECIES
    }


def _fuel_lhv_J_per_mol(gas: ct.Solution) -> tuple[float, float]:
    h = {name: species_enthalpy_J_per_mol(gas, name) for name in ("NH3", "H2", "N2", "O2", "H2O")}
    nh3 = h["NH3"] + 0.75 * h["O2"] - (0.5 * h["N2"] + 1.5 * h["H2O"])
    h2 = h["H2"] + 0.5 * h["O2"] - h["H2O"]
    return nh3, h2


def simulate_psr_pfr(
    gas: ct.Solution,
    temperature_K: float,
    pressure_Pa: float,
    equivalence_ratio: float,
    cracking_ratio: float,
    total_residence_time_s: float,
    psr_fraction: float = 0.5,
    heat_loss_W_per_K: float = 0.0,
    ambient_temperature_K: float = 300.0,
    reactor_volume_m3: float = 1.0e-3,
    steady_max_steps: int = 100000,
) -> ReactorNetworkResult:
    """Run an ignited PSR followed by a constant-pressure Lagrangian PFR.

    ``heat_loss_W_per_K`` is a lumped UA applied separately to each ideal
    reactor.  The result is a sensitivity surrogate, not a universal turbine
    combustor prediction.
    """
    if total_residence_time_s <= 0.0:
        raise ValueError("total residence time must be positive")
    if not 0.0 < psr_fraction < 1.0:
        raise ValueError("psr_fraction must be in (0, 1)")
    if heat_loss_W_per_K < 0.0 or reactor_volume_m3 <= 0.0:
        raise ValueError("heat loss must be non-negative and volume positive")
    psr_tau = total_residence_time_s * psr_fraction
    pfr_tau = total_residence_time_s - psr_tau
    lhv_nh3, lhv_h2 = _fuel_lhv_J_per_mol(gas)
    set_cracked_ammonia_state(gas, temperature_K, pressure_Pa, equivalence_ratio, cracking_ratio)
    inlet_mean_mw = float(gas.mean_molecular_weight)
    inlet_heat_J_per_kmol_mixture = 1000.0 * (
        float(gas["NH3"].X[0]) * lhv_nh3 + float(gas["H2"].X[0]) * lhv_h2
    )
    inlet = ct.Reservoir(gas, clone=True)
    gas.equilibrate("HP")
    psr = ct.IdealGasConstPressureReactor(
        gas, energy="on", volume=reactor_volume_m3, clone=True
    )
    exhaust = ct.Reservoir(gas, clone=True)
    mass_flow = ct.MassFlowController(inlet, psr, mdot=lambda _: psr.mass / psr_tau)
    ct.PressureController(psr, exhaust, primary=mass_flow, K=1e-5)
    if heat_loss_W_per_K > 0.0:
        gas.TPX = ambient_temperature_K, pressure_Pa, {"N2": 1.0}
        ambient = ct.Reservoir(gas, clone=True)
        ct.Wall(psr, ambient, A=1.0, U=heat_loss_W_per_K)
    network = ct.ReactorNet([psr])
    try:
        network.advance_to_steady_state(max_steps=steady_max_steps)
    except Exception as exc:
        return ReactorNetworkResult(
            total_residence_time_s, psr_fraction, heat_loss_W_per_K,
            ambient_temperature_K, float(psr.T), np.nan, network.solver_stats.get("steps", 0),
            0, False, f"PSR {type(exc).__name__}: {exc}", {}, {}, {}, np.nan, np.nan, {},
        )
    psr_steps = int(network.solver_stats.get("steps", 0))
    psr_temperature = float(psr.T)
    pfr = ct.IdealGasConstPressureReactor(
        psr.phase, energy="on", volume=reactor_volume_m3, clone=True
    )
    if heat_loss_W_per_K > 0.0:
        ct.Wall(pfr, ambient, A=1.0, U=heat_loss_W_per_K)
    pfr_network = ct.ReactorNet([pfr])
    try:
        pfr_network.advance(pfr_tau)
    except Exception as exc:
        return ReactorNetworkResult(
            total_residence_time_s, psr_fraction, heat_loss_W_per_K,
            ambient_temperature_K, psr_temperature, float(pfr.T), psr_steps,
            int(pfr_network.solver_stats.get("steps", 0)), False,
            f"PFR {type(exc).__name__}: {exc}", {}, {}, {}, np.nan, np.nan, {},
        )
    outlet = _tracked(pfr.phase)
    outlet_mass = _tracked_mass(pfr.phase)
    steady_mdot = float(mass_flow.mass_flow_rate)
    heat_input = steady_mdot / inlet_mean_mw * inlet_heat_J_per_kmol_mixture
    emission_indices = {
        species: float(outlet_mass[species] * steady_mdot * 1e9 / heat_input)
        for species in ("NO", "NO2", "N2O", "NH3")
        if np.isfinite(outlet_mass[species])
    }
    return ReactorNetworkResult(
        total_residence_time_s, psr_fraction, heat_loss_W_per_K,
        ambient_temperature_K, psr_temperature, float(pfr.T), psr_steps,
        int(pfr_network.solver_stats.get("steps", 0)), True, "", outlet,
        dry_mole_fractions(outlet), outlet_mass, steady_mdot, heat_input, emission_indices,
    )
