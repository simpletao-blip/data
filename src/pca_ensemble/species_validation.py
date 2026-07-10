"""Apparatus-matched species-validation reactor utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cantera as ct

from .reactors import set_reported_mixture_state


@dataclass(frozen=True)
class JSRResult:
    temperature_K: float
    pressure_Pa: float
    steps: int
    converged: bool
    failure_reason: str
    mole_fractions: dict[str, float]
    solver_path: str


def simulate_isothermal_jsr(
    gas: ct.Solution,
    temperature_K: float,
    pressure_Pa: float,
    composition: Mapping[str, float],
    residence_time_s: float,
    volume_m3: float = 1.0e-4,
    max_steps: int = 100000,
) -> JSRResult:
    """Run an isothermal stirred reactor with pressure-controlled outlet."""
    if residence_time_s <= 0.0 or volume_m3 <= 0.0:
        raise ValueError("residence time and volume must be positive")
    set_reported_mixture_state(gas, temperature_K, pressure_Pa, composition)
    inlet = ct.Reservoir(gas, clone=True)
    reactor = ct.IdealGasReactor(gas, energy="off", volume=volume_m3, clone=True)
    exhaust = ct.Reservoir(gas, clone=True)
    mass_flow = ct.MassFlowController(inlet, reactor, mdot=lambda _: reactor.mass / residence_time_s)
    ct.PressureController(reactor, exhaust, primary=mass_flow, K=1e-5)
    network = ct.ReactorNet([reactor])
    solver_path = "advance_to_steady_state"
    try:
        network.advance_to_steady_state(max_steps=max_steps)
    except Exception as steady_exc:
        solver_path = "100_residence_time_fallback"
        try:
            network.advance(network.time + 100.0 * residence_time_s)
        except Exception as fallback_exc:
            return JSRResult(float(reactor.T), float(reactor.phase.P),
                             int(network.solver_stats.get("steps", 0)), False,
                             f"steady={type(steady_exc).__name__}: {steady_exc}; "
                             f"fallback={type(fallback_exc).__name__}: {fallback_exc}", {},
                             solver_path)
    fractions = {
        name: float(value) for name, value in zip(reactor.phase.species_names,
                                                  reactor.phase.X, strict=True)
    }
    return JSRResult(float(reactor.T), float(reactor.phase.P),
                     int(network.solver_stats.get("steps", 0)), True, "", fractions, solver_path)
