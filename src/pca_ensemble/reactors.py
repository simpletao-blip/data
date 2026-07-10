"""Cantera reactor and flame models with explicit numerical metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import cantera as ct
import numpy as np

from .composition import cracked_ammonia_fuel


@dataclass(frozen=True)
class IgnitionResult:
    ignition_delay_s: float
    criterion: str
    reactor: str
    final_temperature_K: float
    max_temperature_K: float
    integration_steps: int
    converged: bool
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlameResult:
    laminar_burning_velocity_m_per_s: float
    max_temperature_K: float
    grid_points: int
    transport_model: str
    soret_enabled: bool
    converged: bool
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_cracked_ammonia_state(
    gas: ct.Solution,
    temperature_K: float,
    pressure_Pa: float,
    equivalence_ratio: float,
    cracking_ratio: float,
    oxidizer: dict[str, float] | None = None,
) -> None:
    if temperature_K <= 0.0 or pressure_Pa <= 0.0:
        raise ValueError("temperature and pressure must be positive")
    if equivalence_ratio <= 0.0:
        raise ValueError("equivalence ratio must be positive")
    fuel = cracked_ammonia_fuel(cracking_ratio)
    oxidizer = oxidizer or {"O2": 1.0, "N2": 3.76}
    gas.TP = temperature_K, pressure_Pa
    gas.set_equivalence_ratio(equivalence_ratio, fuel, oxidizer)


def set_reported_mixture_state(
    gas: ct.Solution,
    temperature_K: float,
    pressure_Pa: float,
    composition: Mapping[str, float],
) -> None:
    """Set a reported experimental mixture without reconstructing phi or alpha."""
    if temperature_K <= 0.0 or pressure_Pa <= 0.0:
        raise ValueError("temperature and pressure must be positive")
    reported = {str(name): float(value) for name, value in composition.items() if float(value) > 0.0}
    if not reported:
        raise ValueError("composition must contain at least one positive amount")
    by_upper = {name.upper(): name for name in gas.species_names}
    missing = sorted(name for name in reported if name.upper() not in by_upper)
    if missing:
        raise ValueError(f"mechanism is missing reported species: {missing}")
    clean = {by_upper[name.upper()]: value for name, value in reported.items()}
    gas.TPX = temperature_K, pressure_Pa, clean


def simulate_ignition(
    gas: ct.Solution,
    temperature_K: float,
    pressure_Pa: float,
    equivalence_ratio: float,
    cracking_ratio: float,
    reactor: Literal["constant_volume", "constant_pressure"] = "constant_volume",
    criterion: Literal["max_dTdt", "OH_peak"] = "max_dTdt",
    max_time_s: float = 5.0,
    max_steps: int = 200000,
) -> IgnitionResult:
    """Simulate homogeneous ignition and return a criterion-specific delay."""
    set_cracked_ammonia_state(
        gas, temperature_K, pressure_Pa, equivalence_ratio, cracking_ratio
    )
    if reactor == "constant_volume":
        vessel = ct.IdealGasReactor(gas, energy="on", clone=True)
    elif reactor == "constant_pressure":
        vessel = ct.IdealGasConstPressureReactor(gas, energy="on", clone=True)
    else:
        raise ValueError(f"unsupported reactor: {reactor}")
    network = ct.ReactorNet([vessel])
    times = [0.0]
    temperatures = [float(vessel.T)]
    oh = [float(vessel.thermo["OH"].X[0]) if "OH" in vessel.thermo.species_names else np.nan]
    try:
        while network.time < max_time_s and len(times) < max_steps:
            time = float(network.step())
            times.append(time)
            temperatures.append(float(vessel.T))
            oh.append(
                float(vessel.thermo["OH"].X[0])
                if "OH" in vessel.thermo.species_names
                else np.nan
            )
    except Exception as exc:
        return IgnitionResult(
            np.nan, criterion, reactor, float(vessel.T), max(temperatures),
            len(times) - 1, False, f"{type(exc).__name__}: {exc}",
        )
    time_array = np.asarray(times)
    temperature_array = np.asarray(temperatures)
    if criterion == "max_dTdt":
        if len(time_array) < 3 or temperature_array.max() - temperature_array[0] < 10.0:
            return IgnitionResult(
                np.nan, criterion, reactor, float(vessel.T), float(temperature_array.max()),
                len(times) - 1, False, "temperature rise below 10 K or insufficient steps",
            )
        gradient = np.gradient(temperature_array, time_array)
        index = int(np.nanargmax(gradient))
    elif criterion == "OH_peak":
        if np.isnan(oh).all():
            return IgnitionResult(
                np.nan, criterion, reactor, float(vessel.T), float(temperature_array.max()),
                len(times) - 1, False, "OH is absent from mechanism",
            )
        index = int(np.nanargmax(np.asarray(oh)))
    else:
        raise ValueError(f"unsupported ignition criterion: {criterion}")
    delay = float(time_array[index])
    converged = 0.0 < delay < max_time_s
    return IgnitionResult(
        delay if converged else np.nan,
        criterion,
        reactor,
        float(vessel.T),
        float(temperature_array.max()),
        len(times) - 1,
        converged,
        "" if converged else "criterion maximum occurred at integration boundary",
    )


def simulate_reported_ignition(
    gas: ct.Solution,
    temperature_K: float,
    pressure_Pa: float,
    composition: Mapping[str, float],
    reactor: Literal["constant_volume", "constant_pressure"] = "constant_volume",
    criterion: Literal["max_dTdt", "OH_peak", "species_relative"] = "max_dTdt",
    max_time_s: float = 5.0,
    max_steps: int = 200000,
    target_species: str | None = None,
    relative_amount: float | None = None,
) -> IgnitionResult:
    """Simulate a reported mixture with an explicitly matched ignition criterion."""
    set_reported_mixture_state(gas, temperature_K, pressure_Pa, composition)
    if criterion == "species_relative":
        if not target_species or target_species not in gas.species_names:
            raise ValueError("species_relative requires a target species present in the mechanism")
        if relative_amount is None or not 0.0 < relative_amount < 1.0:
            raise ValueError("species_relative requires relative_amount in (0, 1)")
        initial_target = float(gas[target_species].X[0])
    else:
        initial_target = np.nan
    if reactor == "constant_volume":
        vessel = ct.IdealGasReactor(gas, energy="on", clone=True)
    elif reactor == "constant_pressure":
        vessel = ct.IdealGasConstPressureReactor(gas, energy="on", clone=True)
    else:
        raise ValueError(f"unsupported reactor: {reactor}")
    network = ct.ReactorNet([vessel])
    times = [0.0]
    temperatures = [float(vessel.T)]
    oh = [float(vessel.thermo["OH"].X[0]) if "OH" in vessel.thermo.species_names else np.nan]
    targets = [initial_target]
    try:
        while network.time < max_time_s and len(times) < max_steps:
            times.append(float(network.step()))
            temperatures.append(float(vessel.T))
            oh.append(float(vessel.thermo["OH"].X[0])
                      if "OH" in vessel.thermo.species_names else np.nan)
            targets.append(float(vessel.thermo[target_species].X[0])
                           if target_species else np.nan)
            if criterion == "species_relative" and targets[-1] <= relative_amount * initial_target:
                break
    except Exception as exc:
        return IgnitionResult(
            np.nan, criterion, reactor, float(vessel.T), max(temperatures),
            len(times) - 1, False, f"{type(exc).__name__}: {exc}",
        )
    time_array = np.asarray(times)
    temperature_array = np.asarray(temperatures)
    if criterion == "max_dTdt":
        if len(time_array) < 3 or temperature_array.max() - temperature_array[0] < 10.0:
            return IgnitionResult(
                np.nan, criterion, reactor, float(vessel.T), float(temperature_array.max()),
                len(times) - 1, False, "temperature rise below 10 K or insufficient steps",
            )
        index = int(np.nanargmax(np.gradient(temperature_array, time_array)))
    elif criterion == "OH_peak":
        if np.isnan(oh).all():
            return IgnitionResult(
                np.nan, criterion, reactor, float(vessel.T), float(temperature_array.max()),
                len(times) - 1, False, "OH is absent from mechanism",
            )
        index = int(np.nanargmax(np.asarray(oh)))
    elif criterion == "species_relative":
        reached = np.flatnonzero(np.asarray(targets) <= relative_amount * initial_target)
        if not len(reached):
            return IgnitionResult(
                np.nan, criterion, reactor, float(vessel.T), float(temperature_array.max()),
                len(times) - 1, False, "target relative concentration not reached",
            )
        index = int(reached[0])
    else:
        raise ValueError(f"unsupported ignition criterion: {criterion}")
    delay = float(time_array[index])
    converged = 0.0 < delay < max_time_s
    return IgnitionResult(
        delay if converged else np.nan, criterion, reactor, float(vessel.T),
        float(temperature_array.max()), len(times) - 1, converged,
        "" if converged else "criterion maximum occurred at integration boundary",
    )


def simulate_free_flame(
    gas: ct.Solution,
    temperature_K: float,
    pressure_Pa: float,
    equivalence_ratio: float,
    cracking_ratio: float,
    width_m: float = 0.05,
    transport_model: str = "multicomponent",
    soret: bool = True,
    refine: dict[str, float] | None = None,
    loglevel: int = 0,
) -> FlameResult:
    """Solve a freely propagating flame; failures are returned, never hidden."""
    set_cracked_ammonia_state(
        gas, temperature_K, pressure_Pa, equivalence_ratio, cracking_ratio
    )
    flame = ct.FreeFlame(gas, width=width_m)
    refine = refine or {"ratio": 3.0, "slope": 0.06, "curve": 0.10}
    flame.set_refine_criteria(**refine)
    try:
        _solve_free_flame_staged(flame, transport_model, soret, loglevel)
    except Exception as exc:
        return FlameResult(
            np.nan, np.nan, len(flame.grid), transport_model, bool(soret),
            False, f"{type(exc).__name__}: {exc}",
        )
    return FlameResult(
        float(flame.velocity[0]),
        float(np.max(flame.T)),
        len(flame.grid),
        str(flame.transport_model),
        bool(flame.soret_enabled),
        True,
    )


def simulate_reported_free_flame(
    gas: ct.Solution,
    temperature_K: float,
    pressure_Pa: float,
    composition: Mapping[str, float],
    width_m: float = 0.05,
    transport_model: str = "multicomponent",
    soret: bool = True,
    refine: dict[str, float] | None = None,
    loglevel: int = 0,
) -> FlameResult:
    """Solve a flame from the reported inlet mixture without reconstructing it."""
    set_reported_mixture_state(gas, temperature_K, pressure_Pa, composition)
    flame = ct.FreeFlame(gas, width=width_m)
    refine = refine or {"ratio": 3.0, "slope": 0.06, "curve": 0.10}
    flame.set_refine_criteria(**refine)
    try:
        _solve_free_flame_staged(flame, transport_model, soret, loglevel)
    except Exception as exc:
        return FlameResult(
            np.nan, np.nan, len(flame.grid), transport_model, bool(soret),
            False, f"{type(exc).__name__}: {exc}",
        )
    return FlameResult(
        float(flame.velocity[0]), float(np.max(flame.T)), len(flame.grid),
        str(flame.transport_model), bool(flame.soret_enabled), True,
    )


def _solve_free_flame_staged(
    flame: ct.FreeFlame,
    transport_model: str,
    soret: bool,
    loglevel: int,
) -> None:
    """Build a robust initial flame before enabling costly transport physics."""
    if transport_model == "multicomponent":
        flame.transport_model = "mixture-averaged"
        flame.soret_enabled = False
        flame.solve(loglevel=loglevel, auto=True)
        flame.transport_model = "multicomponent"
        flame.solve(loglevel=loglevel, auto=False)
        if soret:
            flame.soret_enabled = True
            flame.solve(loglevel=loglevel, auto=False)
        return
    flame.transport_model = transport_model
    flame.soret_enabled = bool(soret)
    flame.solve(loglevel=loglevel, auto=True)
