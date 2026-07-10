"""Mechanism loading and reproducibility checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import cantera as ct


@dataclass(frozen=True)
class MechanismAudit:
    path: str
    phase: str
    species_count: int
    reaction_count: int
    duplicate_equation_count: int
    transport_model: str
    multicomponent_transport_available: bool
    nonfinite_forward_rate_count: int
    negative_forward_rate_count: int
    declared_negative_A_count: int
    negative_equation_aggregate_count: int
    benchmark_composition: tuple[tuple[str, float], ...]
    elements: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_solution(path: str | Path, phase: str | None = None) -> ct.Solution:
    source = str(path)
    return ct.Solution(source, phase) if phase else ct.Solution(source)


def audit_mechanism(
    path: str | Path,
    phase: str | None = None,
    benchmark_temperature_K: float = 1200.0,
    benchmark_pressure_Pa: float = 10.0e5,
) -> MechanismAudit:
    gas = load_solution(path, phase)
    requested_composition = {"NH3": 1.0, "H2": 0.5, "O2": 1.25, "N2": 4.7}
    benchmark_composition = {
        species: amount
        for species, amount in requested_composition.items()
        if species in gas.species_names
    }
    if not benchmark_composition:
        benchmark_composition = {gas.species_names[0]: 1.0}
    gas.TPX = benchmark_temperature_K, benchmark_pressure_Pa, benchmark_composition
    equations = gas.reaction_equations()
    duplicate_count = sum(count - 1 for count in Counter(equations).values() if count > 1)
    rates = gas.forward_rate_constants
    nonfinite_rates = sum(not isfinite(float(value)) for value in rates)
    negative_rates = sum(isfinite(float(value)) and float(value) < 0.0 for value in rates)
    declared_negative_A = sum(
        bool(gas.reaction(index).input_data.get("negative-A", False))
        for index in range(gas.n_reactions)
    )
    aggregate_rates: dict[str, float] = {}
    for equation, rate in zip(equations, rates, strict=True):
        aggregate_rates[equation] = aggregate_rates.get(equation, 0.0) + float(rate)
    negative_aggregates = sum(
        isfinite(value) and value < 0.0 for value in aggregate_rates.values()
    )
    transport_available = True
    try:
        gas.transport_model = "multicomponent"
        _ = gas.mix_diff_coeffs
    except Exception:
        transport_available = False
    return MechanismAudit(
        path=str(path),
        phase=gas.name,
        species_count=gas.n_species,
        reaction_count=gas.n_reactions,
        duplicate_equation_count=duplicate_count,
        transport_model=str(gas.transport_model),
        multicomponent_transport_available=transport_available,
        nonfinite_forward_rate_count=nonfinite_rates,
        negative_forward_rate_count=negative_rates,
        declared_negative_A_count=declared_negative_A,
        negative_equation_aggregate_count=negative_aggregates,
        benchmark_composition=tuple(benchmark_composition.items()),
        elements=tuple(gas.element_names),
    )
