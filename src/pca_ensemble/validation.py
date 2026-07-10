"""Experiment-matched validation helpers and explicit eligibility decisions."""

from __future__ import annotations

import json
from dataclasses import asdict
from time import perf_counter
from typing import Any

import cantera as ct
import numpy as np
import pandas as pd

from .metrics import ignition_log_error
from .reactors import simulate_reported_ignition


def parse_ignition_definition(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def ignition_support(raw: str, species_names: list[str]) -> tuple[bool, str]:
    definition = parse_ignition_definition(raw)
    kind = definition.get("type", "").lower()
    target = definition.get("target", "")
    if kind == "relative concentration":
        if target.upper() not in {name.upper() for name in species_names}:
            return False, f"target species {target!r} absent from mechanism"
        try:
            amount = float(definition.get("amount", "nan"))
        except ValueError:
            return False, "invalid relative concentration amount"
        if not 0.0 < amount < 1.0:
            return False, "relative concentration amount must be in (0, 1)"
        return True, "exact species-relative criterion"
    if target.upper() == "OH*":
        return False, "OH* optical criterion is not represented by a ground-state mechanism"
    return False, f"unsupported ignition definition: {raw}"


def validate_idt_table(
    data: pd.DataFrame,
    mechanism_path: str,
    mechanism_id: str,
    max_time_factor: float = 20.0,
    minimum_max_time_s: float = 0.01,
    maximum_max_time_s: float = 2.0,
) -> pd.DataFrame:
    """Validate all IDT rows while retaining unsupported and failed cases."""
    required = {"dataset_id", "observable", "value", "temperature_K", "pressure_Pa",
                "initial_composition", "definition", "campaign_id", "doi"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"validation table is missing columns: {sorted(missing)}")
    gas = ct.Solution(mechanism_path)
    rows: list[dict[str, Any]] = []
    subset = data[data.observable.eq("ignition delay")].copy()
    for record in subset.itertuples(index=False):
        supported, support_reason = ignition_support(record.definition, gas.species_names)
        base = {
            "dataset_id": record.dataset_id,
            "campaign_id": record.campaign_id,
            "doi": record.doi,
            "mechanism_id": mechanism_id,
            "mechanism_path": mechanism_path,
            "experimental_s": float(record.value),
            "temperature_K": float(record.temperature_K),
            "pressure_Pa": float(record.pressure_Pa),
            "definition": record.definition,
            "criterion_supported": supported,
            "support_reason": support_reason,
        }
        if not supported:
            rows.append({**base, "status": "skipped", "simulated_s": np.nan,
                         "absolute_log10_error": np.nan, "runtime_s": 0.0,
                         "converged": False, "failure_reason": support_reason})
            continue
        definition = parse_ignition_definition(record.definition)
        max_time = min(max(float(record.value) * max_time_factor, minimum_max_time_s),
                       maximum_max_time_s)
        start = perf_counter()
        result = simulate_reported_ignition(
            gas,
            float(record.temperature_K),
            float(record.pressure_Pa),
            json.loads(record.initial_composition),
            reactor="constant_volume",
            criterion="species_relative",
            target_species=definition["target"],
            relative_amount=float(definition["amount"]),
            max_time_s=max_time,
        )
        runtime = perf_counter() - start
        simulated = result.ignition_delay_s
        log_ratio = float(np.log10(simulated / float(record.value))) if result.converged else np.nan
        error = (float(ignition_log_error(np.array([simulated]), np.array([record.value]))[0])
                 if result.converged else np.nan)
        rows.append({
            **base,
            **{f"solver_{key}": value for key, value in asdict(result).items()
               if key not in {"ignition_delay_s", "converged", "failure_reason"}},
            "status": "completed" if result.converged else "failed",
            "simulated_s": simulated,
            "prediction_ratio": simulated / float(record.value) if result.converged else np.nan,
            "signed_log10_residual": log_ratio,
            "absolute_log10_error": error,
            "runtime_s": runtime,
            "converged": result.converged,
            "failure_reason": result.failure_reason,
        })
    return pd.DataFrame(rows)
