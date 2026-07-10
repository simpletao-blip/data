"""Configuration and tabular I/O helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_EXPERIMENT_COLUMNS = {
    "dataset_id", "campaign_id", "doi", "laboratory", "apparatus",
    "observable", "value", "unit", "uncertainty", "uncertainty_type",
    "temperature_K", "pressure_Pa", "equivalence_ratio", "cracking_ratio",
    "fuel_composition", "oxidizer_composition", "definition",
    "source_location", "digitized", "quality_status", "exclusion_reason",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_experimental_table(frame: pd.DataFrame) -> None:
    missing = REQUIRED_EXPERIMENT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    included = frame[frame["quality_status"] == "included"]
    if included.empty:
        return
    if included[["dataset_id", "campaign_id", "observable", "value", "unit"]].isna().any().any():
        raise ValueError("included records contain missing identifiers or measurements")
    if not included["cracking_ratio"].between(0.0, 1.0).all():
        raise ValueError("included cracking ratios must be in [0, 1]")
    if not (included["pressure_Pa"] > 0.0).all() or not (included["temperature_K"] > 0.0).all():
        raise ValueError("included temperatures and pressures must be positive")
    for column in ("fuel_composition", "oxidizer_composition"):
        for raw in included[column]:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(parsed, dict) or not parsed:
                raise ValueError(f"{column} must contain a non-empty JSON mapping")

