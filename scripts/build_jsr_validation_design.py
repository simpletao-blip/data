"""Build a campaign-balanced JSR species-validation design from ReSpecTh."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.design import farthest_point_indices


ALLOWED_INLET = {"NH3", "H2", "O2", "N2", "AR", "HE"}


def clean_composition(raw: str) -> dict[str, float]:
    values = {str(key).upper(): float(value) for key, value in json.loads(raw).items()}
    return {key: value for key, value in values.items() if value > 0.0}


def main() -> None:
    data = pd.read_csv("data/processed/respecth_nh3_long.csv")
    jsr = data[data.experiment_type.eq("jet stirred reactor measurement")].copy()
    jsr = jsr.dropna(subset=["temperature_K", "pressure_Pa", "residence_time_s",
                             "initial_composition", "value"])
    jsr["condition_id"] = jsr.dataset_id.str.rsplit(":", n=1).str[0]
    jsr["composition_dict"] = jsr.initial_composition.map(clean_composition)
    jsr = jsr[jsr.composition_dict.map(lambda value: set(value).issubset(ALLOWED_INLET))].copy()
    conditions = (jsr.sort_values("dataset_id").drop_duplicates("condition_id")
                  [["condition_id", "campaign_id", "temperature_K", "pressure_Pa",
                    "residence_time_s", "composition_dict"]].copy())
    conditions["X_NH3"] = conditions.composition_dict.map(lambda x: x.get("NH3", 0.0))
    conditions["X_H2"] = conditions.composition_dict.map(lambda x: x.get("H2", 0.0))
    conditions["X_O2"] = conditions.composition_dict.map(lambda x: x.get("O2", 0.0))
    chosen = []
    features = ["temperature_K", "pressure_Pa", "residence_time_s", "X_NH3", "X_H2", "X_O2"]
    for _, campaign in conditions.groupby("campaign_id", sort=True):
        values = campaign[features].to_numpy(float)
        values[:, 1:3] = np.log10(values[:, 1:3])
        low = values.min(axis=0)
        span = values.max(axis=0) - low
        span[span == 0] = 1.0
        indexes = farthest_point_indices((values - low) / span, 8)
        chosen.extend(campaign.iloc[indexes].condition_id.tolist())
    design = jsr[jsr.condition_id.isin(chosen)].copy()
    design["initial_composition"] = design.composition_dict.map(
        lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"))
    )
    design = design.drop(columns="composition_dict").sort_values(
        ["campaign_id", "condition_id", "observable"]
    ).reset_index(drop=True)
    path = Path("data/processed/jsr_validation_design.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(path, index=False)
    print(f"conditions={design.condition_id.nunique()} rows={len(design)} ")
    print(design.groupby("campaign_id").agg(
        conditions=("condition_id", "nunique"), rows=("dataset_id", "size")
    ).to_string())
    print(path)


if __name__ == "__main__":
    main()
