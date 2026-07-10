"""Build a bounded PSR-PFR map plus explicit thermal/residence sensitivities."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.design import farthest_point_indices
from pca_ensemble.io import load_yaml


config = load_yaml("config/study.yaml")
composition = config["composition"]
reactor = config["reactor_network"]

primary = pd.DataFrame(
    product([750], reactor["pressures_bar"], reactor["residence_times_ms"],
            composition["equivalence_ratios"], composition["cracking_ratios"], [0.5], [0.0]),
    columns=["temperature_K", "pressure_bar", "residence_time_ms", "equivalence_ratio",
             "cracking_ratio", "psr_fraction", "heat_loss_W_per_K"],
)
features = primary[["pressure_bar", "residence_time_ms", "equivalence_ratio", "cracking_ratio"]].to_numpy(float)
features[:, :2] = np.log10(features[:, :2])
features = (features - features.min(axis=0)) / np.where(
    features.max(axis=0) > features.min(axis=0), features.max(axis=0) - features.min(axis=0), 1.0
)
primary = primary.iloc[farthest_point_indices(features, 160)].copy()
primary["design_role"] = "primary_map"

anchors = [(0.1, 0.7), (0.3, 1.0), (0.5, 1.2), (0.7, 1.0)]
sensitivity_rows = []
for (alpha, phi), temperature, pressure, residence, heat_loss in product(
    anchors, reactor["inlet_temperatures_K"], [5, 20], [5, 20], reactor["heat_loss_cases_W_per_K"]
):
    sensitivity_rows.append({
        "temperature_K": temperature, "pressure_bar": pressure,
        "residence_time_ms": residence, "equivalence_ratio": phi,
        "cracking_ratio": alpha, "psr_fraction": 0.5,
        "heat_loss_W_per_K": heat_loss, "design_role": "thermal_residence_sensitivity",
    })
for (alpha, phi), fraction in product(anchors, reactor["psr_fractions"]):
    sensitivity_rows.append({
        "temperature_K": 750, "pressure_bar": 10, "residence_time_ms": 10,
        "equivalence_ratio": phi, "cracking_ratio": alpha, "psr_fraction": fraction,
        "heat_loss_W_per_K": 0.0, "design_role": "psr_fraction_sensitivity",
    })
sensitivity = pd.DataFrame(sensitivity_rows)
design = pd.concat([primary, sensitivity], ignore_index=True).drop_duplicates(
    ["temperature_K", "pressure_bar", "residence_time_ms", "equivalence_ratio",
     "cracking_ratio", "psr_fraction", "heat_loss_W_per_K"], keep="first"
).reset_index(drop=True)
design.insert(0, "design_id", [f"reactor_{index:04d}" for index in range(1, len(design) + 1)])
path = Path("data/processed/reactor_operating_design.csv")
path.parent.mkdir(parents=True, exist_ok=True)
design.to_csv(path, index=False)
print(design.groupby("design_role").size().to_string())
print(path)

