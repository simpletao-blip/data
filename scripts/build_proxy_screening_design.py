"""Build a common grid for paired combustion-side proxy screening.

The flame/reactor inlet temperature and the ignition reference temperature are
deliberately separate.  A row is a paired comparison, not one physical reactor
state.  Surrogate-screened Pareto finalists must be rerun with Cantera.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.design import convex_hull_membership
from pca_ensemble.io import load_yaml


FEATURES = ["temperature_K", "pressure_bar", "equivalence_ratio", "cracking_ratio"]


def membership(reference: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    ref = reference[FEATURES].dropna().to_numpy(float)
    points = query[FEATURES].to_numpy(float)
    ref[:, 1] = np.log10(ref[:, 1])
    points[:, 1] = np.log10(points[:, 1])
    return convex_hull_membership(ref, points)


def main() -> None:
    config = load_yaml("config/study.yaml")
    composition = config["composition"]
    temperature_pairs = [(300.0, 600.0), (450.0, 750.0)]
    rows = pd.DataFrame(
        product(
            temperature_pairs,
            config["reactor_network"]["pressures_bar"],
            composition["equivalence_ratios"],
            composition["cracking_ratios"],
        ),
        columns=["temperature_pair", "pressure_bar", "equivalence_ratio", "cracking_ratio"],
    )
    rows[["flame_temperature_K", "temperature_K"]] = pd.DataFrame(
        rows.pop("temperature_pair").tolist(), index=rows.index
    )
    rows.insert(0, "design_id", [f"screen_{i:04d}" for i in range(1, len(rows) + 1)])
    rows.insert(1, "design_role", "surrogate_screening")
    rows["ignition_temperature_K"] = 1200.0
    rows["residence_time_ms"] = 10.0
    rows["psr_fraction"] = 0.5
    rows["heat_loss_W_per_K"] = 0.0

    data = pd.read_csv("data/processed/respecth_nh3_long.csv")
    lbv = data[data.observable.eq("laminar burning velocity")].copy()
    lbv["pressure_bar"] = lbv.pressure_Pa / 1e5
    lbv_query = rows.copy()
    lbv_query["temperature_K"] = lbv_query.flame_temperature_K
    rows["inside_lbv_convex_hull"] = membership(lbv, lbv_query)

    idt = data[data.observable.eq("ignition delay")].copy()
    idt["pressure_bar"] = idt.pressure_Pa / 1e5
    idt_query = rows.copy()
    idt_query["temperature_K"] = idt_query.ignition_temperature_K
    rows["inside_all_idt_convex_hull"] = membership(idt, idt_query)
    exact = idt[idt.definition.str.contains("relative concentration", na=False)]
    rows["inside_exact_criterion_convex_hull"] = membership(exact, idt_query)

    output = Path("data/processed/proxy_screening_design.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False)
    print(rows.groupby(["inside_lbv_convex_hull", "inside_all_idt_convex_hull",
                        "inside_exact_criterion_convex_hull"]).size().to_string())
    print(output)


if __name__ == "__main__":
    main()
