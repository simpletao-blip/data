"""Build a deterministic 0D ignition-map design with two validation hulls."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from pca_ensemble.design import convex_hull_membership, select_operating_design
from pca_ensemble.io import load_yaml


def hull_membership(reference: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    features = ["temperature_K", "pressure_bar", "equivalence_ratio", "cracking_ratio"]
    ref = reference[features].dropna().to_numpy(float)
    ref[:, 1] = np.log10(ref[:, 1])
    points = query[features].to_numpy(float)
    points[:, 1] = np.log10(points[:, 1])
    return convex_hull_membership(ref, points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/study.yaml"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/respecth_nh3_long.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/idt_operating_design.csv"))
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--holdout-count", type=int, default=48)
    args = parser.parse_args()
    config = load_yaml(args.config)
    composition = config["composition"]
    ignition = config["ignition"]
    candidates = pd.DataFrame(
        product(ignition["temperatures_K"], ignition["pressures_bar"],
                composition["equivalence_ratios"], composition["cracking_ratios"]),
        columns=["temperature_K", "pressure_bar", "equivalence_ratio", "cracking_ratio"],
    )
    data = pd.read_csv(args.validation)
    idt = data[data.observable.eq("ignition delay")].copy()
    idt["pressure_bar"] = idt.pressure_Pa / 1e5
    design = select_operating_design(candidates, idt, args.train_count, args.holdout_count)
    design = design.rename(columns={"inside_experimental_convex_hull": "inside_all_idt_convex_hull"})
    exact = idt[idt.definition.str.contains("relative concentration", na=False)]
    design["inside_exact_criterion_convex_hull"] = hull_membership(exact, design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output, index=False)
    print(design.groupby(["design_role", "inside_all_idt_convex_hull",
                          "inside_exact_criterion_convex_hull"]).size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()

