"""Build deterministic 1D map and interpolation-holdout designs."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from pca_ensemble.design import select_operating_design
from pca_ensemble.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/study.yaml"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/respecth_nh3_long.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lbv_operating_design.csv"))
    parser.add_argument("--train-count", type=int, default=120)
    parser.add_argument("--holdout-count", type=int, default=24)
    args = parser.parse_args()
    config = load_yaml(args.config)
    composition = config["composition"]
    flame = config["flame"]
    candidates = pd.DataFrame(
        product(flame["unburned_temperatures_K"], flame["pressures_bar"],
                composition["equivalence_ratios"], composition["cracking_ratios"]),
        columns=["temperature_K", "pressure_bar", "equivalence_ratio", "cracking_ratio"],
    )
    validation = pd.read_csv(args.validation)
    validation = validation[validation.observable.eq("laminar burning velocity")].copy()
    validation["pressure_bar"] = validation.pressure_Pa / 1e5
    design = select_operating_design(candidates, validation, args.train_count, args.holdout_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output, index=False)
    print(design.groupby(["design_role", "inside_experimental_convex_hull"]).size().to_string())
    print(args.output)


if __name__ == "__main__":
    main()

