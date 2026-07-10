"""Export cracking-energy sensitivity on one declared thermochemical basis."""

from pathlib import Path

import cantera as ct
import pandas as pd

from pca_ensemble.energy import cracking_energy_sensitivity
from pca_ensemble.io import load_yaml, sha256


config = load_yaml("config/study.yaml")
mechanism = Path("mechanisms/raw/POLIMI_2023.yaml")
gas = ct.Solution(str(mechanism))
frames = []
for alpha in config["composition"]["cracking_ratios"]:
    frame = cracking_energy_sensitivity(
        gas, alpha, tuple(config["energy"]["heat_recovery_fractions"])
    )
    frame["thermochemical_basis"] = "POLIMI_2023"
    frame["mechanism_sha256"] = sha256(mechanism)
    frames.append(frame)
output = pd.concat(frames, ignore_index=True)
path = Path("results/processed/cracking_energy_sensitivity.csv")
path.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(path, index=False)
print(output.to_string(index=False))
print(path)
