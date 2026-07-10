"""Run one bounded PSR-PFR feasibility case."""

import cantera as ct

from pca_ensemble.reactor_network import simulate_psr_pfr


gas = ct.Solution("mechanisms/raw/POLIMI_2023.yaml")
result = simulate_psr_pfr(
    gas, temperature_K=750.0, pressure_Pa=10.0e5, equivalence_ratio=1.0,
    cracking_ratio=0.3, total_residence_time_s=0.020, psr_fraction=0.5,
)
print(result.to_dict())
