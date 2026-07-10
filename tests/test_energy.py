import cantera as ct
import pytest

from pca_ensemble.energy import cracking_energy_sensitivity


def test_zero_recovery_preserves_uncracked_net_energy():
    gas = ct.Solution("mechanisms/raw/POLIMI_2023.yaml")
    base = cracking_energy_sensitivity(gas, 0.0, (0.0,)).iloc[0]
    cracked = cracking_energy_sensitivity(gas, 0.5, (0.0, 0.8))
    no_recovery = cracked[cracked.heat_recovery_fraction.eq(0.0)].iloc[0]
    high_recovery = cracked[cracked.heat_recovery_fraction.eq(0.8)].iloc[0]
    assert no_recovery.net_after_external_heat_J_per_mol_initial_NH3 == pytest.approx(
        base.net_after_external_heat_J_per_mol_initial_NH3
    )
    assert high_recovery.net_after_external_heat_J_per_mol_initial_NH3 > no_recovery.net_after_external_heat_J_per_mol_initial_NH3

