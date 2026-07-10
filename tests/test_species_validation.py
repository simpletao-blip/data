import cantera as ct

from pca_ensemble.species_validation import simulate_isothermal_jsr


def test_isothermal_jsr_smoke_with_bundled_gas():
    gas = ct.Solution("h2o2.yaml")
    result = simulate_isothermal_jsr(
        gas, 900.0, ct.one_atm, {"H2": 0.02, "O2": 0.01, "N2": 0.97}, 0.01,
        volume_m3=1e-5,
    )
    assert result.converged
    assert abs(result.temperature_K - 900.0) < 1e-8
    assert "H2" in result.mole_fractions
