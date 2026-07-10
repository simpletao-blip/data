import pytest

from pca_ensemble.reactor_network import dry_mole_fractions, oxygen_corrected_dry_fraction


def test_dry_basis_excludes_water_and_renormalizes():
    dry = dry_mole_fractions({"NO": 0.01, "O2": 0.09, "H2O": 0.10})
    assert "H2O" not in dry
    assert dry["NO"] == pytest.approx(0.01 / 0.9)


def test_oxygen_correction_identity_at_reference():
    assert oxygen_corrected_dry_fraction(100e-6, 0.15, 0.15) == pytest.approx(100e-6)

