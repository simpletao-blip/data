import pytest
import cantera as ct

from pca_ensemble.composition import ammonia_atom_balance, cracked_ammonia_fuel
from pca_ensemble.reactors import set_reported_mixture_state


@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.3, 0.7, 1.0])
def test_cracking_preserves_atoms(alpha):
    hydrogen, nitrogen = ammonia_atom_balance(alpha)
    assert hydrogen == pytest.approx(3.0)
    assert nitrogen == pytest.approx(1.0)


def test_cracking_endpoints():
    assert cracked_ammonia_fuel(0.0) == {"NH3": 1.0, "H2": 0.0, "N2": 0.0}
    assert cracked_ammonia_fuel(1.0) == {"NH3": 0.0, "H2": 1.5, "N2": 0.5}


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_invalid_cracking_ratio(alpha):
    with pytest.raises(ValueError):
        cracked_ammonia_fuel(alpha)


def test_reported_mixture_state_preserves_ratios():
    gas = ct.Solution("mechanisms/raw/POLIMI_2023.yaml")
    mixture = {"NH3": 0.05, "H2": 0.05, "O2": 0.15, "N2": 0.70, "Ar": 0.05}
    set_reported_mixture_state(gas, 1200.0, 1.0e5, mixture)
    assert gas["H2"].X[0] / gas["NH3"].X[0] == pytest.approx(1.0)
    assert gas.T == pytest.approx(1200.0)
    assert gas.P == pytest.approx(1.0e5)
