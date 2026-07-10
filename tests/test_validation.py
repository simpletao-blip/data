import cantera as ct

from pca_ensemble.validation import ignition_support, parse_ignition_definition


def test_parse_and_support_species_relative_definition():
    raw = '{"amount": "0.95", "target": "NH3", "type": "relative concentration"}'
    gas = ct.Solution("mechanisms/raw/POLIMI_2023.yaml")
    assert parse_ignition_definition(raw)["target"] == "NH3"
    supported, reason = ignition_support(raw, gas.species_names)
    assert supported
    assert reason == "exact species-relative criterion"


def test_oh_star_is_not_claimed_as_exactly_supported():
    raw = '{"target": "OH*", "type": "baseline min intercept from d/dt"}'
    supported, reason = ignition_support(raw, ["OH", "NH3"])
    assert not supported
    assert "ground-state" in reason

