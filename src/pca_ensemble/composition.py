"""Fuel-composition utilities with explicit atom-balanced cracking semantics."""

from __future__ import annotations

from collections.abc import Mapping


def cracked_ammonia_fuel(alpha: float, basis_mol_nh3: float = 1.0) -> dict[str, float]:
    """Return the ideal partial-cracking products on an initial NH3 basis.

    NH3 -> 1.5 H2 + 0.5 N2. ``alpha`` is the fraction of the initial NH3
    molecules cracked, not the H2 mole fraction in the final fuel stream.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("cracking ratio alpha must be in [0, 1]")
    if basis_mol_nh3 <= 0.0:
        raise ValueError("basis_mol_nh3 must be positive")
    return {
        "NH3": basis_mol_nh3 * (1.0 - alpha),
        "H2": basis_mol_nh3 * 1.5 * alpha,
        "N2": basis_mol_nh3 * 0.5 * alpha,
    }


def normalized_composition(composition: Mapping[str, float]) -> dict[str, float]:
    """Normalize a non-negative composition mapping to mole fractions."""
    if any(value < 0.0 for value in composition.values()):
        raise ValueError("composition cannot contain negative amounts")
    total = float(sum(composition.values()))
    if total <= 0.0:
        raise ValueError("composition total must be positive")
    return {species: float(value) / total for species, value in composition.items()}


def ammonia_atom_balance(alpha: float) -> tuple[float, float]:
    """Return H and N atom totals for a one-mole initial NH3 basis."""
    fuel = cracked_ammonia_fuel(alpha)
    hydrogen_atoms = 3.0 * fuel["NH3"] + 2.0 * fuel["H2"]
    nitrogen_atoms = fuel["NH3"] + 2.0 * fuel["N2"]
    return hydrogen_atoms, nitrogen_atoms

