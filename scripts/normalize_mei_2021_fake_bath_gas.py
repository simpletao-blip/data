"""Make the MEI 2021 artificial FN2 bath gas loadable in Cantera.

The authors define FN2 only as an optional third-body collider for a thermal-
effect study. Its Chemkin thermo record encodes one He atom while its transport
record declares a linear molecule, which Cantera rejects as inconsistent. This
derived copy changes only the unused FN2 transport geometry to ``atom``. The
official Chemkin files remain unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import cantera as ct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    # ck2yaml preserves a few Windows-1252 bytes from the authors' comments.
    text = args.input.read_text(encoding="cp1252")
    pattern = re.compile(
        r"(- name: FN2\n.*?\n  transport:\n    model: gas\n    geometry:) linear",
        flags=re.DOTALL,
    )
    normalized, count = pattern.subn(r"\1 atom", text, count=1)
    if count != 1:
        raise SystemExit(f"expected one FN2 transport record, found {count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(normalized, encoding="utf-8", newline="")
    gas = ct.Solution(str(args.output))
    participating = [
        reaction.equation
        for reaction in gas.reactions()
        if "FN2" in reaction.reactants or "FN2" in reaction.products
    ]
    if participating:
        raise SystemExit(f"FN2 unexpectedly participates stoichiometrically: {participating}")
    print(f"species={gas.n_species} reactions={gas.n_reactions}")
    print("FN2 is absent from all stoichiometric reactants and products")
    print(args.output)


if __name__ == "__main__":
    main()
