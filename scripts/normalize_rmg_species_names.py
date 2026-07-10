"""Strip RMG numeric species suffixes in a derived Cantera YAML copy.

RMG exports species such as ``NH3(1)`` and ``O2(3)``. The project validation
tables use conventional names (``NH3`` and ``O2``). This script preserves the
official source and rewrites only the derived computational copy after checking
that suffix removal cannot create duplicate species names.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import cantera as ct


SUFFIX = re.compile(r"\(\d+\)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    gas = ct.Solution(str(args.input))
    mapping = {name: SUFFIX.sub("", name) for name in gas.species_names}
    normalized = list(mapping.values())
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise SystemExit(f"suffix removal creates duplicate species: {duplicates}")
    changed = {old: new for old, new in mapping.items() if old != new}
    if not changed:
        raise SystemExit("no RMG numeric species suffixes found")

    text = args.input.read_text(encoding="utf-8")
    for old in sorted(changed, key=len, reverse=True):
        text = text.replace(old, changed[old])

    used_elements = [
        element
        for element in gas.element_names
        if any(species.composition.get(element, 0.0) for species in gas.species())
    ]
    phase_elements = re.compile(r"(?ms)^  elements:\s*\[[^\]]*\]\n")
    text, phase_count = phase_elements.subn(
        f"  elements: [{', '.join(used_elements)}]\n", text, count=1
    )
    if phase_count != 1:
        raise SystemExit(f"expected one phase element list, found {phase_count}")
    # The exported RMG YAML defines isotope/pseudo-element weights that are
    # unused by every species and fail when Cantera clones the phase into a
    # reactor. They are removed only from this derived copy.
    custom_elements = re.compile(r"(?ms)^elements:\n.*?(?=^species:\n)")
    text = custom_elements.sub("", text, count=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="")

    check = ct.Solution(str(args.output))
    missing = sorted(set(normalized).difference(check.species_names))
    if missing or check.n_species != gas.n_species or check.n_reactions != gas.n_reactions:
        raise SystemExit(
            f"normalization verification failed: missing={missing}, "
            f"species={check.n_species}/{gas.n_species}, "
            f"reactions={check.n_reactions}/{gas.n_reactions}"
        )
    print(f"renamed_species={len(changed)}")
    print(f"retained_elements={','.join(used_elements)}")
    print(args.output)


if __name__ == "__main__":
    main()
