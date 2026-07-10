"""Re-encode a derived ck2yaml file from Windows-1252 to UTF-8."""

from __future__ import annotations

import argparse
from pathlib import Path

import cantera as ct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text(encoding="cp1252")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="")
    gas = ct.Solution(str(args.output))
    print(f"species={gas.n_species} reactions={gas.n_reactions}")
    print(args.output)


if __name__ == "__main__":
    main()
