"""Create a minimally normalized Chemkin copy of the Shrestha 2018 mechanism.

The official 2020 format-update file declares the argon species as ``AR`` but
uses ``Ar`` in one reaction. Chemkin is commonly case insensitive, whereas
Cantera species names are case sensitive. The raw source is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text(encoding="latin-1")
    old = "OH*+Ar=OH+Ar"
    new = "OH*+AR=OH+AR"
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"expected one exact argon reaction token, found {count}")

    normalized = source.replace(old, new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Convert the derived copy to UTF-8 so the generated YAML can be reloaded
    # by Cantera. The official source uses a legacy single-byte encoding.
    args.output.write_text(normalized, encoding="utf-8", newline="")
    print(f"normalized_replacements={count}")
    print(args.output)


if __name__ == "__main__":
    main()
