"""Verify that file paths named in Supplementary Information exist."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "manuscript" / "supplementary_information_draft.md"
OUTPUT = ROOT / "results" / "logs" / "supplementary_reference_audit.csv"
PATH_PATTERN = re.compile(r"`([A-Za-z0-9_.*-]+(?:[/\\][A-Za-z0-9_.*-]+)+)`")


def main() -> None:
    text = SI.read_text(encoding="utf-8")
    rows = []
    for token in sorted(set(PATH_PATTERN.findall(text))):
        for item in [part.strip() for part in token.split(",")]:
            normalized = item.replace("\\", "/")
            if "/" not in normalized:
                continue
            matches = list(ROOT.glob(normalized)) if "*" in normalized else [ROOT / normalized]
            exists = bool(matches) and all(path.exists() for path in matches)
            kinds = {"directory" if path.is_dir() else "file" if path.is_file() else "missing" for path in matches}
            rows.append(
                {
                    "reference": item,
                    "exists": exists,
                    "kind": "+".join(sorted(kinds)),
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False, lineterminator="\n")
    missing = frame.loc[~frame["exists"]]
    print(frame.to_string(index=False))
    if not missing.empty:
        raise SystemExit(f"Missing {len(missing)} referenced paths; see {OUTPUT}")
    print(f"All {len(frame)} Supplementary Information path references exist.")


if __name__ == "__main__":
    main()
