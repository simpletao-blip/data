"""Check numbered manuscript citations against the generated reference list."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "manuscript_working.md"
OUTPUT = ROOT / "results" / "logs" / "manuscript_citation_audit.csv"


def expand_group(group: str) -> set[int]:
    cited: set[int] = set()
    for part in re.split(r"\s*,\s*", group.replace("–", "-")):
        if re.fullmatch(r"\d+", part):
            cited.add(int(part))
        elif re.fullmatch(r"\d+-\d+", part):
            start, end = map(int, part.split("-"))
            cited.update(range(start, end + 1))
    return cited


def main() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    body, marker, references = text.partition("# References")
    if not marker:
        raise SystemExit("No References section found")
    reference_numbers = {
        int(match.group(1)) for match in re.finditer(r"(?m)^(\d+)\.\s", references)
    }
    cited: set[int] = set()
    for match in re.finditer(r"\[([0-9,\-–\s]+)\]", body):
        cited.update(expand_group(match.group(1)))
    rows = []
    for number in sorted(reference_numbers | cited):
        rows.append(
            {
                "reference_number": number,
                "exists_in_reference_list": number in reference_numbers,
                "cited_in_manuscript": number in cited,
            }
        )
    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(frame.to_string(index=False))
    if not frame[["exists_in_reference_list", "cited_in_manuscript"]].all().all():
        raise SystemExit(f"Citation audit failed; see {OUTPUT}")
    if reference_numbers != set(range(1, max(reference_numbers) + 1)):
        raise SystemExit("Reference numbering is not contiguous")
    print(f"All {len(reference_numbers)} references are present and cited.")


if __name__ == "__main__":
    main()
