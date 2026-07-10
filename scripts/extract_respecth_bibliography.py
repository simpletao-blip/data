"""Extract DOI-linked bibliography metadata from the archived ReSpecTh XML files."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


def clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_doi(value: str | None) -> str:
    doi = clean(value).lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi.rstrip(". ")


def extract(xml_root: Path) -> pd.DataFrame:
    rows: dict[str, dict[str, object]] = {}
    for path in sorted(xml_root.rglob("*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for link in root.findall(".//bibliographyLink"):
            doi = normalize_doi(link.findtext("referenceDOI"))
            details = link.find("details")
            if not doi or details is None:
                continue
            row = {
                "doi": doi,
                "authors": clean(details.findtext("author")),
                "title": clean(details.findtext("title")),
                "journal": clean(details.findtext("journal")),
                "year": clean(details.findtext("year")),
                "volume": clean(details.findtext("volume")),
                "pages_or_article": clean(details.findtext("pages")),
                "description": clean(link.findtext("description")),
                "source_xml_count": 1,
                "example_source_xml": path.relative_to(xml_root).as_posix(),
            }
            if doi not in rows:
                rows[doi] = row
            else:
                rows[doi]["source_xml_count"] = int(rows[doi]["source_xml_count"]) + 1
                for field in ("authors", "title", "journal", "year", "volume", "pages_or_article"):
                    if not rows[doi][field] and row[field]:
                        rows[doi][field] = row[field]
    return pd.DataFrame(rows.values()).sort_values(["year", "doi"], ascending=[False, True])


def citation_key(row: pd.Series) -> str:
    first = re.split(r"\s+and\s+|,", str(row.authors), maxsplit=1)[0]
    first = re.sub(r"[^A-Za-z0-9]", "", first.split()[0] if first.split() else "Reference")
    keyword = re.sub(r"[^A-Za-z0-9]", "", str(row.title).split()[0] if str(row.title).split() else "paper")
    return f"{first}{row.year}{keyword}"


def bibtex(frame: pd.DataFrame) -> str:
    entries = []
    used: dict[str, int] = {}
    for _, row in frame.iterrows():
        base = citation_key(row)
        used[base] = used.get(base, 0) + 1
        key = base if used[base] == 1 else f"{base}{used[base]}"
        fields = [
            ("author", row.authors),
            ("title", "{" + str(row.title) + "}"),
            ("journal", row.journal),
            ("year", row.year),
            ("volume", row.volume),
            ("pages", row.pages_or_article),
            ("doi", row.doi),
        ]
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields if str(value).strip())
        entries.append(f"@article{{{key},\n{body}\n}}")
    return "\n\n".join(entries) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml-root", type=Path, default=Path("data/raw/respecth_nh3_v2_3/extracted")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("literature/respecth_bibliography.csv")
    )
    parser.add_argument("--bib-output", type=Path, default=Path("literature/respecth_references.bib"))
    args = parser.parse_args()
    frame = extract(args.xml_root)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv_output, index=False, lineterminator="\n")
    args.bib_output.write_text(bibtex(frame), encoding="utf-8")
    complete = frame[["authors", "title", "journal", "year"]].ne("").all(axis=1)
    print(f"Extracted {len(frame)} unique DOI records; {int(complete.sum())} have core metadata.")
    print(args.csv_output)
    print(args.bib_output)


if __name__ == "__main__":
    main()
