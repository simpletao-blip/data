"""Build the numbered manuscript reference section from locally verified metadata."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
XML_METADATA = ROOT / "literature" / "respecth_bibliography.csv"
OUTPUT = ROOT / "manuscript" / "references_draft.md"
REGISTRY_OUTPUT = ROOT / "literature" / "manuscript_reference_registry.csv"


SEQUENCE = [
    "10.1016/j.combustflame.2024.113560",
    "10.1016/j.fuel.2021.120979",
    "10.1016/j.combustflame.2021.111472",
    "10.1016/j.combustflame.2025.114049",
    "10.1016/j.cej.2024.157283",
    "10.1016/j.combustflame.2019.05.003",
    "10.1016/j.combustflame.2020.08.004",
    "10.1016/j.combustflame.2021.111753",
    "10.1016/j.fuel.2015.06.070",
    "10.1016/j.fuel.2019.116653",
    "10.1016/j.fuel.2020.118425",
    "10.1016/j.ijhydene.2009.11.071",
    "10.1016/j.ijhydene.2010.07.104",
    "10.1016/j.ijhydene.2015.04.024",
    "10.1016/j.ijhydene.2021.07.063",
    "10.1016/j.ijhydene.2021.09.188",
    "10.1016/j.jhazmat.2018.09.073",
    "10.1016/j.proci.2020.06.197",
    "10.1299/transjsme.14-00423",
    "10.1016/S0082-0784(81)80091-4",
    "10.1016/j.combustflame.2021.111653",
    "10.1016/j.fuel.2019.116768",
    "10.1016/j.fuel.2020.118054",
    "10.1016/j.fuel.2021.122202",
    "10.1039/C9RE00429G",
    "10.1080/00102202.2019.1678380",
    "10.1021/acs.energyfuels.8b01056",
    "10.1016/j.cej.2023.144577",
    "10.1016/j.ijhydene.2017.12.066",
    "10.1016/j.combustflame.2023.113239",
    "10.1016/j.fuel.2023.127676",
    "10.1016/j.jaecs.2021.100043",
    "10.1039/D5CP04149J",
    "10.5281/zenodo.19725531",
    "10.1016/j.combustflame.2019.08.033",
]


MANUAL = {
    "10.1016/j.combustflame.2024.113560": dict(
        authors="S. Girhe, A. Snackers, T.P.S. Lehmann, R.T. Langer, F. Loffredo, R. Glaznev, J. Beeckmann, H. Pitsch",
        title="Ammonia and ammonia/hydrogen combustion: Comprehensive quantitative assessment of kinetic models and examination of critical parameters",
        journal="Combustion and Flame", year="2024", volume="267", pages_or_article="113560", provenance="RWTH institutional record and publisher DOI",
    ),
    "10.1016/j.combustflame.2025.114049": dict(
        authors="H.M. Colmán, M.E. Mueller", title="Rush-to-equilibrium concept for minimizing reactive nitrogen emissions in ammonia combustion",
        journal="Combustion and Flame", year="2025", volume="275", pages_or_article="114049", provenance="Princeton institutional record and publisher DOI",
    ),
    "10.1016/j.cej.2024.157283": dict(
        authors="N. Wang, T. Li, X. Guo, Z. Wu, S. Huang, X. Zhou, S. Li, R. Chen", title="Laminar burning characteristics of ammonia and hydrogen blends at elevated initial pressures up to 2.5 MPa",
        journal="Chemical Engineering Journal", year="2024", volume="500", pages_or_article="157283", provenance="archived publisher PDF and publisher DOI",
    ),
    "10.1021/acs.energyfuels.8b01056": dict(
        authors="K.P. Shrestha, L. Seidel, T. Zeuch, F. Mauss", title="Detailed kinetic mechanism for the oxidation of ammonia including the formation and reduction of nitrogen oxides",
        journal="Energy & Fuels", year="2018", volume="32", pages_or_article="10202–10217", provenance="ACS DOI metadata",
    ),
    "10.1016/j.cej.2023.144577": dict(
        authors="A. Stagni et al.", title="Low- and intermediate-temperature ammonia/hydrogen oxidation in a flow reactor: Experiments and a wide-range kinetic modeling",
        journal="Chemical Engineering Journal", year="2023", volume="471", pages_or_article="144577", provenance="publisher DOI and locally archived mechanism source",
    ),
    "10.1016/j.ijhydene.2017.12.066": dict(
        authors="J. Otomo, M. Koshi, T. Mitsumori, H. Iwasaki, K. Yamada", title="Chemical kinetic modeling of ammonia oxidation with improved reaction mechanism for ammonia/air and ammonia/hydrogen/air combustion",
        journal="International Journal of Hydrogen Energy", year="2018", volume="43", pages_or_article="3004–3014", provenance="publisher DOI metadata",
    ),
    "10.1016/j.combustflame.2023.113239": dict(
        authors="Y. Zhu, H.J. Curran, S. Girhe, Y. Murakami, H. Pitsch, K. Senecal, L. Yang, C.-W. Zhou", title="The combustion chemistry of ammonia and ammonia/hydrogen mixtures: A comprehensive chemical kinetic modeling study",
        journal="Combustion and Flame", year="2024", volume="260", pages_or_article="113239", provenance="publisher DOI and institutional record",
    ),
    "10.1016/j.fuel.2023.127676": dict(
        authors="X. Zhang, K.K. Yalamanchi, S.M. Sarathy", title="Combustion chemistry of ammonia/C1 fuels: A comprehensive kinetic modeling study",
        journal="Fuel", year="2023", volume="341", pages_or_article="127676", provenance="publisher DOI and archived mechanism header",
    ),
    "10.1016/j.jaecs.2021.100043": dict(
        authors="S. Dong et al.", title="A new detailed kinetic model for surrogate fuels: C3MechV3.3",
        journal="Applications in Energy and Combustion Science", year="2022", volume="9", pages_or_article="100043", provenance="institutional record and publisher DOI",
    ),
    "10.1039/D5CP04149J": dict(
        authors="Y.-C. Kao, A.C. Doner, T.T. Pekkanen, C. Cao, S. Shin, A. Grinberg Dana, Y.-P. Li, W.H. Green", title="Detailed kinetic model for combustion of NH3/H2 blends",
        journal="Physical Chemistry Chemical Physics", year="2026", volume="28", pages_or_article="6411–6424", provenance="RSC article and supplementary mechanism",
    ),
    "10.5281/zenodo.19725531": dict(
        authors="U.P. Padhi, A.A. Konnov", title="A detailed chemical kinetic model of NH3 combustion, version v1",
        journal="Zenodo", year="2026", volume="", pages_or_article="", provenance="Zenodo model record",
    ),
}


def normalize_doi(value: str) -> str:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I).lower()


def format_reference(number: int, row: dict[str, str]) -> str:
    journal = row["journal"]
    volume = f" {row['volume']}" if row.get("volume") else ""
    pages = f", {row['pages_or_article']}" if row.get("pages_or_article") else ""
    return (
        f"{number}. {row['authors']}, {row['title']}, *{journal}*{volume} "
        f"({row['year']}){pages}. https://doi.org/{row['doi']}"
    )


def main() -> None:
    extracted = pd.read_csv(XML_METADATA, dtype=str).fillna("")
    records = {
        normalize_doi(str(row["doi"])): {key: str(value) for key, value in row.items()}
        for _, row in extracted.iterrows()
    }
    for doi, values in MANUAL.items():
        records[normalize_doi(doi)] = {**values, "doi": doi}
    ordered = []
    lines = ["# References", ""]
    for number, doi in enumerate(SEQUENCE, start=1):
        key = normalize_doi(doi)
        if key not in records:
            raise SystemExit(f"Missing verified metadata for {doi}")
        row = {k: str(v) for k, v in records[key].items()}
        row["doi"] = doi
        row["reference_number"] = number
        row.setdefault("provenance", "ReSpecTh v2.3 XML bibliography metadata")
        ordered.append(row)
        lines.append(format_reference(number, row))
        lines.append("")
    OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    pd.DataFrame(ordered).to_csv(REGISTRY_OUTPUT, index=False, lineterminator="\n")
    print(f"Wrote {len(ordered)} references to {OUTPUT}")
    print(REGISTRY_OUTPUT)


if __name__ == "__main__":
    main()
