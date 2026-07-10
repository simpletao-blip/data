"""Parse ReSpecTh RKD XML files into an auditable experimental long table.

The parser preserves reported values and adds SI-normalized fields.  Records are
left as ``pending_manual_review`` because apparatus-specific definitions and
experimental independence must be checked against the source publication before
they are admitted to model validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


OUTPUT_COLUMNS = [
    "dataset_id", "campaign_id", "doi", "laboratory", "apparatus",
    "observable", "value", "unit", "uncertainty", "uncertainty_type",
    "temperature_K", "pressure_Pa", "equivalence_ratio", "cracking_ratio",
    "fuel_composition", "oxidizer_composition", "definition",
    "source_location", "digitized", "quality_status", "exclusion_reason",
    "source_file", "file_doi", "experiment_type", "residence_time_s",
    "time_s", "reported_value", "reported_unit", "curator",
    "initial_composition",
]


def _text(node: ET.Element | None, path: str, default: str = "") -> str:
    if node is None:
        return default
    value = node.findtext(path)
    return value.strip() if value and value.strip() else default


def _number(value: str | None) -> float | None:
    try:
        result = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def _convert(value: float | None, unit: str, quantity: str) -> tuple[float | None, str]:
    if value is None:
        return None, unit
    key = unit.strip().lower()
    if quantity == "pressure":
        factors = {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "bar": 1e5,
                   "atm": 101325.0, "torr": 101325.0 / 760.0}
        return value * factors[key], "Pa" if key in factors else unit
    if quantity in {"time", "ignition delay", "residence time"}:
        factors = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6}
        return value * factors[key], "s" if key in factors else unit
    if quantity == "laminar burning velocity":
        factors = {"m/s": 1.0, "cm/s": 1e-2, "mm/s": 1e-3}
        return value * factors[key], "m/s" if key in factors else unit
    if quantity == "composition":
        factors = {"mole fraction": 1.0, "unitless": 1.0,
                   "percent": 1e-2, "%": 1e-2, "ppm": 1e-6}
        return value * factors[key], "mole fraction" if key in factors else unit
    if quantity == "temperature" and key == "k":
        return value, "K"
    return value, unit


def _species(prop: ET.Element) -> str:
    link = prop.find("speciesLink")
    return link.get("preferredKey", "") if link is not None else ""


def _property_value(prop: ET.Element) -> float | None:
    return _number(prop.findtext("value"))


def _composition(prop: ET.Element) -> dict[str, float]:
    result: dict[str, float] = {}
    for component in prop.findall("component"):
        link = component.find("speciesLink")
        amount = component.find("amount")
        if link is None or amount is None:
            continue
        species = link.get("preferredKey", "")
        value, _ = _convert(_number(amount.text), amount.get("units", ""), "composition")
        if species and value is not None:
            result[species] = value
    return result


def equivalent_cracking_ratio(composition: dict[str, float]) -> float | None:
    """Return alpha implied by H2/NH3 in ideal NH3 -> 1.5 H2 + 0.5 N2.

    Bath-gas N2 is intentionally ignored.  This is an equivalent fuel-side
    descriptor, not proof that the experimental mixture was produced by cracking.
    """
    nh3 = composition.get("NH3", 0.0)
    h2 = composition.get("H2", 0.0)
    if nh3 <= 0.0 and h2 <= 0.0:
        return None
    if nh3 <= 0.0:
        return 1.0
    ratio = h2 / nh3
    return ratio / (1.5 + ratio)


def _campaign_id(doi: str, fallback: str) -> str:
    key = (doi.strip().lower() or fallback).encode("utf-8")
    return "campaign_" + hashlib.sha1(key).hexdigest()[:12]


def _common_scalar(properties: list[ET.Element], name: str) -> tuple[float | None, str]:
    for prop in properties:
        if prop.get("name", "").lower() == name:
            return _property_value(prop), prop.get("units", "")
    return None, ""


def _uncertainty(
    observable: ET.Element,
    group_properties: list[ET.Element],
    common_properties: list[ET.Element],
) -> tuple[float | None, str]:
    target_name = observable.get("name", "")
    target_species = _species(observable)
    candidates: list[tuple[int, ET.Element]] = []
    for scope_priority, properties in ((0, group_properties), (1, common_properties)):
        for prop in properties:
            pname = prop.get("name", "").lower()
            if pname not in {"evaluated standard deviation", "uncertainty"}:
                continue
            reference = prop.get("reference", "")
            species = _species(prop)
            if reference and reference != target_name:
                continue
            if species and target_species and species != target_species:
                continue
            # Evaluated standard deviations are preferred to reported bounds.
            rank = scope_priority + (0 if pname == "evaluated standard deviation" else 10)
            candidates.append((rank, prop))
    if not candidates:
        return None, "not reported"
    prop = min(candidates, key=lambda item: item[0])[1]
    value = _property_value(prop)
    if value is None and prop.get("id"):
        return None, "pointwise"
    kind = prop.get("kind", "")
    unit = prop.get("units", "")
    name = prop.get("name", "")
    uncertainty_type = ":".join(part for part in (name, kind, unit) if part)
    return value, uncertainty_type or "reported"


def _point_uncertainty(
    observable: ET.Element,
    point: ET.Element,
    group_properties: list[ET.Element],
    common_properties: list[ET.Element],
) -> tuple[float | None, str]:
    target_name = observable.get("name", "")
    target_species = _species(observable)
    candidates: list[tuple[int, ET.Element, bool]] = []
    for scope, properties in ((0, group_properties), (1, common_properties)):
        for prop in properties:
            pname = prop.get("name", "").lower()
            if pname not in {"evaluated standard deviation", "uncertainty"}:
                continue
            if prop.get("reference", "") not in {"", target_name}:
                continue
            species = _species(prop)
            if species and target_species and species != target_species:
                continue
            rank = scope + (0 if pname == "evaluated standard deviation" else 10)
            candidates.append((rank, prop, scope == 0))
    for _, prop, is_group in sorted(candidates, key=lambda item: item[0]):
        value = (_number(point.findtext(prop.get("id", "")))
                 if is_group else _property_value(prop))
        if value is None:
            continue
        pname = prop.get("name", "")
        kind = prop.get("kind", "")
        unit = prop.get("units", "")
        if kind.lower() != "relative" and unit.lower() not in {"", "unitless"}:
            value, unit = _convert(value, unit, target_name)
        utype = ":".join(part for part in (pname, kind, unit) if part)
        return value, utype or "reported"
    value, utype = _uncertainty(observable, group_properties, common_properties)
    if value is not None:
        # _uncertainty has already selected the best common candidate; recover
        # its unit through the type suffix and normalize absolute SI quantities.
        unit = utype.rsplit(":", 1)[-1] if ":" in utype else ""
        if "relative" not in utype.lower() and unit.lower() not in {"", "unitless"}:
            value, canonical_unit = _convert(value, unit, target_name)
            if canonical_unit != unit:
                utype = utype[: -len(unit)] + canonical_unit
    return value, utype


def parse_rkd_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    root = ET.parse(path).getroot()
    experiment_type = _text(root, "experimentType")
    doi = _text(root, "bibliographyLink/referenceDOI")
    file_doi = _text(root, "fileDOI")
    campaign_id = _campaign_id(doi, file_doi or path.stem)
    apparatus_parts = [_text(root, "apparatus/kind"), _text(root, "apparatus/mode")]
    apparatus = "; ".join(part for part in apparatus_parts if part)
    curator = _text(root, "fileAuthor")
    ignition_node = root.find("ignitionType")
    ignition_type = (json.dumps(dict(sorted(ignition_node.attrib.items())), sort_keys=True)
                     if ignition_node is not None else "")
    bibliography = root.find("bibliographyLink")
    source_parts = []
    for tag in ("location", "table", "figure"):
        value = _text(bibliography, tag)
        if value:
            source_parts.append(f"{tag}: {value}")
    source_location = "; ".join(source_parts)

    common = list(root.findall("commonProperties/property"))
    initial: dict[str, float] = {}
    for prop in common:
        if prop.get("name", "").lower() == "initial composition":
            initial.update(_composition(prop))
    common_t, common_t_unit = _common_scalar(common, "temperature")
    common_p, common_p_unit = _common_scalar(common, "pressure")
    common_phi, _ = _common_scalar(common, "equivalence ratio")
    common_res, common_res_unit = _common_scalar(common, "residence time")
    common_t, _ = _convert(common_t, common_t_unit, "temperature")
    common_p, _ = _convert(common_p, common_p_unit, "pressure")
    common_res, _ = _convert(common_res, common_res_unit, "residence time")

    rows: list[dict[str, object]] = []
    point_serial = 0
    for group in root.findall("dataGroup"):
        properties = list(group.findall("property"))
        by_id = {prop.get("id", ""): prop for prop in properties}
        output_names = {"laminar burning velocity"}
        if experiment_type == "ignition delay measurement":
            output_names.add("ignition delay")
        if "concentration" in experiment_type or "reactor" in experiment_type:
            output_names.add("composition")
        observables = [p for p in properties if p.get("name", "").lower() in output_names]

        for point in group.findall("dataPoint"):
            point_serial += 1
            point_values = {child.tag: _number(child.text) for child in point}
            point_comp = dict(initial)
            if experiment_type == "laminar burning velocity measurement":
                for pid, prop in by_id.items():
                    if prop.get("name", "").lower() == "composition":
                        value, _ = _convert(point_values.get(pid), prop.get("units", ""), "composition")
                        if _species(prop) and value is not None:
                            point_comp[_species(prop)] = value

            scalars: dict[str, float | None] = {
                "temperature": common_t, "pressure": common_p,
                "equivalence ratio": common_phi, "residence time": common_res,
                "time": None,
            }
            for pid, prop in by_id.items():
                name = prop.get("name", "").lower()
                if name not in scalars:
                    continue
                value, _ = _convert(point_values.get(pid), prop.get("units", ""), name)
                scalars[name] = value

            for observable in observables:
                pid = observable.get("id", "")
                reported = point_values.get(pid)
                if reported is None:
                    continue
                name = observable.get("name", "").lower()
                value, unit = _convert(reported, observable.get("units", ""), name)
                species = _species(observable)
                observable_name = species if name == "composition" and species else name
                uncertainty, uncertainty_type = _point_uncertainty(
                    observable, point, properties, common
                )
                definition = ignition_type if name == "ignition delay" else observable.get("label", "")
                row_id = f"{path.stem}:g{group.get('id', '')}:p{point_serial}:{observable_name}"
                fuel = {key: value for key, value in point_comp.items() if key in {"NH3", "H2"}}
                oxidizer = {key: value for key, value in point_comp.items() if key not in {"NH3", "H2"}}
                rows.append({
                    "dataset_id": row_id,
                    "campaign_id": campaign_id,
                    "doi": doi,
                    "laboratory": "",
                    "apparatus": apparatus,
                    "observable": observable_name,
                    "value": value,
                    "unit": unit,
                    "uncertainty": uncertainty,
                    "uncertainty_type": uncertainty_type,
                    "temperature_K": scalars["temperature"],
                    "pressure_Pa": scalars["pressure"],
                    "equivalence_ratio": scalars["equivalence ratio"],
                    "cracking_ratio": equivalent_cracking_ratio(point_comp),
                    "fuel_composition": json.dumps(fuel, sort_keys=True),
                    "oxidizer_composition": json.dumps(oxidizer, sort_keys=True),
                    "definition": definition,
                    "source_location": source_location,
                    "digitized": False,
                    "quality_status": "pending_manual_review",
                    "exclusion_reason": "",
                    "source_file": path.name,
                    "file_doi": file_doi,
                    "experiment_type": experiment_type,
                    "residence_time_s": scalars["residence time"],
                    "time_s": scalars["time"],
                    "reported_value": reported,
                    "reported_unit": observable.get("units", ""),
                    "curator": curator,
                    "initial_composition": json.dumps(point_comp, sort_keys=True),
                })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def rkd_file_metadata(path: str | Path) -> dict[str, str]:
    """Return version and identifier metadata embedded in one RKD XML file."""
    path = Path(path)
    root = ET.parse(path).getroot()
    return {
        "file_doi": _text(root, "fileDOI"),
        "rkd_format_version": ".".join(
            part for part in (_text(root, "ReSpecThVersion/major"),
                              _text(root, "ReSpecThVersion/minor")) if part
        ),
        "record_version": ".".join(
            part for part in (_text(root, "fileVersion/major"),
                              _text(root, "fileVersion/minor")) if part
        ),
        "first_publication_date": _text(root, "firstPublicationDate"),
        "last_modification_date": _text(root, "lastModificationDate"),
    }


def parse_rkd_directory(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(Path(path).glob("*.xml"))
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for file in files:
        try:
            frame = parse_rkd_file(file)
            metadata = rkd_file_metadata(file)
            frames.append(frame)
            audit_rows.append({
                "source_file": file.name,
                "status": "parsed",
                "rows": len(frame),
                "error": "",
                **metadata,
            })
        except Exception as exc:  # retain a complete failure ledger
            audit_rows.append({"source_file": file.name, "status": "failed", "rows": 0,
                               "error": f"{type(exc).__name__}: {exc}",
                               "file_doi": "", "rkd_format_version": "",
                               "record_version": "", "first_publication_date": "",
                               "last_modification_date": ""})
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    audit = pd.DataFrame(audit_rows)
    return combined, audit
