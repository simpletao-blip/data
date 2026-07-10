from pathlib import Path

import pytest

from pca_ensemble.rkd import equivalent_cracking_ratio, parse_rkd_file, rkd_file_metadata


SAMPLE = Path("data/raw/respecth_nh3_v2_3/extracted/x00100000.xml")
LBV_SAMPLE = Path("data/raw/respecth_nh3_v2_3/extracted/x20100004.xml")
IDT_SAMPLE = Path("data/raw/respecth_nh3_v2_3/extracted/x10100001.xml")


def test_equivalent_cracking_ratio() -> None:
    assert equivalent_cracking_ratio({"NH3": 0.9, "H2": 0.1}) == pytest.approx(0.0689655172)
    assert equivalent_cracking_ratio({"NH3": 1.0}) == 0.0
    assert equivalent_cracking_ratio({"H2": 1.0}) == 1.0


def test_parse_jsr_sample() -> None:
    frame = parse_rkd_file(SAMPLE)
    assert len(frame) == 40
    assert set(frame.observable) == {"H2O", "NH3", "NO", "N2O"}
    assert frame.campaign_id.nunique() == 1
    assert frame.pressure_Pa.dropna().unique() == pytest.approx([101325.0])
    assert frame.equivalence_ratio.dropna().unique() == pytest.approx([0.25])
    assert frame.cracking_ratio.dropna().unique() == pytest.approx([0.0689655172])
    assert (frame.quality_status == "pending_manual_review").all()
    no = frame[frame.observable == "NO"]
    assert no.uncertainty.dropna().iloc[0] == pytest.approx(5e-6)


def test_absolute_lbv_uncertainty_is_converted_to_si() -> None:
    frame = parse_rkd_file(LBV_SAMPLE)
    first = frame.iloc[0]
    assert first.value == pytest.approx(0.962)
    assert first.unit == "m/s"
    assert first.uncertainty == pytest.approx(0.05743)
    assert first.uncertainty_type.endswith("m/s")


def test_ignition_apparatus_and_definition_are_preserved() -> None:
    frame = parse_rkd_file(IDT_SAMPLE)
    assert frame.iloc[0].apparatus == "shock tube; reflected shock"
    assert '"target": "NH3"' in frame.iloc[0].definition
    assert '"type": "relative concentration"' in frame.iloc[0].definition


def test_embedded_rkd_and_record_versions_are_distinguished() -> None:
    metadata = rkd_file_metadata(SAMPLE)
    assert metadata["rkd_format_version"] == "2.4"
    assert metadata["record_version"] == "2.0"
    assert metadata["file_doi"] == "10.24388/x00100000"
