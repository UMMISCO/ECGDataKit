"""Tests for the HL7 aECG exporter.

Every document is checked against the official HL7 schema. The schema is
not shipped with ECGDataKit: it is downloaded once into the user cache
(hash-checked). Without network access the schema checks are skipped.
"""

from __future__ import annotations

import dataclasses
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import ExportError
from ecgdatakit.exporter import (
    HL7aECGExporter,
    fetch_hl7_aecg_schema,
    validate_hl7_aecg,
)
from ecgdatakit.exporter import hl7_aecg as module
from ecgdatakit.models import (
    ECGRecord,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    PatientInfo,
)
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.hl7_aecg import HL7aECGParser
from ecgdatakit.parsing.parsers.xle import XLEParser
from tests.test_anonymize import ALL_FORMATS, _any_format_file

pytest.importorskip("lxml")


@pytest.fixture(scope="session")
def schema_dir() -> Path:
    try:
        return fetch_hl7_aecg_schema()
    except ExportError as e:
        pytest.skip(f"HL7 aECG schema not available: {e}")


def quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def read_back(content: bytes, tmp_path: Path) -> ECGRecord:
    path = tmp_path / "exported.xml"
    path.write_bytes(content)
    return quiet(HL7aECGParser().parse, path)


def millivolts(lead: Lead) -> np.ndarray:
    lead = lead.to_physical() if lead.is_raw else lead
    return lead.convert_units("mV").samples


def make_record(**changes) -> ECGRecord:
    """A 12-lead record in raw counts, 500 Hz, with patient, measurements,
    interpretation, filters and representative beats."""
    rng = np.random.default_rng(0)
    labels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    leads = [Lead(label=label, samples=rng.integers(-2000, 2000, 1000).astype(float),
                  sampling_rate=500, resolution=2.5, resolution_unit="uV", is_raw=True)
             for label in labels]
    beats = [Lead(label=label, samples=rng.integers(-800, 800, 600).astype(float),
                  sampling_rate=500, resolution=2.5, resolution_unit="uV", is_raw=True)
             for label in labels]
    record = ECGRecord(
        patient=PatientInfo(patient_id="SUBJ-9", first_name="Ada", last_name="Test",
                            sex="F", birth_date=datetime(1970, 5, 4), age=53,
                            clinical_history="no symptoms"),
        leads=leads,
        representative_beats=beats,
        measurements=GlobalMeasurements(heart_rate=61, rr_interval=984, pr_interval=168,
                                        qrs_duration=92, qt_interval=402, qtc_bazett=405,
                                        qtc_fridericia=404, p_axis=48, qrs_axis=12,
                                        t_axis=35),
        interpretation=Interpretation(statements=[("Sinus rhythm", ""), ("Normal ECG", "")],
                                      source="machine"),
        annotations={"representative_beat_qrs_onset_ms": "390",
                     "representative_beat_qrs_offset_ms": "482", "rhythm": "Sinus rhythm"},
    )
    record.recording.date = datetime(2024, 2, 3, 4, 5, 6)
    record.recording.duration = timedelta(seconds=2)
    record.recording.technician = "TECH-1"
    record.recording.location = "Ward B"
    record.recording.device.manufacturer = "Acme"
    record.recording.device.model = "ECG-9"
    record.recording.device.software_version = "1.2"
    record.recording.device.serial_number = "SN-77"
    record.recording.device.institution = "City Hospital"
    record.recording.acquisition.filters = FilterSettings(highpass=0.05, lowpass=150.0,
                                                          notch=50.0, notch_active=True)
    return dataclasses.replace(record, **changes)


class TestSchema:
    def test_fetch_keeps_matching_files(self, schema_dir):
        before = {p: p.stat().st_mtime_ns for p in schema_dir.rglob("*.xsd")}
        assert fetch_hl7_aecg_schema(schema_dir) == schema_dir
        assert {p: p.stat().st_mtime_ns for p in schema_dir.rglob("*.xsd")} == before

    def test_changed_schema_is_refused(self, schema_dir, tmp_path):
        copy = tmp_path / "schema"
        for name in module._SCHEMA_FILES:
            (copy / name).parent.mkdir(parents=True, exist_ok=True)
            (copy / name).write_bytes((schema_dir / name).read_bytes())
        (copy / "voc" / "voc.xsd").write_text("changed")
        with pytest.raises(FileNotFoundError, match="fetch_hl7_aecg_schema"):
            validate_hl7_aecg(b"<x/>", copy)

    def test_missing_schema_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fetch_hl7_aecg_schema"):
            validate_hl7_aecg(b"<x/>", tmp_path / "nothing")

    def test_invalid_document_is_reported(self, schema_dir):
        errors = validate_hl7_aecg(b'<AnnotatedECG xmlns="urn:hl7-org:v3"/>', schema_dir)
        assert errors and all(e.startswith("line ") for e in errors)

    def test_not_xml_is_reported(self, schema_dir):
        assert validate_hl7_aecg(b"not xml", schema_dir)[0].startswith("not well-formed")


class TestDocument:
    def test_valid_and_read_back(self, schema_dir, tmp_path):
        record = make_record()
        content = HL7aECGExporter(validate=True, schema_dir=schema_dir,
                                  trial_id="STUDY-1").to_bytes(record)
        back = read_back(content, tmp_path)
        assert [lead.label for lead in back.leads] == [lead.label for lead in record.leads]
        for a, b in zip(record.leads, back.leads):
            assert b.sampling_rate == 500
            np.testing.assert_allclose(millivolts(b), millivolts(a), atol=1e-12)
        for a, b in zip(record.representative_beats, back.representative_beats):
            np.testing.assert_allclose(millivolts(b), millivolts(a), atol=1e-12)
        p = back.patient
        assert (p.patient_id, p.first_name, p.last_name, p.sex) == ("SUBJ-9", "Ada", "Test", "F")
        assert p.birth_date == datetime(1970, 5, 4) and p.age == 53
        assert p.clinical_history == "no symptoms"
        assert back.measurements == record.measurements
        assert back.interpretation.statements == record.interpretation.statements
        assert back.recording.date == record.recording.date
        assert back.recording.duration == timedelta(seconds=2)
        assert back.recording.technician == "TECH-1"
        assert back.recording.location == "Ward B"
        device = back.recording.device
        assert (device.manufacturer, device.model, device.software_version,
                device.serial_number, device.institution) == (
            "Acme", "ECG-9", "1.2", "SN-77", "City Hospital")
        filters = back.recording.acquisition.filters
        assert (filters.highpass, filters.lowpass, filters.notch) == (0.05, 150.0, 50.0)
        assert back.annotations["representative_beat_qrs_onset_ms"] == "390"
        assert back.annotations["representative_beat_qrs_offset_ms"] == "482"
        assert back.annotations["rhythm"] == "Sinus rhythm"
        assert back.raw_metadata["trial_id"] == "STUDY-1"

    def test_ids_use_the_id_root(self, tmp_path):
        content = HL7aECGExporter(id_root="2.16.840.1.113883.3.999").to_bytes(make_record())
        assert b'<id root="2.16.840.1.113883.3.999" extension="SUBJ-9"' in content
        assert b'extension="SN-77"' in content

    def test_missing_ids_are_null_flavored(self, schema_dir):
        record = make_record(patient=PatientInfo())
        content = HL7aECGExporter(validate=True, schema_dir=schema_dir).to_bytes(record)
        assert content.count(b'<id nullFlavor="NI" />') >= 2   # subject and trial

    def test_minimal_record(self, schema_dir, tmp_path):
        lead = Lead(label="II", samples=np.arange(10, dtype=float), sampling_rate=250,
                    resolution=1.0, resolution_unit="uV", is_raw=True)
        content = HL7aECGExporter(validate=True, schema_dir=schema_dir).to_bytes(
            ECGRecord(leads=[lead]))
        assert b"TIME_RELATIVE" in content and b"TIME_ABSOLUTE" not in content
        back = read_back(content, tmp_path)
        np.testing.assert_allclose(back.leads[0].samples, np.arange(10))
        assert back.recording.date is None

    def test_time_zone_and_fraction_of_second(self, tmp_path):
        record = make_record()
        record.recording.date = datetime(2024, 1, 2, 3, 4, 5, 250000,
                                         tzinfo=timezone(timedelta(hours=1)))
        back = read_back(HL7aECGExporter().to_bytes(record), tmp_path)
        assert back.recording.date == record.recording.date

    def test_leads_with_different_rates(self, schema_dir, tmp_path):
        slow = Lead(label="V1", samples=np.arange(50, dtype=float), sampling_rate=250,
                    resolution=2.5, resolution_unit="uV", is_raw=True)
        record = make_record(representative_beats=[])
        record.leads.append(slow)
        record.leads = [record.leads[0], slow]
        content = HL7aECGExporter(validate=True, schema_dir=schema_dir).to_bytes(record)
        assert content.count(b"<sequenceSet>") == 2

    def test_control_characters_are_removed(self, schema_dir, tmp_path):
        # binary formats can store control characters in their text fields
        record = make_record(patient=PatientInfo(patient_id="A\x00B", last_name="Do\x07e",
                                                 clinical_history="x\x1by"))
        content = HL7aECGExporter(validate=True, schema_dir=schema_dir).to_bytes(record)
        back = read_back(content, tmp_path)
        assert (back.patient.patient_id, back.patient.last_name) == ("AB", "Doe")
        assert back.patient.clinical_history == "xy"

    def test_lead_without_sampling_rate(self):
        record = make_record()
        record.leads[0] = dataclasses.replace(record.leads[0], sampling_rate=0)
        with pytest.raises(ExportError, match="no sampling rate"):
            HL7aECGExporter().to_bytes(record)

    def test_export_writes_the_file(self, tmp_path):
        path = HL7aECGExporter().export(make_record(), tmp_path / "out.xml")
        assert path.read_bytes().startswith(b"<?xml")


class TestSamples:
    def test_counts_written_exactly(self):
        lead = Lead(label="I", samples=np.array([1.0, -2.0, 30000.0]), sampling_rate=500,
                    resolution=2.5, resolution_unit="uV", is_raw=True)
        digits, scale, origin, error = module._digits(lead, None)
        assert digits.tolist() == [1, -2, 30000] and scale == 2.5 and origin == 0
        assert error == 0

    def test_physical_leads_go_back_to_counts(self):
        lead = Lead(label="I", samples=np.array([3.0, 7.0]), sampling_rate=500,
                    resolution=0.9375, resolution_unit="uV", is_raw=True)
        digits, scale, _, error = module._digits(lead.to_physical().convert_units("mV"), None)
        assert digits.tolist() == [3, 7] and scale == 0.9375 and error == 0

    def test_fractions_of_a_count_written_exactly(self):
        # Wilson correction: CR - (I + II) / 3 gives thirds of a count
        lead = Lead(label="V1", samples=np.array([1 / 3, 2 / 3, 5.0]), sampling_rate=500,
                    resolution=1.0, resolution_unit="uV", is_raw=True)
        digits, scale, _, error = module._digits(lead, None)
        assert digits.tolist() == [1, 2, 15] and scale == pytest.approx(1 / 3) and error == 0

    def test_other_values_rounded_with_a_warning(self, tmp_path):
        record = make_record()
        record.representative_beats[0].samples = record.representative_beats[0].samples / 7
        with pytest.warns(UserWarning, match=r"representative beats \['I'\].*rounded to 0.01 uV"):
            content = HL7aECGExporter().to_bytes(record)
        back = read_back(content, tmp_path)
        np.testing.assert_allclose(millivolts(back.representative_beats[0]),
                                   millivolts(record.representative_beats[0]), atol=5e-6)

    def test_rounding_step_is_at_most_a_hundredth_of_a_microvolt(self):
        # A lead stored in mV with a no-op scale of 1 mV per count
        lead = Lead(label="I", samples=np.array([0.0123456, -1.5]), sampling_rate=500,
                    resolution=1.0, resolution_unit="mV", units="mV", is_raw=False)
        digits, scale, _, error = module._digits(lead, None)
        assert scale == 0.01 and digits.tolist() == [1235, -150000] and error <= 0.005

    def test_step_option(self):
        lead = Lead(label="I", samples=np.array([1.0, 2.0]), sampling_rate=500,
                    resolution=2.5, resolution_unit="uV", is_raw=True)
        digits, scale, _, _ = module._digits(lead, 0.5)
        assert digits.tolist() == [5, 10] and scale == 0.5
        with pytest.raises(ValueError):
            HL7aECGExporter(step_uv=0)

    def test_uncalibrated_lead_is_refused(self):
        record = make_record()
        record.leads[0] = dataclasses.replace(record.leads[0], resolution_unit="")
        with pytest.raises(ExportError, match="no voltage scale"):
            HL7aECGExporter().to_bytes(record)

    def test_nan_is_refused(self):
        record = make_record()
        record.leads[0].samples[3] = np.nan
        with pytest.raises(ExportError, match="NaN"):
            HL7aECGExporter().to_bytes(record)


class TestLeadCodes:
    @pytest.mark.parametrize("label, code", [
        ("I", "MDC_ECG_LEAD_I"), ("aVR", "MDC_ECG_LEAD_AVR"), ("-aVR", "MDC_ECG_LEAD_AVRneg"),
        ("V4R", "MDC_ECG_LEAD_V4R"), ("v1", "MDC_ECG_LEAD_V1"), ("CR1", None), ("CH5", None),
    ])
    def test_mdc_codes(self, label, code):
        assert module._lead_code(label) == code

    def test_unknown_lead_is_refused(self):
        record = make_record()
        record.leads[6] = dataclasses.replace(record.leads[6], label="CR1")
        with pytest.raises(ExportError, match="wilson_correction"):
            HL7aECGExporter().to_bytes(record)

    def test_unknown_lead_can_be_skipped(self, tmp_path):
        record = make_record()
        record.leads[6] = dataclasses.replace(record.leads[6], label="CR1")
        with pytest.warns(UserWarning, match="left out"):
            content = HL7aECGExporter(skip_unknown_leads=True).to_bytes(record)
        assert "CR1" not in [lead.label for lead in read_back(content, tmp_path).leads]

    def test_record_without_codable_lead(self):
        record = make_record()
        record.leads = [dataclasses.replace(record.leads[0], label="CR1")]
        with pytest.raises(ExportError, match="no lead"):
            quiet(HL7aECGExporter(skip_unknown_leads=True).to_bytes, record)


@pytest.mark.parametrize("name", ALL_FORMATS)
def test_every_format_exports_a_valid_document(name, request, tmp_path, schema_dir):
    """A record read from each supported format gives a schema-valid
    document that reads back with the same leads."""
    path = _any_format_file(name, request, tmp_path)[0]
    if name == "xle":
        record = quiet(XLEParser(wilson_correction=True, compute_missing_leads=True).parse, path)
    else:
        record = quiet(FileParser.parse, path)
    content = quiet(HL7aECGExporter(validate=True, schema_dir=schema_dir).to_bytes, record)
    back = read_back(content, tmp_path)
    assert [lead.label for lead in back.leads] == [lead.label for lead in record.leads]
    for a, b in zip(record.leads, back.leads):
        assert b.sampling_rate == a.sampling_rate
        np.testing.assert_allclose(millivolts(b), millivolts(a), atol=1e-9)
    assert len(back.representative_beats) == len(record.representative_beats)
    assert back.patient.patient_id == record.patient.patient_id


def test_edan_holter_has_no_voltage_scale(edan_arc_dir):
    """EDAN ARC Holter files do not state a voltage scale: they cannot be
    written as HL7 aECG, which stores voltages."""
    record = quiet(FileParser.parse, edan_arc_dir)
    with pytest.raises(ExportError, match="no voltage scale"):
        HL7aECGExporter().to_bytes(record)


def test_xle_default_chest_leads_need_wilson(request, tmp_path):
    path = _any_format_file("xle", request, tmp_path)[0]
    record = quiet(XLEParser().parse, path)
    with pytest.raises(ExportError, match="wilson_correction=True"):
        HL7aECGExporter().to_bytes(record)
