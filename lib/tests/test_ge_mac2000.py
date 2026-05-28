"""Tests for the GE MAC 2000 parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.ge_mac2000 import GEMAC2000Parser


class TestGEMAC2000Parser:
    def test_parse_returns_ecg_record(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert record.source_format == "ge_mac2000"

    def test_patient_id(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert record.patient.patient_id == "MAC001"

    def test_patient_name(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert record.patient.first_name == "Bob"
        assert record.patient.last_name == "Builder"

    def test_patient_sex(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1978

    def test_patient_age(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert record.patient.age == 45

    def test_recording_date(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023

    def test_recording_sampling_rate(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert record.recording.acquisition.signal.sampling_rate == 500

    def test_lead_count(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples_are_float(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) > 0

    def test_lead_units_and_is_raw(self, ge_mac2000_file: Path):
        """Fixture has no LeadAmplitudeUnitsPerBit → scale=1.0 → already physical."""
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        for lead in record.leads:
            assert lead.resolution == 1.0
            assert lead.is_raw is False

    def test_annotations(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert "ventricularrate" in record.annotations
        assert record.annotations["ventricularrate"] == "68"

    def test_does_not_match_muse(self, tmp_path: Path):
        f = tmp_path / "muse.xml"
        f.write_text('<?xml version="1.0"?><RestingECG><MuseInfo/></RestingECG>')
        header = f.read_bytes()[:4096]
        assert GEMAC2000Parser.can_parse(f, header) is False

    def test_to_dict_unified_schema(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "ge_mac2000"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, ge_mac2000_file: Path):
        fp = FileParser()
        record = fp.parse(ge_mac2000_file)
        assert record.source_format == "ge_mac2000"
