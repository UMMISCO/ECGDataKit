"""Tests for the HL7 aECG parser."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.hl7_aecg import HL7aECGParser


class TestHL7aECGParser:
    def test_parse_returns_ecg_record(self, hl7_aecg_file: Path):
        parser = HL7aECGParser()
        record = parser.parse(hl7_aecg_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        assert record.source_format == "hl7_aecg"

    def test_patient_sex(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        assert record.patient.birth_date == datetime(1980, 1, 1)

    def test_recording_dates(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        assert record.recording.date == datetime(2023, 6, 15, 10, 30, 0)
        assert record.recording.end_date == datetime(2023, 6, 15, 10, 30, 10)

    def test_recording_duration(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() == 10.0

    def test_lead_count(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        lead_i = next(l for l in record.leads if l.label == "I")
        np.testing.assert_array_equal(lead_i.samples, [100, 200, 300, 400, 500])

    def test_lead_units_and_is_raw(self, hl7_aecg_file: Path):
        """Fixture has no scale/origin → defaults 1.0/0.0 → already physical."""
        record = HL7aECGParser().parse(hl7_aecg_file)
        for lead in record.leads:
            assert lead.resolution == 1.0
            assert lead.offset == 0.0
            assert lead.is_raw is False

    def test_to_dict_unified_schema(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "hl7_aecg"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, hl7_aecg_file: Path):
        fp = FileParser()
        record = fp.parse(hl7_aecg_file)
        assert record.source_format == "hl7_aecg"
