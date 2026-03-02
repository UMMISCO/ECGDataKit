"""Tests for the EDF/EDF+ parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.edf import EDFParser


class TestEDFParser:
    def test_parse_returns_ecg_record(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert isinstance(record, ECGRecord)

    def test_source_format_edfplus(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.source_format == "edf_plus"

    def test_patient_id(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.patient.patient_id == "P001"

    def test_patient_sex(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1980

    def test_recording_date(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023
        assert record.recording.date.month == 12

    def test_recording_sample_rate(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.recording.sample_rate == 500

    def test_recording_duration(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() > 0

    def test_lead_count(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        labels = [l.label for l in record.leads]
        assert "EDF ECG I" in labels
        assert "EDF ECG II" in labels

    def test_lead_samples_are_float(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) > 0

    def test_lead_sample_rate(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        for lead in record.leads:
            assert lead.sample_rate == 500

    def test_corrupted_file_too_small(self, tmp_path: Path):
        f = tmp_path / "tiny.edf"
        f.write_bytes(b"0       " + b"\x00" * 10)
        with pytest.raises(Exception):
            EDFParser().parse(f)

    def test_to_dict_unified_schema(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording", "device", "filters",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] in ("edf", "edf_plus")
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, edf_file: Path):
        fp = FileParser()
        record = fp.parse(edf_file)
        assert record.source_format in ("edf", "edf_plus")
