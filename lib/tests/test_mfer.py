"""Tests for the MFER parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.mfer import MFERParser


class TestMFERParser:
    def test_parse_returns_ecg_record(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert record.source_format == "mfer"

    def test_patient_id(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert "MFER001" in record.patient.patient_id

    def test_patient_name(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert "MferTest" in record.patient.last_name

    def test_patient_sex(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1980

    def test_recording_date(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023

    def test_recording_sample_rate(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert record.recording.sample_rate == 500

    def test_recording_duration(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() > 0

    def test_lead_count(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples_are_float(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) == 500

    def test_lead_sample_rate(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        for lead in record.leads:
            assert lead.sample_rate == 500

    def test_can_parse_rejects_non_mfer(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\xFF\xFF\xFF\xFF")
        header = f.read_bytes()[:4096]
        assert MFERParser.can_parse(f, header) is False

    def test_to_dict_unified_schema(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording", "device", "filters",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "mfer"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, mfer_file: Path):
        fp = FileParser()
        record = fp.parse(mfer_file)
        assert record.source_format == "mfer"
