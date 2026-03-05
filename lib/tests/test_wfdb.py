"""Tests for the WFDB (PhysioNet) parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.wfdb import WFDBParser


class TestWFDBParser:
    def test_parse_returns_ecg_record(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.source_format == "wfdb"

    def test_patient_age(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.patient.age == 43

    def test_patient_sex(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.patient.sex == "M"

    def test_recording_sample_rate(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.recording.acquisition.signal.sample_rate == 500

    def test_recording_date(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.recording.date is not None
        assert record.recording.date.hour == 10
        assert record.recording.date.minute == 30

    def test_recording_duration(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() == 1.0

    def test_lead_count(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples_are_float(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) > 0

    def test_lead_sample_rate(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        for lead in record.leads:
            assert lead.sample_rate == 500

    def test_can_parse_requires_hea_extension(self, tmp_path: Path):
        f = tmp_path / "test.dat"
        f.write_bytes(b"\x00" * 100)
        header = f.read_bytes()[:4096]
        assert WFDBParser.can_parse(f, header) is False

    def test_can_parse_detects_hea_file(self, wfdb_file: Path):
        header = wfdb_file.read_bytes()[:4096]
        assert WFDBParser.can_parse(wfdb_file, header) is True

    def test_to_dict_unified_schema(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "wfdb"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, wfdb_file: Path):
        fp = FileParser()
        record = fp.parse(wfdb_file)
        assert record.source_format == "wfdb"
