"""Tests for the SCP-ECG parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.scp_ecg import SCPECGParser


class TestSCPECGParser:
    def test_parse_returns_ecg_record(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert record.source_format == "scp_ecg"

    def test_patient_last_name(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert "TestSCP" in record.patient.last_name

    def test_patient_id(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert "SCP001" in record.patient.patient_id

    def test_patient_sex(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1980

    def test_recording_sample_rate(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert record.recording.sample_rate == 500

    def test_lead_count(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples_are_float(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) > 0

    def test_recording_duration(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() > 0

    def test_raw_metadata_sections(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert "sections_found" in record.raw_metadata

    def test_to_dict_unified_schema(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording", "device", "filters",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "scp_ecg"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, scp_ecg_file: Path):
        fp = FileParser()
        record = fp.parse(scp_ecg_file)
        assert record.source_format == "scp_ecg"
