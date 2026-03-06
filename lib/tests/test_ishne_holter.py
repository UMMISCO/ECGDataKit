"""Tests for the ISHNE Holter parser."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.ishne_holter import ISHNEHolterParser


class TestISHNEHolterParser:
    def test_parse_returns_ecg_record(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.source_format == "ishne_holter"

    def test_patient_name(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert "Test" in record.patient.first_name
        assert "Ecg" in record.patient.last_name

    def test_patient_sex(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1980

    def test_recording_sample_rate(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.recording.acquisition.signal.sample_rate == 200

    def test_recording_date(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023

    def test_lead_count(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_sample_rate(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        for lead in record.leads:
            assert lead.sample_rate == 200

    def test_lead_samples_are_float(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64

    def test_lead_units_and_is_raw(self, ishne_file: Path):
        """Fixture has ampl_res=1000 nV → resolution=1.0 uV/count → already physical."""
        record = ISHNEHolterParser().parse(ishne_file)
        for lead in record.leads:
            assert lead.resolution_unit == "uV"
            assert lead.units == "uV"
            assert lead.resolution == 1.0
            assert lead.adc_resolution == 1000.0
            assert lead.adc_resolution_unit == "nV"
            assert lead.is_raw is False

    def test_recording_duration(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() > 0

    def test_corrupted_file_too_small(self, tmp_path: Path):
        f = tmp_path / "tiny.ecg"
        f.write_bytes(b"ISHNE1.0" + b"\x00" * 10)
        with pytest.raises(Exception):
            ISHNEHolterParser().parse(f)

    def test_to_dict_unified_schema(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "ishne_holter"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, ishne_file: Path):
        fp = FileParser()
        record = fp.parse(ishne_file)
        assert record.source_format == "ishne_holter"
