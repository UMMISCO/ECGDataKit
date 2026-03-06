"""Tests for the Mindray BeneHeart R12 parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.beneheart_r12 import BeneHeartR12Parser


class TestBeneHeartR12Parser:
    def test_parse_returns_ecg_record(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert record.source_format == "beneheart_r12"

    def test_patient_id(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert record.patient.patient_id == "BH001"

    def test_patient_name(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert record.patient.first_name == "Alice"
        assert record.patient.last_name == "Wonder"

    def test_patient_sex(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert record.patient.sex == "F"

    def test_patient_birth_date(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1992

    def test_patient_age(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert record.patient.age == 31

    def test_recording_date(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023

    def test_recording_sample_rate(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert record.recording.acquisition.signal.sample_rate == 500

    def test_recording_device(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert "BeneHeart" in record.recording.device.model

    def test_lead_count(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples_are_float(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) > 0

    def test_lead_units_and_is_raw(self, beneheart_r12_file: Path):
        """Fixture has resolution=1.0 (1 uV/LSB) → already physical in uV."""
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        for lead in record.leads:
            assert lead.resolution_unit == "uV"
            assert lead.units == "uV"
            assert lead.resolution == 1.0
            assert lead.is_raw is False

    def test_to_dict_unified_schema(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "beneheart_r12"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, beneheart_r12_file: Path):
        fp = FileParser()
        record = fp.parse(beneheart_r12_file)
        assert record.source_format == "beneheart_r12"
