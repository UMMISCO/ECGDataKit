"""Tests for the Mortara EL250 parser."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.mortara_el250 import MortaraEL250Parser


class TestMortaraEL250Parser:
    def test_parse_returns_ecg_record(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.source_format == "mortara_el250"

    def test_patient_info(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.patient.first_name == "John"
        assert record.patient.last_name == "Doe"
        assert record.patient.patient_id == "PAT001"
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.patient.birth_date == datetime(1990, 5, 15)

    def test_patient_age(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.patient.age == 33

    def test_recording_date(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.recording.date == datetime(2023, 12, 1, 12, 0, 0)

    def test_recording_sampling_rate(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.recording.acquisition.signal.sampling_rate == 500

    def test_device_model(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.recording.device.model == "EL250"

    def test_lead_count(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_annotations(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.annotations.get("num_qrs") == "75"
        assert record.annotations.get("vent_rate") == "75"

    def test_lead_units_and_is_raw(self, mortara_file: Path):
        """Fixture has UNITS_PER_MV=200 → resolution=0.005 mV/count → raw ADC."""
        record = MortaraEL250Parser().parse(mortara_file)
        for lead in record.leads:
            assert lead.resolution_unit == "mV"
            assert lead.units == ""
            assert lead.resolution == pytest.approx(0.005)
            assert lead.is_raw is True

    def test_median_beat_units_and_is_raw(self, mortara_file: Path):
        """Fixture TYPICAL_CYCLE has UNITS_PER_MV=200 → same scaling as rhythm leads."""
        record = MortaraEL250Parser().parse(mortara_file)
        for beat in record.median_beats:
            assert beat.resolution_unit == "mV"
            assert beat.units == ""
            assert beat.resolution == pytest.approx(0.005)
            assert beat.is_raw is True

    def test_representative_beats(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        rep_beats = record.raw_metadata.get("representative_beats", {})
        assert "I" in rep_beats

    def test_signal_characteristics(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        sc = record.raw_metadata.get("signal_characteristics", {})
        assert sc["device_manufacturer"] == "MORTARA"

    def test_to_dict_unified_schema(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "mortara_el250"
        assert parsed["patient"]["first_name"] == "John"

    def test_auto_detection_via_file_parser(self, mortara_file: Path):
        fp = FileParser()
        record = fp.parse(mortara_file)
        assert record.source_format == "mortara_el250"
