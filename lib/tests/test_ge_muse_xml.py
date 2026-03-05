"""Tests for the GE MUSE XML parser."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.ge_muse_xml import GEMuseXMLParser


class TestGEMuseXMLParser:
    def test_parse_returns_ecg_record(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.source_format == "ge_muse_xml"

    def test_patient_id(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.patient.patient_id == "MUSE001"

    def test_patient_name(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.patient.first_name == "Jane"
        assert record.patient.last_name == "Smith"

    def test_patient_sex(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.patient.sex == "F"

    def test_patient_birth_date(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1985

    def test_patient_age(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.patient.age == 38

    def test_recording_date(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023
        assert record.recording.date.month == 12

    def test_device_model(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.recording.device.model == "MAC5500"

    def test_recording_sample_rate(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.recording.acquisition.signal.sample_rate == 500

    def test_lead_count(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples_are_float(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) > 0

    def test_annotations(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert "diagnosis" in record.annotations
        assert "Normal sinus rhythm" in record.annotations["diagnosis"]
        assert "ventricularrate" in record.annotations

    def test_does_not_match_sierra(self, tmp_path: Path):
        f = tmp_path / "sierra.xml"
        f.write_text('<?xml version="1.0"?><restingecgdata></restingecgdata>')
        header = f.read_bytes()[:4096]
        assert GEMuseXMLParser.can_parse(f, header) is False

    def test_to_dict_unified_schema(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "ge_muse_xml"
        assert len(parsed["leads"]) == 2

    def test_measurements_populated(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.measurements.heart_rate == 72
        assert record.measurements.qrs_duration == 88

    def test_interpretation_source_machine(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.interpretation.source == "machine"
        assert "Normal sinus rhythm" in record.interpretation.statements

    def test_auto_detection_via_file_parser(self, ge_muse_xml_file: Path):
        fp = FileParser()
        record = fp.parse(ge_muse_xml_file)
        assert record.source_format == "ge_muse_xml"
