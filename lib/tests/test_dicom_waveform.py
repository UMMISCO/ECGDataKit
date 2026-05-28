"""Tests for the DICOM Waveform parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.dicom_waveform import DICOMWaveformParser


class TestDICOMWaveformParser:
    def test_parse_returns_ecg_record(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.source_format == "dicom_waveform"

    def test_patient_id(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.patient.patient_id == "DCM001"

    def test_patient_name(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.patient.last_name == "Doe"
        assert record.patient.first_name == "John"

    def test_patient_sex(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1990

    def test_recording_date(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023

    def test_recording_sampling_rate(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.recording.acquisition.signal.sampling_rate == 500

    def test_lead_count(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples_are_float(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) == 100

    def test_lead_units_and_is_raw(self, dicom_file: Path):
        """Fixture has sensitivity=1.0, baseline=0 → already physical."""
        record = DICOMWaveformParser().parse(dicom_file)
        for lead in record.leads:
            assert lead.resolution == 1.0
            assert lead.is_raw is False

    def test_recording_duration(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() > 0

    def test_can_parse_detects_dicm_magic(self, dicom_file: Path):
        header = dicom_file.read_bytes()[:4096]
        assert DICOMWaveformParser.can_parse(dicom_file, header) is True

    def test_can_parse_rejects_non_dicom(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00" * 200)
        header = f.read_bytes()[:4096]
        assert DICOMWaveformParser.can_parse(f, header) is False

    def test_to_dict_unified_schema(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "dicom_waveform"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, dicom_file: Path):
        fp = FileParser()
        record = fp.parse(dicom_file)
        assert record.source_format == "dicom_waveform"
