"""Tests for ECGDataKit data models and JSON serialisation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from ecgdatakit.models import (
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
)


class TestPatientInfo:
    def test_defaults(self):
        p = PatientInfo()
        assert p.patient_id == ""
        assert p.sex == ""
        assert p.birth_date is None
        assert p.age is None

    def test_to_dict(self):
        p = PatientInfo(
            patient_id="P001",
            first_name="John",
            last_name="Doe",
            birth_date=datetime(1990, 5, 15),
            sex="M",
            age=33,
            weight=80.0,
            height=175.0,
        )
        d = p.to_dict()
        assert d["patient_id"] == "P001"
        assert d["birth_date"] == "1990-05-15T00:00:00"
        assert d["sex"] == "M"
        assert d["age"] == 33

    def test_to_dict_null_birth_date(self):
        d = PatientInfo().to_dict()
        assert d["birth_date"] is None


class TestDeviceInfo:
    def test_defaults(self):
        d = DeviceInfo()
        assert d.manufacturer == ""
        assert d.model == ""
        assert d.serial_number == ""
        assert d.software_version == ""

    def test_to_dict(self):
        d = DeviceInfo(
            manufacturer="Philips",
            model="TC70",
            serial_number="SN123",
            software_version="3.1",
            institution="Hospital",
            department="Cardiology",
            acquisition_type="STD-12",
        ).to_dict()
        assert d["manufacturer"] == "Philips"
        assert d["model"] == "TC70"
        assert d["acquisition_type"] == "STD-12"

    def test_dict_keys_stable(self):
        d = DeviceInfo().to_dict()
        expected = {
            "manufacturer", "model", "serial_number", "software_version",
            "institution", "department", "acquisition_type",
        }
        assert set(d.keys()) == expected


class TestFilterSettings:
    def test_defaults(self):
        f = FilterSettings()
        assert f.highpass is None
        assert f.lowpass is None
        assert f.notch is None

    def test_to_dict(self):
        d = FilterSettings(
            highpass=0.05, lowpass=150.0, notch=50.0,
            notch_active=True, artifact_filter=False,
        ).to_dict()
        assert d["highpass"] == 0.05
        assert d["lowpass"] == 150.0
        assert d["notch_active"] is True

    def test_dict_keys_stable(self):
        d = FilterSettings().to_dict()
        expected = {"highpass", "lowpass", "notch", "notch_active", "artifact_filter"}
        assert set(d.keys()) == expected


class TestInterpretation:
    def test_defaults(self):
        i = Interpretation()
        assert i.statements == []
        assert i.severity == ""
        assert i.source == ""

    def test_to_dict(self):
        d = Interpretation(
            statements=["Normal sinus rhythm"],
            severity="NORMAL",
            source="machine",
            interpreter="Dr. Smith",
            interpretation_date=datetime(2023, 6, 15, 10, 30),
        ).to_dict()
        assert d["statements"] == ["Normal sinus rhythm"]
        assert d["severity"] == "NORMAL"
        assert d["interpretation_date"] == "2023-06-15T10:30:00"

    def test_dict_keys_stable(self):
        d = Interpretation().to_dict()
        expected = {"statements", "severity", "source", "interpreter", "interpretation_date"}
        assert set(d.keys()) == expected


class TestGlobalMeasurements:
    def test_defaults(self):
        m = GlobalMeasurements()
        assert m.heart_rate is None
        assert m.pr_interval is None

    def test_to_dict(self):
        d = GlobalMeasurements(
            heart_rate=72, pr_interval=160, qrs_duration=88,
            qt_interval=400, qtc_bazett=420, p_axis=45,
            qrs_axis=60, t_axis=30,
        ).to_dict()
        assert d["heart_rate"] == 72
        assert d["qrs_duration"] == 88

    def test_dict_keys_stable(self):
        d = GlobalMeasurements().to_dict()
        expected = {
            "heart_rate", "rr_interval", "pr_interval", "qrs_duration",
            "qt_interval", "qtc_bazett", "qtc_fridericia",
            "p_axis", "qrs_axis", "t_axis", "qrs_count",
        }
        assert set(d.keys()) == expected


class TestRecordingInfo:
    def test_defaults(self):
        r = RecordingInfo()
        assert r.sample_rate == 0
        assert r.duration is None

    def test_to_dict(self):
        r = RecordingInfo(
            date=datetime(2023, 6, 15, 10, 30),
            end_date=datetime(2023, 6, 15, 10, 30, 10),
            duration=timedelta(seconds=10),
            sample_rate=500,
        )
        d = r.to_dict()
        assert d["date"] == "2023-06-15T10:30:00"
        assert d["duration_seconds"] == 10.0
        assert d["sample_rate"] == 500

    def test_to_dict_null_dates(self):
        d = RecordingInfo().to_dict()
        assert d["date"] is None
        assert d["duration_seconds"] is None


class TestLead:
    def test_to_dict_with_samples(self):
        samples = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        lead = Lead(label="I", samples=samples, sample_rate=500)
        d = lead.to_dict()
        assert d["label"] == "I"
        assert d["sample_count"] == 3
        assert d["samples"] == [1.0, 2.0, 3.0]

    def test_to_dict_without_samples(self):
        samples = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        lead = Lead(label="V1", samples=samples, sample_rate=500)
        d = lead.to_dict(include_samples=False)
        assert "samples" not in d
        assert d["sample_count"] == 3


class TestECGRecord:
    def _make_record(self) -> ECGRecord:
        return ECGRecord(
            source_format="test",
            patient=PatientInfo(patient_id="P001", sex="M"),
            recording=RecordingInfo(sample_rate=500, date=datetime(2023, 1, 1)),
            leads=[
                Lead(label="I", samples=np.array([1.0, 2.0]), sample_rate=500),
                Lead(label="II", samples=np.array([3.0, 4.0]), sample_rate=500),
            ],
            annotations={"heartrate": "75"},
        )

    def test_to_dict_schema(self):
        record = self._make_record()
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "patient", "recording", "device", "filters",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }
        assert d["source_format"] == "test"
        assert len(d["leads"]) == 2

    def test_to_dict_without_samples(self):
        record = self._make_record()
        d = record.to_dict(include_samples=False)
        for lead in d["leads"]:
            assert "samples" not in lead
            assert "sample_count" in lead

    def test_to_json_is_valid(self):
        record = self._make_record()
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "test"
        assert parsed["patient"]["patient_id"] == "P001"

    def test_to_json_compact(self):
        record = self._make_record()
        j = record.to_json(indent=None)
        assert "\n" not in j

    def test_unified_schema_keys_stable(self):
        """All parsers must produce dicts with exactly these top-level keys."""
        record = ECGRecord()
        d = record.to_dict()
        assert list(d.keys()) == [
            "source_format", "patient", "recording", "device", "filters",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        ]

    def test_patient_dict_keys_stable(self):
        d = PatientInfo().to_dict()
        expected = {
            "patient_id", "first_name", "last_name", "birth_date",
            "sex", "race", "age", "weight", "height",
            "medications", "clinical_history",
        }
        assert set(d.keys()) == expected

    def test_recording_dict_keys_stable(self):
        d = RecordingInfo().to_dict()
        expected = {
            "date", "end_date", "duration_seconds", "sample_rate",
            "adc_gain", "technician", "referring_physician",
            "room", "location",
        }
        assert set(d.keys()) == expected

    def test_lead_dict_keys_stable(self):
        lead = Lead(label="I", samples=np.array([1.0]), sample_rate=500)
        d = lead.to_dict()
        expected = {
            "label", "sample_count", "sample_rate", "resolution", "samples",
            "units", "quality", "transducer", "prefiltering",
        }
        assert set(d.keys()) == expected
