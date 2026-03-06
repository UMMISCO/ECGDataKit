"""Tests for ECGDataKit data models and JSON serialisation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from ecgdatakit.exceptions import RawSamplesError
from ecgdatakit.models import (
    AcquisitionSetup,
    DeviceInfo,
    ECGRecord,
    FilterSettings,
    GlobalMeasurements,
    Interpretation,
    Lead,
    PatientInfo,
    RecordingInfo,
    SignalCharacteristics,
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
            "manufacturer", "model", "name", "serial_number", "software_version",
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


class TestSignalCharacteristics:
    def test_defaults(self):
        s = SignalCharacteristics()
        assert s.sample_rate == 0
        assert s.resolution == 0.0
        assert s.bits_per_sample is None

    def test_to_dict(self):
        s = SignalCharacteristics(
            sample_rate=500, resolution=5.0, bits_per_sample=16,
            signal_signed=True, data_encoding="base64_int16le",
        ).to_dict()
        assert s["sample_rate"] == 500
        assert s["resolution"] == 5.0
        assert s["bits_per_sample"] == 16

    def test_dict_keys_stable(self):
        d = SignalCharacteristics().to_dict()
        expected = {
            "sample_rate", "resolution", "bits_per_sample", "signal_offset",
            "signal_signed", "number_channels_allocated", "number_channels_valid",
            "electrode_placement", "compression", "data_encoding", "acsetting",
            "filtered", "downsampled", "upsampled", "waveform_modified",
            "downsampling_method", "upsampling_method",
        }
        assert set(d.keys()) == expected


class TestAcquisitionSetup:
    def test_defaults(self):
        a = AcquisitionSetup()
        assert a.signal.sample_rate == 0
        assert a.filters.highpass is None

    def test_to_dict(self):
        a = AcquisitionSetup(
            signal=SignalCharacteristics(sample_rate=500, resolution=5.0),
            filters=FilterSettings(highpass=0.05, lowpass=150.0),
        )
        d = a.to_dict()
        assert d["signal"]["sample_rate"] == 500
        assert d["signal"]["resolution"] == 5.0
        assert d["filters"]["highpass"] == 0.05

    def test_dict_keys_stable(self):
        d = AcquisitionSetup().to_dict()
        expected = {"signal", "filters"}
        assert set(d.keys()) == expected


class TestInterpretation:
    def test_defaults(self):
        i = Interpretation()
        assert i.statements == []
        assert i.severity == ""
        assert i.source == ""

    def test_to_dict(self):
        d = Interpretation(
            statements=[("Normal sinus rhythm", "")],
            severity="NORMAL",
            source="machine",
            interpreter="Dr. Smith",
            interpretation_date=datetime(2023, 6, 15, 10, 30),
        ).to_dict()
        assert d["statements"] == [["Normal sinus rhythm", ""]]
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
        assert r.duration is None
        assert isinstance(r.device, DeviceInfo)
        assert isinstance(r.acquisition, AcquisitionSetup)
        assert r.acquisition.signal.sample_rate == 0

    def test_to_dict(self):
        r = RecordingInfo(
            date=datetime(2023, 6, 15, 10, 30),
            end_date=datetime(2023, 6, 15, 10, 30, 10),
            duration=timedelta(seconds=10),
        )
        r.acquisition.signal.sample_rate = 500
        d = r.to_dict()
        assert d["date"] == "2023-06-15T10:30:00"
        assert d["duration_seconds"] == 10.0
        assert d["acquisition"]["signal"]["sample_rate"] == 500

    def test_to_dict_null_dates(self):
        d = RecordingInfo().to_dict()
        assert d["date"] is None
        assert d["duration_seconds"] is None

    def test_recording_dict_keys_stable(self):
        d = RecordingInfo().to_dict()
        expected = {
            "date", "end_date", "duration_seconds",
            "technician", "referring_physician",
            "room", "location", "device", "acquisition",
        }
        assert set(d.keys()) == expected


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

    def test_annotations(self):
        lead = Lead(
            label="I", samples=np.array([1.0]), sample_rate=500,
            annotations={"pamp": "100", "pdur": "80"},
        )
        d = lead.to_dict()
        assert d["annotations"] == {"pamp": "100", "pdur": "80"}


class TestECGRecord:
    def _make_record(self) -> ECGRecord:
        rec = RecordingInfo(date=datetime(2023, 1, 1))
        rec.acquisition.signal.sample_rate = 500
        return ECGRecord(
            source_format="test",
            patient=PatientInfo(patient_id="P001", sex="M"),
            recording=rec,
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
            "source_format", "patient", "recording",
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
            "source_format", "patient", "recording",
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
            "date", "end_date", "duration_seconds",
            "technician", "referring_physician",
            "room", "location", "device", "acquisition",
        }
        assert set(d.keys()) == expected

    def test_lead_dict_keys_stable(self):
        lead = Lead(label="I", samples=np.array([1.0]), sample_rate=500)
        d = lead.to_dict()
        expected = {
            "label", "sample_count", "sample_rate", "resolution", "resolution_unit",
            "offset", "samples", "units", "is_raw", "adc_resolution",
            "adc_resolution_unit", "quality", "transducer", "prefiltering",
            "annotations",
        }
        assert set(d.keys()) == expected


class TestLeadConversion:
    def test_to_physical_basic(self):
        lead = Lead(
            label="I", samples=np.array([100.0, 200.0, 300.0]),
            sample_rate=500, resolution=0.005, resolution_unit="mV",
        )
        physical = lead.to_physical()
        assert not physical.is_raw
        np.testing.assert_allclose(physical.samples, [0.5, 1.0, 1.5])
        assert physical.units == "mV"

    def test_to_physical_with_offset(self):
        lead = Lead(
            label="I", samples=np.array([100.0, 200.0]),
            sample_rate=500, resolution=2.0, resolution_unit="uV", offset=-50.0,
        )
        physical = lead.to_physical()
        np.testing.assert_allclose(physical.samples, [150.0, 350.0])

    def test_to_physical_idempotent(self):
        lead = Lead(
            label="I", samples=np.array([1.0, 2.0]),
            sample_rate=500, resolution=0.005, is_raw=False, units="mV",
        )
        result = lead.to_physical()
        assert result is lead

    def test_to_physical_zero_resolution_raises(self):
        lead = Lead(
            label="I", samples=np.array([1.0]),
            sample_rate=500, resolution=0.0,
        )
        with pytest.raises(ValueError, match="resolution is 0"):
            lead.to_physical()

    def test_convert_units_raw_raises(self):
        lead = Lead(
            label="I", samples=np.array([100.0]),
            sample_rate=500, is_raw=True,
        )
        with pytest.raises(RawSamplesError):
            lead.convert_units("mV")

    def test_convert_units_uv_to_mv(self):
        lead = Lead(
            label="I", samples=np.array([1000.0, 2000.0]),
            sample_rate=500, is_raw=False, units="uV",
        )
        result = lead.convert_units("mV")
        np.testing.assert_allclose(result.samples, [1.0, 2.0])
        assert result.units == "mV"

    def test_convert_units_mv_to_v(self):
        lead = Lead(
            label="I", samples=np.array([1.0]),
            sample_rate=500, is_raw=False, units="mV",
        )
        result = lead.convert_units("V")
        np.testing.assert_allclose(result.samples, [0.001])
        assert result.units == "V"

    def test_convert_units_same_returns_self(self):
        lead = Lead(
            label="I", samples=np.array([1.0]),
            sample_rate=500, is_raw=False, units="mV",
        )
        result = lead.convert_units("mV")
        assert result is lead

    def test_convert_units_unknown_target_raises(self):
        lead = Lead(
            label="I", samples=np.array([1.0]),
            sample_rate=500, is_raw=False, units="mV",
        )
        with pytest.raises(ValueError, match="Unknown target unit"):
            lead.convert_units("dB")

    def test_convert_units_unknown_current_raises(self):
        lead = Lead(
            label="I", samples=np.array([1.0]),
            sample_rate=500, is_raw=False, units="nV",
        )
        with pytest.raises(ValueError, match="not a recognized voltage unit"):
            lead.convert_units("mV")

    def test_chaining_to_physical_then_convert(self):
        lead = Lead(
            label="I", samples=np.array([1000.0, 2000.0]),
            sample_rate=500, resolution=1.0, resolution_unit="uV",
        )
        result = lead.to_physical().convert_units("mV")
        np.testing.assert_allclose(result.samples, [1.0, 2.0])
        assert result.units == "mV"
        assert not result.is_raw

    def test_to_dict_includes_new_fields(self):
        lead = Lead(
            label="I", samples=np.array([1.0]),
            sample_rate=500, resolution=0.005, offset=-10.0,
            units="mV", is_raw=True,
        )
        d = lead.to_dict()
        assert d["offset"] == -10.0
        assert d["is_raw"] is True


class TestECGRecordConversion:
    def test_record_to_physical(self):
        record = ECGRecord(
            leads=[
                Lead(label="I", samples=np.array([100.0, 200.0]),
                     sample_rate=500, resolution=0.01, resolution_unit="mV"),
                Lead(label="II", samples=np.array([300.0, 400.0]),
                     sample_rate=500, resolution=0.01, resolution_unit="mV"),
            ],
        )
        physical = record.to_physical()
        assert all(not lead.is_raw for lead in physical.leads)
        np.testing.assert_allclose(physical.leads[0].samples, [1.0, 2.0])

    def test_record_convert_units(self):
        record = ECGRecord(
            leads=[
                Lead(label="I", samples=np.array([1.0]),
                     sample_rate=500, is_raw=False, units="mV"),
            ],
        )
        result = record.convert_units("uV")
        np.testing.assert_allclose(result.leads[0].samples, [1000.0])
        assert result.leads[0].units == "uV"

    def test_record_convert_units_raw_raises(self):
        record = ECGRecord(
            leads=[
                Lead(label="I", samples=np.array([1.0]),
                     sample_rate=500, is_raw=True),
            ],
        )
        with pytest.raises(RawSamplesError):
            record.convert_units("mV")
