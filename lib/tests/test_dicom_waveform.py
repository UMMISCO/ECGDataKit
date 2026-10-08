"""Tests for the DICOM Waveform parser."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.waveforms import multiplex_array

from ecgdatakit.exceptions import CorruptedFileError, MissingElementError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.helpers import STANDARD_LEADS
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.dicom_waveform import DICOMWaveformParser
from tests.conftest import (
    dicom_channel,
    dicom_code,
    dicom_group,
    dicom_12lead_samples,
    write_dicom_ecg,
)


def _single_group(tmp_path: Path, samples: np.ndarray, channels=None, name="g.dcm", **kwargs) -> Path:
    if channels is None:
        channels = [dicom_channel(("2:1", "Lead I"), filters=None) for _ in range(samples.shape[1])]
    group_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in (
        "sampling_rate", "interpretation", "bits", "label", "data", "padding")}
    return write_dicom_ecg(tmp_path / name, [dicom_group(samples, channels, **group_kwargs)], **kwargs)


class TestDICOMWaveformParser:
    def test_parse_returns_ecg_record(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert isinstance(record, ECGRecord)
        assert record.source_format == "dicom_waveform"

    def test_patient(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.patient.patient_id == "DCM001"
        assert record.patient.last_name == "Doe"
        assert record.patient.first_name == "John"
        assert record.patient.sex == "M"
        assert record.patient.birth_date == datetime(1990, 1, 1)

    def test_recording_date_from_study_pair(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert record.recording.date == datetime(2023, 12, 1, 10, 30, 0)

    def test_leads(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert [l.label for l in record.leads] == ["I", "II"]
        assert record.recording.acquisition.signal.sampling_rate == 500
        assert record.recording.duration == timedelta(seconds=0.2)
        lead = record.leads[0]
        assert lead.samples.dtype == np.float64
        assert lead.samples[:3].tolist() == [100.0, 101.0, 102.0]

    def test_no_units_means_raw(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        for lead in record.leads:
            assert lead.resolution == 1.0
            assert lead.resolution_unit == ""
            assert lead.is_raw is True
            assert lead.units == ""

    def test_nonzero_baseline_offset(self, dicom_file_baseline: Path):
        """Channel Baseline is in physical units: sample * sensitivity + baseline."""
        lead = DICOMWaveformParser().parse(dicom_file_baseline).leads[0]
        assert lead.resolution == 2.5
        assert lead.resolution_unit == "uV"
        assert lead.offset == 5.0
        assert lead.is_raw is True
        assert np.allclose(lead.to_physical().samples, 255.0)

    def test_can_parse(self, dicom_file: Path, tmp_path: Path):
        assert DICOMWaveformParser.can_parse(dicom_file, dicom_file.read_bytes()[:4096]) is True
        f = tmp_path / "test.bin"
        f.write_bytes(b"\x00" * 200)
        assert DICOMWaveformParser.can_parse(f, f.read_bytes()) is False

    def test_to_dict_and_json(self, dicom_file: Path):
        record = DICOMWaveformParser().parse(dicom_file)
        assert set(record.to_dict().keys()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }
        parsed = json.loads(record.to_json())
        assert parsed["source_format"] == "dicom_waveform"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, dicom_file: Path):
        with pytest.warns(UserWarning, match="raw ADC"):
            record = FileParser().parse(dicom_file)
        assert record.source_format == "dicom_waveform"


class TestDICOM12Lead:
    def test_labels_are_standard(self, dicom_12lead_file: Path):
        record = DICOMWaveformParser().parse(dicom_12lead_file)
        assert [l.label for l in record.leads] == list(STANDARD_LEADS)
        assert record.leads[3].annotations["source_code"] == "MDC:2:62"

    def test_physical_matches_pydicom(self, dicom_12lead_file: Path):
        ds = pydicom.dcmread(dicom_12lead_file)
        expected = multiplex_array(ds, 0, as_raw=False)
        raw = multiplex_array(ds, 0, as_raw=True)
        record = DICOMWaveformParser().parse(dicom_12lead_file)
        for i, lead in enumerate(record.leads):
            assert np.array_equal(lead.samples, raw[:, i])
            assert np.allclose(lead.to_physical().samples, expected[:, i])
            assert lead.resolution == pytest.approx(2.5 * 1.02)
            assert lead.resolution_unit == "uV"
            assert lead.adc_resolution == 2.5
            assert lead.adc_resolution_unit == "uV"

    def test_auto_scale_to_mv(self, dicom_12lead_file: Path):
        expected = multiplex_array(pydicom.dcmread(dicom_12lead_file), 0, as_raw=False) / 1000
        record = FileParser().parse(dicom_12lead_file)
        assert all(l.units == "mV" and not l.is_raw for l in record.leads)
        assert np.allclose(record.leads[5].samples, expected[:, 5])
        with pytest.warns(UserWarning, match="auto_scale=False"):
            raw = FileParser().parse(dicom_12lead_file, auto_scale=False)
        assert raw.leads[0].samples[0] == 100.0

    def test_median_group(self, dicom_12lead_file: Path):
        record = DICOMWaveformParser().parse(dicom_12lead_file)
        assert len(record.leads[0].samples) == 1000
        assert len(record.median_beats) == 12
        assert record.median_beats[0].sampling_rate == 1000
        assert len(record.median_beats[0].samples) == 600
        assert record.median_beats[2].label == "III"
        labels = [g["label"] for g in record.raw_metadata["multiplex_groups"]]
        assert labels == ["RHYTHM", "MEDIAN BEAT"]

    def test_signal_and_filters(self, dicom_12lead_file: Path):
        acq = DICOMWaveformParser().parse(dicom_12lead_file).recording.acquisition
        assert acq.signal.sampling_rate == 500
        assert acq.signal.bits_per_sample == 16
        assert acq.signal.signal_signed is True
        assert acq.signal.number_channels_valid == 12
        assert acq.filters.highpass == 0.05
        assert acq.filters.lowpass == 150
        assert acq.filters.notch == 50
        assert acq.filters.notch_active is True

    def test_dates(self, dicom_12lead_file: Path):
        record = DICOMWaveformParser().parse(dicom_12lead_file)
        tz = timezone(timedelta(hours=1))
        assert record.recording.date == datetime(2024, 1, 2, 3, 4, 5, 250000, tzinfo=tz)
        assert record.recording.duration == timedelta(seconds=2)
        assert record.recording.end_date is None  # not stated in the file, not computed
        assert record.file_format.creation_date == date(2024, 1, 3)
        assert record.file_format.version == "1"

    def test_patient_age_from_birth_date(self, dicom_12lead_file: Path):
        patient = DICOMWaveformParser().parse(dicom_12lead_file).patient
        assert patient.first_name == "Jane"
        assert patient.last_name == "Doe"
        assert patient.sex == "F"
        assert patient.age == 99  # the stored 099Y wins over the DOB
        assert patient.weight == 61.5
        assert patient.height == 168.0

    def test_device_and_staff(self, dicom_12lead_file: Path):
        rec = DICOMWaveformParser().parse(dicom_12lead_file).recording
        assert rec.device.manufacturer == "ACME"
        assert rec.device.model == "ECG-1"
        assert rec.device.name == "CART7"
        assert rec.device.serial_number == "SN42"
        assert rec.device.software_version == "1.2; 3.4"
        assert rec.device.institution == "General Hospital"
        assert rec.device.department == "Cardiology"
        assert rec.device.acquisition_type == "12-lead ECG Waveform"
        assert rec.technician == "Terry Tech"
        assert rec.referring_physician == "Rita Ref"

    def test_measurements_with_units(self, dicom_12lead_file: Path):
        m = DICOMWaveformParser().parse(dicom_12lead_file).measurements
        assert m.heart_rate == 72
        assert m.pr_interval == 160  # 0.1604 s
        assert m.qrs_duration == 98
        assert m.qt_interval == 398
        assert m.qtc_bazett == 422  # rounded, not truncated
        assert m.qrs_axis == 45

    def test_statements_and_fiducials(self, dicom_12lead_file: Path):
        record = DICOMWaveformParser().parse(dicom_12lead_file)
        assert record.interpretation.statements == [("Sinus rhythm", ""), ("Normal ECG", "")]
        assert record.interpretation.source == "machine"
        fiducials = record.raw_metadata["waveform_annotations"]
        assert fiducials == [{
            "concept": "QRS onset", "ReferencedSamplePositions": "120",
            "code": "2:15844", "channels": ["group 1"],
        }]


class TestDICOMSampleDecoding:
    @pytest.mark.parametrize("interp,bits,dtype,values", [
        ("US", 16, "<u2", [40000, 1, 65535, 2]),
        ("SB", 8, "i1", [-5, 5, -128, 127]),
        ("UB", 8, "u1", [0, 5, 200, 255]),
        ("SL", 32, "<i4", [-70000, 1, 2, 3]),
        ("UL", 32, "<u4", [3000000000, 1, 2, 3]),
        ("SV", 64, "<i8", [-(2 ** 40), 1, 2, 3]),
    ])
    def test_interpretations_match_pydicom(self, tmp_path, interp, bits, dtype, values):
        samples = np.array(values, dtype=dtype).reshape(2, 2)
        path = _single_group(tmp_path, samples, interpretation=interp, bits=bits)
        expected = multiplex_array(pydicom.dcmread(path), 0, as_raw=True)
        record = DICOMWaveformParser().parse(path)
        assert record.leads[0].samples.tolist() == expected[:, 0].astype(float).tolist()
        assert record.leads[1].samples.tolist() == expected[:, 1].astype(float).tolist()
        assert record.leads[0].samples[0] == float(values[0])

    def test_mu_law_and_a_law(self, tmp_path: Path):
        # ITU-T G.711 reference values
        mu = np.array([[0xFF], [0x7F], [0x00], [0x80]], dtype="u1")
        record = DICOMWaveformParser().parse(
            _single_group(tmp_path, mu, name="mu.dcm", interpretation="MB", bits=8))
        assert record.leads[0].samples.tolist() == [0.0, 0.0, -32124.0, 32124.0]
        assert record.recording.acquisition.signal.data_encoding == "mu-law"
        a = np.array([[0xD5], [0x55], [0xAA], [0x2A]], dtype="u1")
        record = DICOMWaveformParser().parse(
            _single_group(tmp_path, a, name="a.dcm", interpretation="AB", bits=8))
        assert record.leads[0].samples.tolist() == [8.0, -8.0, 32256.0, -32256.0]

    def test_big_endian_transfer_syntax(self, tmp_path: Path):
        samples = np.array([[1, -2], [300, 4]], dtype=">i2")
        path = _single_group(tmp_path, samples, little_endian=False)
        record = DICOMWaveformParser().parse(path)
        assert record.leads[0].samples.tolist() == [1.0, 300.0]
        assert record.leads[1].samples.tolist() == [-2.0, 4.0]

    def test_padding_value_becomes_nan(self, tmp_path: Path):
        samples = np.array([[10, 20], [-32768, 21], [12, -32768]], dtype="<i2")
        path = _single_group(tmp_path, samples, padding=np.int16(-32768).tobytes())
        record = DICOMWaveformParser().parse(path)
        assert np.isnan(record.leads[0].samples[1])
        assert record.leads[0].samples[[0, 2]].tolist() == [10.0, 12.0]
        assert np.isnan(record.leads[1].samples[2])

    def test_truncated_data_raises(self, tmp_path: Path):
        samples = np.zeros((4, 2), dtype="<i2")
        path = _single_group(tmp_path, samples, data=b"\x00" * 10)
        with pytest.raises(CorruptedFileError, match="Waveform Data has 10 bytes"):
            DICOMWaveformParser().parse(path)

    def test_unsupported_interpretation_raises(self, tmp_path: Path):
        samples = np.zeros((2, 1), dtype="<i2")
        path = _single_group(tmp_path, samples, interpretation="SB")
        with pytest.raises(CorruptedFileError, match="Bits Allocated 16"):
            DICOMWaveformParser().parse(path)

    def test_missing_waveform_raises(self, tmp_path: Path):
        path = write_dicom_ecg(tmp_path / "empty.dcm", [], PatientID="X")
        with pytest.raises(MissingElementError):
            DICOMWaveformParser().parse(path)

    def test_sampling_rate_rounded(self, tmp_path: Path):
        path = _single_group(tmp_path, np.zeros((4, 1), dtype="<i2"), sampling_rate="999.6")
        record = DICOMWaveformParser().parse(path)
        assert record.leads[0].sampling_rate == 1000
        assert record.raw_metadata["multiplex_groups"][0]["sampling_frequency"] == 999.6


class TestDICOMChannelMetadata:
    def _labels(self, tmp_path: Path, sources) -> list[str]:
        channels = []
        for src in sources:
            ch = dicom_channel(None, filters=None)
            if src is not None:
                ch.ChannelSourceSequence = pydicom.sequence.Sequence([dicom_code(*src)])
            channels.append(ch)
        samples = np.zeros((2, len(channels)), dtype="<i2")
        path = _single_group(tmp_path, samples, channels=channels)
        return [l.label for l in DICOMWaveformParser().parse(path).leads]

    def test_labels_from_codes_and_meanings(self, tmp_path: Path):
        labels = self._labels(tmp_path, [
            ("5.6.3-9-62", "SCPECG", "Lead aVR"),
            ("5.6.3-9-1", "SCPECG", "Lead I (Einthoven)"),
            ("MDC_ECG_LEAD_AVL", "MDC", "Lead aVL"),
            ("X1", "99LOCAL", "Lead V3"),
            ("X2", "99LOCAL", "Lead I (Einthoven)"),
            None,
        ])
        assert labels == ["aVR", "I", "aVL", "V3", "I_2", "Ch6"]

    def test_unit_from_meaning(self, tmp_path: Path):
        ch = dicom_channel(("2:2", "Lead II"), sensitivity="0.005", unit=("", "millivolt"), filters=None)
        del ch.ChannelSensitivityUnitsSequence[0].CodeValue
        path = _single_group(tmp_path, np.full((2, 1), 100, dtype="<i2"), channels=[ch])
        lead = DICOMWaveformParser().parse(path).leads[0]
        assert lead.resolution_unit == "mV"
        assert lead.to_physical().samples.tolist() == [0.5, 0.5]

    def test_non_voltage_unit_is_not_claimed(self, tmp_path: Path):
        ch = dicom_channel(("2:2", "Lead II"), sensitivity="0.1", unit=("mm[Hg]", "mmHg"),
                           baseline="3", filters=None)
        path = _single_group(tmp_path, np.full((2, 1), 100, dtype="<i2"), channels=[ch])
        lead = DICOMWaveformParser().parse(path).leads[0]
        assert lead.resolution == 1.0
        assert lead.resolution_unit == ""
        assert lead.is_raw is True
        assert lead.adc_resolution == 0.1
        assert lead.adc_resolution_unit == "mm[Hg]"
        assert lead.annotations["channel_baseline"] == "3.0"

    def test_date_without_time_is_not_invented(self, tmp_path: Path):
        path = _single_group(tmp_path, np.zeros((2, 1), dtype="<i2"), ContentDate="20240105")
        record = DICOMWaveformParser().parse(path)
        assert record.recording.date is None
        assert record.raw_metadata["acquisition_date"] == "2024-01-05"
        assert record.file_format.creation_date == date(2024, 1, 5)

    def test_additional_rhythm_group_warns(self, tmp_path: Path):
        samples = dicom_12lead_samples(10)[:, :2]
        channels = [dicom_channel(("2:1", "Lead I")), dicom_channel(("2:2", "Lead II"))]
        path = write_dicom_ecg(
            tmp_path / "two.dcm",
            [dicom_group(samples, channels, label="RHYTHM"),
             dicom_group(samples, channels, label="RHYTHM 2")],
        )
        with pytest.warns(UserWarning, match="1 additional multiplex group") as caught:
            record = DICOMWaveformParser().parse(path)
        assert caught[0].filename == __file__
        assert len(record.leads) == 2


def test_missing_sampling_frequency_raises(tmp_path: Path):
    path = _single_group(tmp_path, np.zeros((4, 1), np.int16), sampling_rate="0")
    with pytest.raises(CorruptedFileError, match="Sampling Frequency"):
        DICOMWaveformParser().parse(path)
