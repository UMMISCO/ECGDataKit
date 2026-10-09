"""Tests for the MFER parser (fixtures built from MFER Part 1 v1.05)."""

from __future__ import annotations

import json
import struct
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import CorruptedFileError, UnsupportedFormatError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.mfer import MFERParser
from tests.conftest import (
    create_mfer_binary,
    create_scp_ecg_binary,
    mfer_att,
    mfer_frame_data,
    mfer_scaled,
    mfer_test_signal,
    mfer_tlv,
)

BLE_BIG = mfer_tlv(0x01, b"\x00")
BLE_LITTLE = mfer_tlv(0x01, b"\x01")


def _write(tmp_path: Path, data: bytes, name: str = "t.mwf") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _parse(tmp_path: Path, *parts: bytes) -> ECGRecord:
    return MFERParser().parse(_write(tmp_path, b"".join(parts)))


class TestMFERFixture:
    def test_parse_returns_ecg_record(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert isinstance(record, ECGRecord)
        assert record.source_format == "mfer"

    def test_patient(self, mfer_file: Path):
        patient = MFERParser().parse(mfer_file).patient
        assert patient.patient_id == "MFER001"
        assert patient.last_name == "Doe"
        assert patient.first_name == "John"
        assert patient.sex == "M"
        assert patient.birth_date == datetime(1980, 6, 15)
        assert patient.age == 99  # the stored age wins over the DOB

    def test_recording(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert record.recording.date == datetime(2023, 12, 1, 10, 30, 0, 250000)
        assert record.recording.duration == timedelta(seconds=1)
        assert record.recording.end_date is None  # not stated in the file, not computed
        assert record.file_format.version == "1.5.0"
        assert record.raw_metadata["preamble"] == "MFR Standard 12 leads ECG"
        assert record.raw_metadata["events"] == [
            {"code": 7, "start": 100, "duration": 20, "description": "Calibration"}
        ]

    def test_device_filters_comment(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        device = record.recording.device
        assert (device.manufacturer, device.model, device.software_version, device.serial_number) == (
            "ACME", "ECG-2003", "1.02.33", "SN-77")
        assert device.acquisition_type == "Standard 12-lead ECG"
        filters = record.recording.acquisition.filters
        assert (filters.highpass, filters.lowpass, filters.notch, filters.notch_active) == (
            0.05, 150.0, 50.0, True)
        assert record.annotations["comment"] == "Comment é"

    def test_leads_follow_block_layout(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        expected = mfer_test_signal(2, 500)
        assert [ld.label for ld in record.leads] == ["I", "II"]
        for lead, values in zip(record.leads, expected):
            assert lead.samples.dtype == np.float64
            assert np.array_equal(lead.samples, values.astype(float))
            assert lead.sampling_rate == 500

    def test_scaling(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        lead = record.leads[0]
        assert lead.resolution == 5.0
        assert lead.resolution_unit == "uV"
        assert lead.adc_resolution == 5.0
        assert lead.adc_resolution_unit == "uV"
        assert lead.is_raw is True
        signal = record.recording.acquisition.signal
        assert signal.sampling_rate == 500
        assert signal.resolution == 5.0
        assert signal.data_encoding == "int16"

    def test_file_parser_auto_scale(self, mfer_file: Path):
        record = FileParser().parse(mfer_file)
        assert record.source_format == "mfer"
        expected = mfer_test_signal(2, 500)[1] * 0.005
        assert record.leads[1].units == "mV"
        assert np.allclose(record.leads[1].samples, expected)
        with pytest.warns(UserWarning, match="auto_scale=False"):
            raw = FileParser().parse(mfer_file, auto_scale=False)
        assert raw.leads[0].samples[0] == 100.0

    def test_to_dict_and_json(self, mfer_file: Path):
        record = MFERParser().parse(mfer_file)
        assert set(record.to_dict().keys()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }
        parsed = json.loads(record.to_json())
        assert len(parsed["leads"]) == 2


class TestMFERSpecExamples:
    def test_annex_a_standard_12_lead(self, tmp_path: Path):
        """Part 1 Annex A: BLK=1, 8 channels, 1 ms interval, 1000e-9 V."""
        signal = mfer_test_signal(8, 40)
        record = _parse(
            tmp_path,
            BLE_BIG,
            mfer_tlv(0x08, b"\x01"),
            mfer_tlv(0x0B, bytes([0x01, 0xFD, 0x00, 0x01])),
            mfer_tlv(0x0C, bytes([0x00, 0xF7, 0x03, 0xE8])),
            mfer_tlv(0x04, bytes([0, 0, 0, 1])),
            mfer_tlv(0x05, bytes([0, 0, 0, 8])),
            mfer_tlv(0x06, struct.pack(">I", 40)),
            *[mfer_att(ch, mfer_tlv(0x09, bytes([code])))
              for ch, code in enumerate([1, 2, 3, 4, 5, 6, 7, 8])],
            mfer_tlv(0x1E, mfer_frame_data(signal, 1)),
        )
        assert [ld.label for ld in record.leads] == ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
        lead = record.leads[2]
        assert lead.sampling_rate == 1000
        assert np.array_equal(lead.samples, signal[2].astype(float))
        # 1000 x 10^-9 V = 1 uV per count: samples are already physical
        assert lead.resolution == 1.0
        assert lead.resolution_unit == "uV"
        assert lead.adc_resolution == 1000.0
        assert lead.adc_resolution_unit == "nV"
        assert lead.is_raw is False
        assert lead.units == "uV"

    def test_little_endian_applies_to_header_and_data(self, tmp_path: Path):
        signal = mfer_test_signal(2, 6)
        record = _parse(
            tmp_path,
            BLE_LITTLE,
            mfer_tlv(0x0B, mfer_scaled(0, 0, 250, big=False)),
            mfer_tlv(0x04, struct.pack("<I", 3)),
            mfer_tlv(0x05, struct.pack("<I", 2)),
            mfer_att(1, mfer_tlv(0x09, struct.pack("<H", 61))),
            mfer_tlv(0x1E, mfer_frame_data(signal, 3, "<i2")),
        )
        assert record.leads[0].sampling_rate == 250
        assert record.leads[1].label == "III"
        assert np.array_equal(record.leads[1].samples, signal[1].astype(float))

    def test_channel_block_override(self, tmp_path: Path):
        """Fig. 5-18: parent BLK=2, channel 3 BLK=5, 3 channels, 4 sequences."""
        ch0, ch1 = np.arange(8), np.arange(100, 108)
        ch2 = np.arange(200, 220)
        data = b"".join(
            np.array(ch0[s * 2:s * 2 + 2], ">i2").tobytes()
            + np.array(ch1[s * 2:s * 2 + 2], ">i2").tobytes()
            + np.array(ch2[s * 5:s * 5 + 5], ">i2").tobytes()
            for s in range(4)
        )
        record = _parse(
            tmp_path,
            mfer_tlv(0x04, struct.pack(">I", 2)),
            mfer_tlv(0x05, struct.pack(">I", 3)),
            mfer_tlv(0x06, struct.pack(">I", 4)),
            mfer_att(2, mfer_tlv(0x04, struct.pack(">I", 5)),
                     mfer_tlv(0x0B, mfer_scaled(0, 0, 2500))),
            mfer_tlv(0x1E, data),
        )
        assert record.leads[0].samples.tolist() == list(map(float, ch0))
        assert record.leads[1].samples.tolist() == list(map(float, ch1))
        assert record.leads[2].samples.tolist() == list(map(float, ch2))
        assert record.leads[2].sampling_rate == 2500

    def test_channel_sequence_override(self, tmp_path: Path):
        """Fig. 5-19: parent SEQ=4, channel 3 SEQ=2."""
        ch = [np.arange(c * 100, c * 100 + 20) for c in range(3)]
        data = bytearray()
        for s in range(4):
            for c in range(3):
                if c == 2 and s >= 2:
                    continue
                data += np.array(ch[c][s * 5:s * 5 + 5], ">i2").tobytes()
        record = _parse(
            tmp_path,
            mfer_tlv(0x04, struct.pack(">I", 5)),
            mfer_tlv(0x05, struct.pack(">I", 3)),
            mfer_tlv(0x06, struct.pack(">I", 4)),
            mfer_att(2, mfer_tlv(0x06, struct.pack(">I", 2))),
            mfer_tlv(0x1E, bytes(data)),
        )
        assert record.leads[1].samples.tolist() == list(map(float, ch[1]))
        assert record.leads[2].samples.tolist() == list(map(float, ch[2][:10]))

    def test_indefinite_length_attribute(self, tmp_path: Path):
        record = _parse(
            tmp_path,
            mfer_tlv(0x05, b"\x02"),
            mfer_att(0, mfer_tlv(0x09, b"\x3E"), indefinite=True),
            mfer_att(1, mfer_tlv(0x09, b"\x40")),
            mfer_tlv(0x1E, mfer_frame_data(mfer_test_signal(2, 4), 1)),
        )
        assert [ld.label for ld in record.leads] == ["aVR", "aVF"]

    def test_channel_count_resets_attributes(self, tmp_path: Path):
        record = _parse(
            tmp_path,
            mfer_tlv(0x05, b"\x01"),
            mfer_att(0, mfer_tlv(0x09, b"\x01")),
            mfer_tlv(0x05, b"\x01"),
            mfer_tlv(0x1E, b"\x00\x01"),
        )
        assert record.leads[0].label == "Ch1"

    def test_parent_lead_name_applies_to_channel_one(self, tmp_path: Path):
        record = _parse(
            tmp_path,
            mfer_tlv(0x05, b"\x02"),
            mfer_tlv(0x09, b"\x02"),
            mfer_tlv(0x1E, b"\x00\x01\x00\x02"),
        )
        assert [ld.label for ld in record.leads] == ["II", "Ch2"]

    def test_duplicate_lead_codes_are_unique(self, tmp_path: Path):
        record = _parse(
            tmp_path,
            mfer_tlv(0x05, b"\x02"),
            mfer_att(0, mfer_tlv(0x09, b"\x01")),
            mfer_att(1, mfer_tlv(0x09, b"\x01")),
            mfer_tlv(0x1E, b"\x00\x01\x00\x02"),
        )
        assert [ld.label for ld in record.leads] == ["I", "I_2"]

    @pytest.mark.parametrize("dtp,dtype", [
        (1, ">u2"), (2, ">i4"), (3, "u1"), (5, "i1"), (6, ">u4"), (7, ">f4"), (8, ">f8"),
    ])
    def test_data_types(self, tmp_path: Path, dtp: int, dtype: str):
        info = np.iinfo(dtype) if np.dtype(dtype).kind in "iu" else None
        values = np.array([info.min, info.max, 1, 2] if info else [-1.5, 2.25, 0.0, 1e3], dtype=dtype)
        record = _parse(tmp_path, mfer_tlv(0x0A, bytes([dtp])), mfer_tlv(0x1E, values.tobytes()))
        assert record.leads[0].samples.tolist() == values.astype(float).tolist()

    def test_offset_and_null(self, tmp_path: Path):
        values = np.array([10, -32768, 30], ">i2")
        record = _parse(
            tmp_path,
            mfer_tlv(0x0C, mfer_scaled(0, -6, 2)),
            mfer_tlv(0x0D, struct.pack(">h", -10)),
            mfer_tlv(0x12, struct.pack(">h", -32768)),
            mfer_tlv(0x1E, values.tobytes()),
        )
        lead = record.leads[0]
        assert lead.samples[0] == 10.0 and np.isnan(lead.samples[1])
        assert lead.offset == -20.0  # physical = (sample + OFF) * SEN
        assert lead.to_physical().samples[2] == 40.0

    def test_frequency_unit_and_non_voltage_sensitivity(self, tmp_path: Path):
        record = _parse(
            tmp_path,
            mfer_tlv(0x0B, mfer_scaled(0, 2, 2)),      # 200 Hz
            mfer_tlv(0x0C, mfer_scaled(1, -1, 5)),     # 0.5 mmHg
            mfer_tlv(0x09, struct.pack(">H", 143)),
            mfer_tlv(0x1E, b"\x00\x01"),
        )
        lead = record.leads[0]
        assert lead.sampling_rate == 200
        assert lead.label == "Blood pressure"
        assert lead.resolution == 1.0 and lead.resolution_unit == "" and lead.is_raw
        assert lead.adc_resolution == 0.5 and lead.adc_resolution_unit == "mmHg"

    def test_status_channel_kept(self, tmp_path: Path):
        status = np.array([0, 0, 0b100, 0], ">u2")
        ecg = np.array([5, 6, 7, 8], ">i2")
        record = _parse(
            tmp_path,
            mfer_tlv(0x04, b"\x04"),
            mfer_tlv(0x05, b"\x02"),
            mfer_att(0, mfer_tlv(0x09, b"\x01")),
            mfer_att(1, mfer_tlv(0x09, struct.pack(">H", 4160)), mfer_tlv(0x0A, b"\x04")),
            mfer_tlv(0x1E, ecg.tobytes() + status.tobytes()),
        )
        assert [ld.label for ld in record.leads] == ["I"]
        assert record.patient.has_pacemaker is None  # pacing detections are not a patient attribute
        assert record.raw_metadata["status_channels"]["Status"].tolist() == [0, 0, 4, 0]

    def test_dominant_beat_frame_goes_to_median_beats(self, tmp_path: Path):
        record = _parse(
            tmp_path,
            mfer_tlv(0x08, b"\x01"),
            mfer_tlv(0x1E, np.arange(6, dtype=">i2").tobytes()),
            mfer_tlv(0x08, b"\x09"),
            mfer_tlv(0x1E, np.arange(3, dtype=">i2").tobytes()),
        )
        assert len(record.leads[0].samples) == 6
        assert record.median_beats[0].samples.tolist() == [0.0, 1.0, 2.0]

    def test_frames_are_concatenated(self, tmp_path: Path):
        record = _parse(
            tmp_path,
            mfer_tlv(0x1E, np.array([1, 2], ">i2").tobytes()),
            mfer_tlv(0x1E, np.array([3, 4], ">i2").tobytes()),
        )
        assert record.leads[0].samples.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_text_encoding(self, tmp_path: Path):
        record = _parse(
            tmp_path,
            mfer_tlv(0x03, b"ISO-8859-1"),
            mfer_tlv(0x81, "Müller^^Jürgen".encode("latin-1")),
            mfer_tlv(0x1E, b"\x00\x01"),
        )
        assert record.patient.last_name == "Müller"
        assert record.patient.first_name == "Jürgen"

    def test_end_tag_stops_parsing(self, tmp_path: Path):
        record = _parse(
            tmp_path, mfer_tlv(0x1E, b"\x00\x01"), b"\x80", mfer_tlv(0x82, b"AFTER"),
        )
        assert record.patient.patient_id == ""


class TestMFERErrors:
    def test_short_data_is_nan_with_warning(self, tmp_path: Path):
        path = _write(tmp_path, b"".join([
            mfer_tlv(0x05, b"\x02"), mfer_tlv(0x06, b"\x03"),
            mfer_tlv(0x1E, np.array([1, 2, 3, 4, 5], ">i2").tobytes()),
        ]))
        with pytest.warns(UserWarning, match="1 missing sample") as caught:
            record = MFERParser().parse(path)
        assert caught[0].filename == __file__
        assert record.leads[0].samples.tolist() == [1.0, 3.0, 5.0]
        assert record.leads[1].samples[:2].tolist() == [2.0, 4.0]
        assert np.isnan(record.leads[1].samples[2])

    def test_extra_data_is_discarded(self, tmp_path: Path):
        record = _parse(
            tmp_path, mfer_tlv(0x06, b"\x02"), mfer_tlv(0x1E, np.arange(5, dtype=">i2").tobytes()),
        )
        assert record.leads[0].samples.tolist() == [0.0, 1.0]

    def test_no_waveform_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError, match="no waveform"):
            _parse(tmp_path, BLE_BIG, mfer_tlv(0x05, b"\x02"))

    def test_truncated_value_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError, match="declares"):
            _parse(tmp_path, create_mfer_binary()[:-200])

    def test_oversized_length_field_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError, match="at most 4"):
            _parse(tmp_path, b"\x1e\x88" + b"\xff" * 8)

    def test_aha_compression_unsupported(self, tmp_path: Path):
        with pytest.raises(UnsupportedFormatError):
            _parse(tmp_path, mfer_tlv(0x0A, b"\x09"), mfer_tlv(0x1E, b"\x00"))

    def test_compressed_content_unsupported(self, tmp_path: Path):
        with pytest.raises(UnsupportedFormatError):
            _parse(tmp_path, mfer_tlv(0x0E, b"\x00\x03\x00\x00\x00\x00"))


class TestMFERDetection:
    def test_detects_fixture(self, mfer_file: Path):
        assert MFERParser.can_parse(mfer_file, mfer_file.read_bytes()[:4096]) is True

    def test_detects_without_preamble(self, tmp_path: Path):
        data = create_mfer_binary()
        data = data[data.index(b"\x01\x01\x00"):]
        assert MFERParser.can_parse(tmp_path / "x.mwf", data[:4096]) is True

    def test_rejects_non_mfer(self, tmp_path: Path):
        assert MFERParser.can_parse(tmp_path / "x", b"\xFF\xFF\xFF\xFF") is False
        assert MFERParser.can_parse(tmp_path / "x", b"<?xml version='1.0'?><a/>") is False

    def test_scp_files_are_not_claimed(self, tmp_path: Path):
        scp = create_scp_ecg_binary()
        for crc_low in range(256):
            assert MFERParser.can_parse(tmp_path / "x.scp", bytes([crc_low]) + scp[1:4096]) is False
        path = _write(tmp_path, bytes([0x07]) + scp[1:], "x.scp")
        assert FileParser().parse(path, auto_scale=False).source_format != "mfer"


def test_missing_interval_uses_mfer_default(tmp_path: Path):
    with pytest.warns(UserWarning, match="MFER default of 1000 Hz"):
        record = _parse(
            tmp_path,
            mfer_tlv(0x04, b"\x02"),
            mfer_tlv(0x05, b"\x01"),
            mfer_tlv(0x1E, np.array([5, 6], ">i2").tobytes()),
        )
    assert record.leads[0].sampling_rate == 1000
    assert record.raw_metadata["sampling_rate_stated"] is False


# ---------------------------------------------------------------------------
# Frames that redefine their definitions (Part 1 5.2: definitions override
# in order) and empty or non-ECG content
# ---------------------------------------------------------------------------

IVL_2MS = mfer_tlv(0x0B, mfer_scaled(1, -3, 2))


class TestMFERFrameDefinitions:
    def test_redefined_scale_is_not_joined(self, tmp_path: Path):
        data = np.arange(4, dtype=">i2").tobytes()
        path = _write(tmp_path, b"".join([
            IVL_2MS, mfer_tlv(0x0C, mfer_scaled(0, -6, 1)), mfer_tlv(0x1E, data),
            mfer_tlv(0x0C, mfer_scaled(0, -6, 10)), mfer_tlv(0x1E, data),
        ]))
        with pytest.warns(UserWarning, match="1 frame\\(s\\) after frame 1 redefine") as caught:
            record = MFERParser().parse(path)
        assert caught[0].filename == __file__
        lead = record.leads[0]
        assert lead.samples.tolist() == [0.0, 1.0, 2.0, 3.0]
        assert lead.resolution == 1.0 and lead.resolution_unit == "uV"
        assert record.recording.duration == timedelta(milliseconds=8)
        assert [f["samples"] for f in record.raw_metadata["frames"]] == [4, 4]

    def test_redefined_interval_is_not_joined(self, tmp_path: Path):
        data = np.arange(4, dtype=">i2").tobytes()
        path = _write(tmp_path, b"".join([
            IVL_2MS, mfer_tlv(0x1E, data),
            mfer_tlv(0x0B, mfer_scaled(1, -3, 4)), mfer_tlv(0x1E, data),
        ]))
        with pytest.warns(UserWarning, match="redefine"):
            record = MFERParser().parse(path)
        assert record.leads[0].sampling_rate == 500
        assert len(record.leads[0].samples) == 4

    def test_repeated_identical_definitions_are_joined(self, tmp_path: Path):
        sen = mfer_tlv(0x0C, mfer_scaled(0, -6, 1))
        record = _parse(
            tmp_path, IVL_2MS, sen, mfer_tlv(0x1E, np.array([1, 2], ">i2").tobytes()),
            sen, mfer_tlv(0x1E, np.array([3, 4], ">i2").tobytes()),
        )
        assert record.leads[0].samples.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_empty_frame_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError, match="no samples"):
            _parse(tmp_path, IVL_2MS, mfer_tlv(0x1E, b""))
        with pytest.raises(CorruptedFileError, match="no samples"):
            _parse(tmp_path, IVL_2MS, mfer_tlv(0x06, b"\x00"), mfer_tlv(0x1E, bytes(8)))

    def test_status_only_is_unsupported(self, tmp_path: Path):
        with pytest.raises(UnsupportedFormatError, match="only status channels"):
            _parse(tmp_path, IVL_2MS, mfer_tlv(0x0A, b"\x04"), mfer_tlv(0x1E, bytes(8)))

    def test_rate_below_1_hz_is_unsupported(self, tmp_path: Path):
        # Trend data sampled once a minute (MWF_IVL 60 s)
        with pytest.raises(UnsupportedFormatError, match="below 1 Hz"):
            _parse(tmp_path, mfer_tlv(0x0B, mfer_scaled(1, 0, 60)), mfer_tlv(0x1E, bytes(8)))

    def test_iso_8859_part_is_used(self, tmp_path: Path):
        record = _parse(
            tmp_path, IVL_2MS, mfer_tlv(0x03, b"ISO 8859-5"),
            mfer_tlv(0x81, "Иванов".encode("iso8859_5")), mfer_tlv(0x1E, b"\x00\x01"),
        )
        assert record.patient.last_name == "Иванов"

    def test_parser_instance_keeps_no_state(self, tmp_path: Path):
        parser = MFERParser()
        path = _write(tmp_path, b"".join([IVL_2MS, mfer_tlv(0x82, b"P1"), mfer_tlv(0x1E, b"\x00\x01")]))
        assert parser.parse(path).patient.patient_id == "P1"
        assert not hasattr(parser, "_record")


def test_three_frames_with_redefined_third(tmp_path: Path):
    # Frames 1 and 2 share definitions, frame 3 changes the scale
    data = np.arange(4, dtype=">i2").tobytes()
    sen = mfer_tlv(0x0C, mfer_scaled(0, -6, 1))
    path = _write(tmp_path, b"".join([
        IVL_2MS, sen, mfer_tlv(0x1E, data), sen, mfer_tlv(0x1E, data),
        mfer_tlv(0x0C, mfer_scaled(0, -6, 10)), mfer_tlv(0x1E, data),
    ]))
    with pytest.warns(UserWarning, match="1 frame\\(s\\) after frame 2 redefine"):
        record = MFERParser().parse(path)
    assert record.leads[0].samples.tolist() == [0.0, 1.0, 2.0, 3.0] * 2
