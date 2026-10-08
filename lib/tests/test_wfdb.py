"""Tests for the WFDB (PhysioNet) parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.wfdb import WFDBParser


class TestWFDBParser:
    def test_parse_returns_ecg_record(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.source_format == "wfdb"

    def test_patient_age(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.patient.age == 43

    def test_patient_sex(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.patient.sex == "M"

    def test_recording_sampling_rate(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.recording.acquisition.signal.sampling_rate == 500

    def test_recording_date(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.recording.date is not None
        assert record.recording.date.hour == 10
        assert record.recording.date.minute == 30

    def test_recording_duration(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() == 1.0

    def test_lead_count(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_samples_are_float(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) > 0

    def test_lead_sampling_rate(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        for lead in record.leads:
            assert lead.sampling_rate == 500

    def test_lead_units_and_is_raw(self, wfdb_file: Path):
        """Fixture has gain=200 → resolution=0.005 → raw ADC."""
        record = WFDBParser().parse(wfdb_file)
        for lead in record.leads:
            assert lead.resolution_unit == "mV"
            assert lead.units == ""
            assert lead.resolution == pytest.approx(0.005)
            assert lead.is_raw is True

    def test_can_parse_requires_hea_extension(self, tmp_path: Path):
        f = tmp_path / "test.dat"
        f.write_bytes(b"\x00" * 100)
        header = f.read_bytes()[:4096]
        assert WFDBParser.can_parse(f, header) is False

    def test_can_parse_detects_hea_file(self, wfdb_file: Path):
        header = wfdb_file.read_bytes()[:4096]
        assert WFDBParser.can_parse(wfdb_file, header) is True

    def test_to_dict_unified_schema(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, wfdb_file: Path):
        record = WFDBParser().parse(wfdb_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "wfdb"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, wfdb_file: Path):
        fp = FileParser()
        record = fp.parse(wfdb_file)
        assert record.source_format == "wfdb"


# ---------------------------------------------------------------------------
# Spec-based fixtures (https://physionet.org/physiotools/wag/signal-5.htm)
# ---------------------------------------------------------------------------

import struct
import warnings

from ecgdatakit.exceptions import (
    ChecksumWarning,
    CorruptedFileError,
    UnsupportedFormatError,
)


def _write(directory: Path, name: str, lines: list[str], files: dict[str, bytes]) -> Path:
    for fname, data in files.items():
        (directory / fname).write_bytes(data)
    hea = directory / f"{name}.hea"
    hea.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hea


def _enc_212(values: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(values), 2):
        s0 = values[i] & 0xFFF
        s1 = values[i + 1] & 0xFFF if i + 1 < len(values) else 0
        out += bytes([s0 & 0xFF, (s0 >> 8) | ((s1 >> 8) << 4), s1 & 0xFF])
    return bytes(out)


def _enc_310(values: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(values), 3):
        s0, s1, s2 = (v & 0x3FF for v in values[i:i + 3])
        w0 = (s0 << 1) | ((s2 & 0x1F) << 11)
        w1 = (s1 << 1) | ((s2 >> 5) << 11)
        out += struct.pack("<HH", w0, w1)
    return bytes(out)


def _enc_311(values: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(values), 3):
        s0, s1, s2 = (v & 0x3FF for v in values[i:i + 3])
        out += struct.pack("<I", s0 | (s1 << 10) | (s2 << 20))
    return bytes(out)


# Interleaved values of two signals: frames (a0, b0), (a1, b1), (a2, b2)
_VALUES = [100, -200, -2048 + 1, 2047, 0, -1]


def _encode(fmt: int, values: list[int]) -> bytes:
    if fmt == 16:
        return struct.pack(f"<{len(values)}h", *values)
    if fmt == 61:
        return struct.pack(f">{len(values)}h", *values)
    if fmt == 160:
        return struct.pack(f"<{len(values)}H", *(v + 32768 for v in values))
    if fmt == 80:
        return bytes(v + 128 for v in values)
    if fmt == 24:
        return b"".join(struct.pack("<i", v)[:3] for v in values)
    if fmt == 32:
        return struct.pack(f"<{len(values)}i", *values)
    if fmt == 212:
        return _enc_212(values)
    if fmt == 310:
        return _enc_310(values)
    if fmt == 311:
        return _enc_311(values)
    raise AssertionError(fmt)


# Hand-written headers carry placeholder checksums
@pytest.mark.filterwarnings("ignore::ecgdatakit.exceptions.ChecksumWarning")
class TestWFDBFormats:
    @pytest.mark.parametrize("fmt", [16, 61, 160, 80, 24, 32, 212, 310, 311])
    def test_amplitude_formats(self, tmp_path: Path, fmt: int):
        values = _VALUES if fmt not in (80, 310, 311) else [100, -120, -127, 127, 0, -1]
        if fmt in (310, 311):
            values = [100, -200, -511, 511, 0, -1]
        hea = _write(tmp_path, "r", [
            "r 2 250 3",
            f"r.dat {fmt} 200/mV 12 0 0 0 0 I",
            f"r.dat {fmt} 200/mV 12 0 0 0 0 II",
        ], {"r.dat": _encode(fmt, values)})
        record = WFDBParser().parse(hea)
        assert record.leads[0].samples.tolist() == values[0::2]
        assert record.leads[1].samples.tolist() == values[1::2]

    def test_format_212_second_sample(self, tmp_path: Path):
        # MIT-BIH style: 0x3F3 = 1011 in the high nibble of byte 1 + byte 2
        hea = _write(tmp_path, "r", [
            "r 2 360 1", "r.dat 212 200 11 1024 0 0 0 MLII", "r.dat 212 200 11 1024 0 0 0 V5",
        ], {"r.dat": bytes([0xE3, 0x33, 0xF3])})
        record = WFDBParser().parse(hea)
        assert record.leads[0].samples.tolist() == [995]
        assert record.leads[1].samples.tolist() == [1011]

    def test_format_212_odd_sample_count_drops_padding(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 3", "r.dat 212 200/mV 12 0 0 0 0 I"],
                     {"r.dat": _enc_212([1, 2, 3])})
        assert WFDBParser().parse(hea).leads[0].samples.tolist() == [1, 2, 3]

    def test_format_8_differences(self, tmp_path: Path):
        diffs = [5, -3, 10, 1]
        hea = _write(tmp_path, "r", ["r 1 250 4", "r.dat 8 200/mV 12 0 100 0 0 I"],
                     {"r.dat": struct.pack("4b", *diffs)})
        assert WFDBParser().parse(hea).leads[0].samples.tolist() == [105, 102, 112, 113]

    def test_flac_format_unsupported(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 4", "r.dat 516 200/mV 16 0 0 0 0 I"],
                     {"r.dat": b"fLaC"})
        with pytest.raises(UnsupportedFormatError):
            WFDBParser().parse(hea)

    def test_unknown_format_unsupported(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 2", "r.dat 999 200/mV 16 0 0 0 0 I"],
                     {"r.dat": b"\0" * 4})
        with pytest.raises(UnsupportedFormatError):
            WFDBParser().parse(hea)


@pytest.mark.filterwarnings("ignore::ecgdatakit.exceptions.ChecksumWarning")
class TestWFDBSignalLayout:
    def test_byte_offset_skew_and_samples_per_frame(self, tmp_path: Path):
        # Frame = I (2 samples), II (1, skew 1), III (1); 8-byte preamble
        frames = [[i * 10, i * 10 + 1, 500 + i, -i] for i in range(4)]
        data = b"PREAMBLE" + struct.pack("<16h", *[v for f in frames for v in f])
        hea = _write(tmp_path, "r", [
            "r 3 100 3",
            "r.dat 16x2+8 200/mV 16 0 0 0 0 I",
            "r.dat 16:1+8 200/mV 16 0 0 0 0 II",
            "r.dat 16+8 200/mV 16 0 0 0 0 III",
        ], {"r.dat": data})
        record = WFDBParser().parse(hea)
        i, ii, iii = record.leads
        assert i.samples.tolist() == [0, 1, 10, 11, 20, 21]
        assert i.sampling_rate == 200
        assert ii.samples.tolist() == [501, 502, 503]
        assert iii.samples.tolist() == [0, -1, -2]
        assert ii.sampling_rate == 100

    def test_signals_keep_header_order_across_files(self, tmp_path: Path):
        hea = _write(tmp_path, "r", [
            "r 3 250 2",
            "a.dat 16 200/mV 16 0 0 0 0 I",
            "b.dat 16 200/mV 16 0 0 0 0 II",
            "a.dat 16 200/mV 16 0 0 0 0 III",
        ], {"a.dat": struct.pack("<4h", 1, 3, 2, 4), "b.dat": struct.pack("<2h", 7, 8)})
        record = WFDBParser().parse(hea)
        assert [l.label for l in record.leads] == ["I", "II", "III"]
        assert record.leads[1].samples.tolist() == [7, 8]
        assert record.leads[2].samples.tolist() == [3, 4]

    def test_header_sample_count_trims_extra_data(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 2", "r.dat 16 200/mV 16 0 0 0 0 I"],
                     {"r.dat": struct.pack("<4h", 1, 2, 3, 4)})
        assert WFDBParser().parse(hea).leads[0].samples.tolist() == [1, 2]

    def test_short_signal_file_raises(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 10", "r.dat 16 200/mV 16 0 0 0 0 I"],
                     {"r.dat": struct.pack("<4h", 1, 2, 3, 4)})
        with pytest.raises(CorruptedFileError):
            WFDBParser().parse(hea)

    def test_missing_signal_file_raises(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 2", "nope.dat 16 200/mV 16 0 0 0 0 I"], {})
        with pytest.raises(CorruptedFileError):
            WFDBParser().parse(hea)

    @pytest.mark.parametrize("lines", [
        ["r X 250 2"],
        ["r 2 250 2", "r.dat 16 200/mV 16 0 0 0 0 I"],
        ["r 1 250 2", "r.dat 16 abc/mV 16 0 0 0 0 I"],
        ["r 1 250 2 25:61:00 01/13/2020", "r.dat 16 200/mV 16 0 0 0 0 I"],
    ])
    def test_corrupt_header_raises(self, tmp_path: Path, lines: list[str]):
        hea = _write(tmp_path, "r", lines, {"r.dat": b"\0" * 4})
        with pytest.raises(CorruptedFileError):
            WFDBParser().parse(hea)


@pytest.mark.filterwarnings("ignore::ecgdatakit.exceptions.ChecksumWarning")
class TestWFDBMultiSegment:
    def _segments(self, tmp_path: Path):
        _write(tmp_path, "s1", ["s1 2 100 2", "s1.dat 16 200/mV 16 0 0 0 0 I",
                                "s1.dat 16 200/mV 16 0 0 0 0 II"],
               {"s1.dat": struct.pack("<4h", 1, 2, 3, 4)})
        _write(tmp_path, "s2", ["s2 2 100 1", "s2.dat 212 200/mV 12 0 0 0 0 I",
                                "s2.dat 212 200/mV 12 0 0 0 0 II"],
               {"s2.dat": _enc_212([5, 6])})

    def test_fixed_layout_with_null_segment(self, tmp_path: Path):
        self._segments(tmp_path)
        hea = _write(tmp_path, "m", ["m/3 2 100 5", "s1 2", "~ 2", "s2 1"], {})
        record = WFDBParser().parse(hea)
        i, ii = record.leads
        np.testing.assert_array_equal(i.samples, [1, 3, np.nan, np.nan, 5])
        np.testing.assert_array_equal(ii.samples, [2, 4, np.nan, np.nan, 6])
        assert record.recording.duration.total_seconds() == pytest.approx(0.05)
        assert record.file_format.version == "WFDB multi-segment format 16/212"

    def test_variable_layout_rescales(self, tmp_path: Path):
        self._segments(tmp_path)
        _write(tmp_path, "v", ["v 1 100 2", "v.dat 16 400(10)/mV 16 0 0 0 0 II"],
               {"v.dat": struct.pack("<2h", 410, 810)})
        _write(tmp_path, "layout", ["layout 2 100 0", "~ 0 200/mV 16 0 0 0 0 I",
                                    "~ 0 200/mV 16 0 0 0 0 II"], {})
        hea = _write(tmp_path, "m", ["m/3 2 100 4", "layout 0", "s1 2", "v 2"], {})
        record = WFDBParser().parse(hea)
        i, ii = record.leads
        np.testing.assert_array_equal(i.samples, [1, 3, np.nan, np.nan])
        # (410 - 10) / 400 mV = 1 mV = 200 counts at the layout gain
        np.testing.assert_allclose(ii.samples, [2, 4, 200, 400])


@pytest.mark.filterwarnings("ignore::ecgdatakit.exceptions.ChecksumWarning")
class TestWFDBScaling:
    def test_baseline_defaults_to_adc_zero(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 360 2", "r.dat 16 200 11 1024 0 0 0 MLII"],
                     {"r.dat": struct.pack("<2h", 1024, 1224)})
        lead = WFDBParser().parse(hea).leads[0]
        np.testing.assert_allclose(lead.samples * lead.resolution + lead.offset, [0.0, 1.0])
        assert lead.resolution_unit == "mV"
        assert lead.adc_resolution == 200.0
        assert lead.adc_resolution_unit == "adu/mV"

    def test_explicit_baseline_and_units(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 1", "r.dat 16 1000(-50)/uV 16 0 0 0 0 I"],
                     {"r.dat": struct.pack("<h", 950)})
        lead = WFDBParser().parse(hea).leads[0]
        assert lead.resolution_unit == "uV"
        assert lead.samples[0] * lead.resolution + lead.offset == pytest.approx(1.0)

    def test_gain_zero_is_uncalibrated(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 1", "r.dat 16 0/mV 16 0 0 0 0 I"],
                     {"r.dat": struct.pack("<h", 7)})
        lead = WFDBParser().parse(hea).leads[0]
        assert lead.is_raw and lead.resolution_unit == "" and lead.units == ""

    def test_auto_scale_gives_mv_and_raw_without(self, wfdb_file: Path):
        scaled = FileParser().parse(wfdb_file)
        assert scaled.leads[0].units == "mV"
        assert scaled.leads[0].samples[0] == pytest.approx(100 / 200)
        with pytest.warns(UserWarning, match="raw ADC"):
            raw = FileParser().parse(wfdb_file, auto_scale=False)
        assert raw.leads[0].samples[0] == 100


class TestWFDBMetadata:
    def test_checksum_valid_on_fixture(self, wfdb_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            record = WFDBParser().parse(wfdb_file)
        assert record.raw_metadata["checksum_valid"] is True

    def test_checksum_mismatch_warns(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 2", "r.dat 16 200/mV 16 0 1 99 0 I"],
                     {"r.dat": struct.pack("<2h", 1, 2)})
        with pytest.warns(ChecksumWarning) as rec:
            record = WFDBParser().parse(hea)
        assert record.raw_metadata["checksum_valid"] is False
        assert rec[0].filename == __file__

    def test_unsigned_checksum_accepted(self, tmp_path: Path):
        # 40000 written unsigned is -25536 as a signed 16-bit sum
        hea = _write(tmp_path, "r", ["r 1 250 2", "r.dat 16 200/mV 16 0 0 40000 0 I"],
                     {"r.dat": struct.pack("<2h", 20000, 20000)})
        assert WFDBParser().parse(hea).raw_metadata["checksum_valid"] is True

    def test_labels_normalised_and_unique(self, tmp_path: Path):
        names = ["i", "AVR", "avl", "V1", "V1"]
        hea = _write(tmp_path, "r", ["r 5 500 1"] + [
            f"r.dat 16 200/mV 16 0 0 0 0 {n}" for n in names
        ], {"r.dat": b"\0" * 10})
        record = WFDBParser().parse(hea)
        assert [l.label for l in record.leads] == ["I", "aVR", "aVL", "V1", "V1_2"]

    def test_fractional_sampling_rate(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 128.6 2", "r.dat 16 200/mV 16 0 0 0 0 I"],
                     {"r.dat": b"\0" * 4})
        record = WFDBParser().parse(hea)
        assert record.leads[0].sampling_rate == 129
        assert record.raw_metadata["sampling_frequency"] == 128.6

    def test_base_time_fraction_and_time_without_date(self, tmp_path: Path):
        hea = _write(tmp_path, "r", ["r 1 250 2 10:30:05.250 01/12/2023",
                                     "r.dat 16 200/mV 16 0 0 0 0 I"], {"r.dat": b"\0" * 4})
        assert WFDBParser().parse(hea).recording.date.microsecond == 250000
        hea = _write(tmp_path, "r", ["r 1 250 2 10:30:05", "r.dat 16 200/mV 16 0 0 0 0 I"],
                     {"r.dat": b"\0" * 4})
        record = WFDBParser().parse(hea)
        assert record.recording.date is None
        assert record.raw_metadata["base_time"] == "10:30:05"

    def test_ludb_style_comments(self, tmp_path: Path):
        hea = _write(tmp_path, "r", [
            "r 1 500 2", "r.dat 16 200/mV 16 0 0 0 0 i",
            "#<age>: 51", "#<sex>: F", "#<diagnoses>:",
            "#Rhythm: Sinus rhythm.", "#Left ventricular hypertrophy.",
        ], {"r.dat": b"\0" * 4})
        record = WFDBParser().parse(hea)
        assert record.patient.age == 51
        assert record.patient.sex == "F"
        assert record.interpretation.statements == [
            ("Rhythm: Sinus rhythm.", ""), ("Left ventricular hypertrophy.", "")]

    def test_mit_style_comment_and_utf8(self, tmp_path: Path):
        hea = _write(tmp_path, "r", [
            "r 1 360 2", "r.dat 212 200 11 1024 0 0 0 MLII",
            "# 69 M 1085 1629 x1", "# Name: José Müller",
        ], {"r.dat": _enc_212([0, 0])})
        record = WFDBParser().parse(hea)
        assert (record.patient.age, record.patient.sex) == (69, "M")
        assert record.patient.last_name == "Müller"
        assert record.file_format.version == "WFDB format 212"

    def test_dat_file_entry_point(self, wfdb_file: Path):
        dat = wfdb_file.with_suffix(".dat")
        assert WFDBParser.can_parse(dat, dat.read_bytes()[:4096]) is True
        record = FileParser().parse(dat, auto_scale=True)
        assert [l.label for l in record.leads] == ["I", "II"]
