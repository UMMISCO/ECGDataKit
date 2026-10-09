"""Tests for the SCP-ECG parser (EN 1064 / ISO 11073-91064)."""

from __future__ import annotations

import json
import struct
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import ChecksumWarning, CorruptedFileError, UnsupportedFormatError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.scp_ecg import SCPECGParser
from tests.conftest import (
    create_scp_ecg_binary,
    scp_crc,
    scp_huffman_encode,
    scp_signal_data,
    scp_tag,
    scp_test_signal,
)


def _write(tmp_path: Path, data: bytes, name: str = "rec.scp") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _parse(tmp_path: Path, data: bytes) -> ECGRecord:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no unexpected warning
        return SCPECGParser().parse(_write(tmp_path, data))


def _fix_crcs(data: bytearray) -> bytes:
    """Recompute every section CRC and the record CRC after a mutation."""
    sec0_len = struct.unpack_from("<I", data, 10)[0]
    for pos in range(22, 6 + sec0_len, 10):
        _, length, index = struct.unpack_from("<HII", data, pos)
        if length and index and index - 1 + length <= len(data):
            start = index - 1
            struct.pack_into("<H", data, start, scp_crc(bytes(data[start + 2: start + length])))
    struct.pack_into("<H", data, 6, scp_crc(bytes(data[8: 6 + sec0_len])))
    struct.pack_into("<H", data, 0, scp_crc(bytes(data[2:])))
    return bytes(data)


def _section_offset(data: bytes, sid: int) -> int:
    sec0_len = struct.unpack_from("<I", data, 10)[0]
    for pos in range(22, 6 + sec0_len, 10):
        s, length, index = struct.unpack_from("<HII", data, pos)
        if s == sid and length:
            return index - 1
    raise KeyError(sid)


def _section1(*tags: bytes) -> bytes:
    return b"".join(tags) + scp_tag(255, b"")


# ---------------------------------------------------------------------------
# Basic record
# ---------------------------------------------------------------------------


class TestSCPECGParser:
    def test_parse_returns_ecg_record(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert isinstance(record, ECGRecord)
        assert record.source_format == "scp_ecg"

    def test_patient(self, scp_ecg_file: Path):
        p = SCPECGParser().parse(scp_ecg_file).patient
        assert p.last_name == "TestSCP"
        assert p.first_name == "José"
        assert p.patient_id == "SCP001"
        assert p.birth_date == datetime(1980, 6, 15)
        assert p.age == 43  # from DOB and acquisition date 2024-03-09
        assert p.sex == "M"
        assert p.race == "Caucasian"
        assert p.height == 180.0
        assert p.weight == 75.0
        assert p.has_pacemaker is None

    def test_recording_and_filters(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert record.recording.date == datetime(2024, 3, 9, 14, 30, 5)
        assert record.recording.duration == timedelta(seconds=1)
        f = record.recording.acquisition.filters
        assert f.highpass == pytest.approx(0.05)
        assert f.lowpass == 150.0
        assert f.notch == 50.0
        assert f.notch_active is True
        assert f.artifact_filter is False

    def test_file_format_and_checksum(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert record.file_format.version == "2.0"
        assert record.raw_metadata["checksum_valid"] is True
        assert record.raw_metadata["sections_found"] == [0, 1, 3, 6]

    def test_leads(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert [lead.label for lead in record.leads] == ["I", "II"]
        for i, lead in enumerate(record.leads):
            assert lead.samples.dtype == np.float64
            np.testing.assert_array_equal(lead.samples, scp_test_signal(i, 500))
            assert lead.sampling_rate == 500
            assert lead.adc_resolution == 1000.0
            assert lead.adc_resolution_unit == "nV"
            assert lead.resolution == 1.0
            assert lead.resolution_unit == "uV"
            assert lead.units == "uV"
            assert lead.is_raw is False

    def test_signal_characteristics(self, scp_ecg_file: Path):
        sig = SCPECGParser().parse(scp_ecg_file).recording.acquisition.signal
        assert sig.sampling_rate == 500
        assert sig.resolution == 1.0
        assert sig.compression == "none"
        assert sig.data_encoding == "amplitudes"
        assert sig.number_channels_allocated == 2
        assert sig.number_channels_valid == 2

    def test_to_dict_and_json(self, scp_ecg_file: Path):
        record = SCPECGParser().parse(scp_ecg_file)
        assert set(record.to_dict()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats", "leads_enhanced",
            "annotations",
        }
        parsed = json.loads(record.to_json())
        assert parsed["file_format"]["version"] == "2.0"
        assert len(parsed["leads"]) == 2

    def test_auto_scale(self, tmp_path: Path):
        path = _write(tmp_path, create_scp_ecg_binary(avm=5000))
        scaled = FileParser().parse(path)
        assert scaled.source_format == "scp_ecg"
        assert scaled.leads[0].units == "mV"
        np.testing.assert_allclose(scaled.leads[0].samples, scp_test_signal(0, 500) * 0.005)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = FileParser().parse(path, auto_scale=False)
        assert raw.leads[0].is_raw is True
        assert raw.leads[0].units == ""
        assert raw.leads[0].resolution == 5.0
        np.testing.assert_array_equal(raw.leads[0].samples, scp_test_signal(0, 500))

    def test_can_parse_requires_section0_layout(self, tmp_path: Path):
        data = create_scp_ecg_binary()
        assert SCPECGParser.can_parse(Path("x.scp"), data[:4096])
        bad = bytearray(data)
        struct.pack_into("<H", bad, 8, 1)  # Section 0 ID
        assert not SCPECGParser.can_parse(Path("x.scp"), bytes(bad[:4096]))


# ---------------------------------------------------------------------------
# Section 1
# ---------------------------------------------------------------------------


class TestSection1:
    def test_age_without_dob_and_units(self, tmp_path: Path):
        sec1 = _section1(
            scp_tag(4, struct.pack("<HB", 18, 2)),  # 18 months
            scp_tag(6, struct.pack("<HB", 70, 2)),  # inches
            scp_tag(7, struct.pack("<HB", 22000, 2)),  # grams
            scp_tag(8, bytes([2])),
            scp_tag(9, bytes([0])),
        )
        p = _parse(tmp_path, create_scp_ecg_binary(section1=sec1)).patient
        assert p.age == 1
        assert p.height == pytest.approx(177.8)
        assert p.weight == pytest.approx(22.0)
        assert p.sex == "F"
        assert p.race == ""
        assert p.birth_date is None

    def test_default_age_zero_is_ignored(self, tmp_path: Path):
        sec1 = _section1(scp_tag(4, struct.pack("<HB", 0, 1)), scp_tag(8, bytes([0])))
        p = _parse(tmp_path, create_scp_ecg_binary(section1=sec1)).patient
        assert p.age is None
        assert p.sex == "U"

    def test_texts_devices_and_codes(self, tmp_path: Path):
        machine = (
            struct.pack("<HHHBB", 7, 3, 51, 0, 255) + b"MOD123" + bytes([20, 0xA0, 0x01, 0xD0, 1])
            + bytes(16) + bytes([4]) + b"A1\x00SN-9\x00SW 2.1\x00SCPLIB\x00Acme \xc4rzte\x00"
        )
        sec1 = _section1(
            scp_tag(0, "Müller".encode("latin-1") + b"\x00"),
            scp_tag(10, bytes([0, 1, 2]) + b"Aspirin\x00"),
            scp_tag(10, bytes([0, 4, 7])),
            scp_tag(11, struct.pack("<H", 120)),
            scp_tag(12, struct.pack("<H", 80)),
            scp_tag(13, b"Chest pain\x00"),
            scp_tag(14, machine),
            scp_tag(16, b"General Hospital\x00"),
            scp_tag(18, b"Cardiology\x00"),
            scp_tag(20, b"Dr Who\x00"),
            scp_tag(22, b"Tech A\x00"),
            scp_tag(23, b"Room 4\x00"),
            scp_tag(25, struct.pack("<HBB", 2024, 3, 9)),
            scp_tag(26, bytes([8, 0, 0])),
            scp_tag(30, b"free comment\x00"),
            scp_tag(32, bytes([0, 1, 42])),
            scp_tag(33, bytes([2, 0])),
            scp_tag(34, struct.pack("<hH", 60, 0) + b"CET\x00"),
            scp_tag(35, b"Old MI\x00"),
            scp_tag(200, b"vendor\x00"),
        )
        record = _parse(tmp_path, create_scp_ecg_binary(section1=sec1))
        p = record.patient
        assert p.last_name == "Müller"
        assert p.medications == ["Aspirin", "table 0 class 4 code 7"]
        assert p.clinical_history == "Chest pain\nOld MI"
        assert p.has_pacemaker is True  # medical history code 42
        dev = record.recording.device
        assert dev.manufacturer == "Acme Ärzte"
        assert dev.model == "MOD123"
        assert dev.serial_number == "SN-9"
        assert dev.software_version == "SW 2.1"
        assert dev.institution == "General Hospital"
        assert dev.department == "Cardiology"
        rec = record.recording
        assert rec.referring_physician == "Dr Who"
        assert rec.technician == "Tech A"
        assert rec.room == "Room 4"
        assert rec.date == datetime(2024, 3, 9, 8, 0, 0, tzinfo=timezone(timedelta(hours=1)))
        assert rec.acquisition.signal.acsetting == 50
        assert rec.acquisition.signal.electrode_placement == "Mason-Likar, individual electrodes"
        assert record.annotations["comments"] == "free comment"
        raw = record.raw_metadata
        assert raw["systolic_bp_mmhg"] == 120 and raw["diastolic_bp_mmhg"] == 80
        assert raw["acquiring_device"]["device_id"] == 51
        assert raw["manufacturer_tags"] == {200: "vendor"}
        assert record.file_format.version == "2.0"

    def test_date_without_time_is_not_invented(self, tmp_path: Path):
        sec1 = _section1(scp_tag(25, struct.pack("<HBB", 2024, 3, 9)))
        record = _parse(tmp_path, create_scp_ecg_binary(section1=sec1))
        assert record.recording.date is None
        assert record.raw_metadata["acquisition_date"] == "2024-03-09"

    def test_wrong_fixed_tag_length_is_recovered(self, tmp_path: Path):
        # Tag 4 written with length 6 instead of 3, as some carts do
        sec1 = (
            bytes([4]) + struct.pack("<H", 6) + struct.pack("<HB", 0, 1)
            + scp_tag(5, bytes(4)) + scp_tag(8, bytes([2]))
            + scp_tag(25, struct.pack("<HBB", 2011, 4, 27)) + scp_tag(255, b"")
        )
        path = _write(tmp_path, create_scp_ecg_binary(section1=sec1))
        with pytest.warns(UserWarning, match="wrong tag length"):
            record = SCPECGParser().parse(path)
        assert record.patient.sex == "F"
        assert record.raw_metadata["acquisition_date"] == "2011-04-27"


# ---------------------------------------------------------------------------
# Leads and encodings
# ---------------------------------------------------------------------------


class TestLeadsAndEncoding:
    def test_lead_code_table(self, tmp_path: Path):
        codes = [1, 2, 61, 62, 63, 64, 3, 4, 5, 6, 7, 8, 150, 1]
        record = _parse(tmp_path, create_scp_ecg_binary(lead_codes=codes, samples_per_lead=20))
        assert [lead.label for lead in record.leads] == [
            "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
            "Lead150", "I_2",
        ]

    @pytest.mark.parametrize("huffman", [False, True])
    @pytest.mark.parametrize("diff", [0, 1, 2])
    def test_encodings_round_trip(self, tmp_path: Path, diff: int, huffman: bool):
        rng = np.random.default_rng(diff)
        signals = [np.cumsum(rng.integers(-300, 300, 400)) for _ in range(3)]
        signals[0][10] = 20000  # forces 16-bit escapes in the differences
        data = create_scp_ecg_binary(
            lead_codes=[1, 2, 3], signals=signals, diff=diff, huffman=huffman,
        )
        record = _parse(tmp_path, data)
        for lead, sig in zip(record.leads, signals):
            np.testing.assert_array_equal(lead.samples, sig)
        sig_info = record.recording.acquisition.signal
        assert sig_info.compression == ("huffman" if huffman else "none")
        assert sig_info.data_encoding == ("amplitudes", "first_difference", "second_difference")[diff]

    def test_default_huffman_codes(self, tmp_path: Path):
        values = [0, 1, -1, 5, -8, 8, 9, -9, 100, -128, 127, 1000, -1000, -32768, 3]
        data = create_scp_ecg_binary(lead_codes=[1], signals=[np.array(values)], huffman=True)
        record = _parse(tmp_path, data)
        np.testing.assert_array_equal(record.leads[0].samples, values)

    def test_custom_huffman_tables_with_mode_switch(self, tmp_path: Path):
        def code(prefix: int, total: int, mode: int, value: int, bits: str) -> bytes:
            return struct.pack("<BBBhI", prefix, total, mode, value, int(bits[::-1], 2))

        table1 = [
            code(1, 1, 1, 0, "0"),
            code(2, 2, 1, 1, "10"),
            code(3, 19, 1, 0, "110"),  # followed by 16 original bits
            code(3, 3, 0, 2, "111"),  # switch to table 2
        ]
        table2 = [code(1, 1, 1, 5, "0"), code(1, 1, 0, 1, "1")]  # 5, back to table 1
        sec2 = struct.pack("<H", 2)
        for table in (table1, table2):
            sec2 += struct.pack("<H", len(table)) + b"".join(table)
        bits = "0" + "10" + "110" + format(1000, "016b") + "111" + "0" + "0" + "1" + "10"
        bits += "0" * (-len(bits) % 8)
        stream = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
        expected = [0, 1, 1000, 5, 5, 1]
        sec6 = struct.pack("<HHBB", 1000, 2000, 0, 0) + struct.pack("<H", len(stream)) + stream
        data = create_scp_ecg_binary(
            lead_codes=[1], signals=[np.array(expected)], sections={2: sec2, 6: sec6},
        )
        record = _parse(tmp_path, data)
        np.testing.assert_array_equal(record.leads[0].samples, expected)
        assert record.raw_metadata["huffman_tables"] == 2

    def test_lead_start_offset_is_kept(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary(samples_per_lead=10))
        sec3 = _section_offset(bytes(data), 3) + 16
        struct.pack_into("<II", data, sec3 + 2, 11, 20)  # lead I covers samples 11-20
        record = _parse(tmp_path, _fix_crcs(data))
        assert record.leads[0].annotations["start_sample"] == "11"
        assert len(record.leads[0].samples) == 10


# ---------------------------------------------------------------------------
# Reference beats (Sections 4 and 5)
# ---------------------------------------------------------------------------


def _reference_record(subtract: bool, median_avm: int = 2000):
    """Two leads at 500 Hz with three beats and a 40-sample reference beat."""
    n = 300
    template = (np.hanning(40) * 300).astype(np.int64)  # median in median_avm units
    templates = [template, template + np.arange(40)]
    fiducials = [50, 150, 250]  # 1-based sample numbers
    ref_fiducial = 20  # 1-based position of the fiducial in the template
    scale = median_avm // 1000
    originals = []
    residuals = []
    for lead in range(2):
        base = scp_test_signal(lead, n)
        full = base.copy()
        for fc in fiducials:
            start = fc - ref_fiducial  # 0-based index of template sample 0
            full[start: start + 40] += templates[lead] * scale
        originals.append(full)
        residuals.append(base if subtract else full)
    zones = b"".join(
        struct.pack("<HIII", 0, fc - ref_fiducial + 1, fc, fc - ref_fiducial + 40) for fc in fiducials
    )
    sec4 = struct.pack("<HHH", 80, ref_fiducial, len(fiducials)) + zones
    sec5 = scp_signal_data(median_avm, 2000, 1, templates, huffman=False)
    data = create_scp_ecg_binary(
        lead_codes=[1, 2], signals=residuals, avm=1000,
        lead_flags=0x05 if subtract else 0x04, sections={4: sec4, 5: sec5},
    )
    return data, originals, template


class TestReferenceBeats:
    def test_median_beats(self, tmp_path: Path):
        data, originals, template = _reference_record(subtract=False)
        record = _parse(tmp_path, data)
        assert [b.label for b in record.median_beats] == ["I", "II"]
        np.testing.assert_array_equal(record.median_beats[0].samples, template)
        np.testing.assert_array_equal(record.median_beats[1].samples, template + np.arange(40))
        assert record.median_beats[0].resolution == 2.0
        assert record.median_beats[0].sampling_rate == 500
        np.testing.assert_array_equal(record.leads[0].samples, originals[0])
        assert record.measurements.qrs_count == 3
        assert record.raw_metadata["qrs_locations"][0] == {
            "type": 0, "start": 31, "fiducial": 50, "end": 70,
        }

    def test_reference_beat_is_added_back(self, tmp_path: Path):
        data, originals, _ = _reference_record(subtract=True)
        record = _parse(tmp_path, data)
        for lead, original in zip(record.leads, originals):
            # common unit gcd(1000, 2000) = 1000 nV, so counts match the originals
            np.testing.assert_array_equal(lead.samples, original)
            assert lead.resolution == 1.0
        assert record.recording.acquisition.signal.compression == "reference beat subtraction"

    def test_subtraction_without_section5_is_corrupt(self, tmp_path: Path):
        data = create_scp_ecg_binary(lead_flags=0x05)
        with pytest.raises(CorruptedFileError, match="reference beat"):
            SCPECGParser().parse(_write(tmp_path, data))

    def test_bimodal_decimation_is_unsupported(self, tmp_path: Path):
        sec6 = scp_signal_data(1000, 4000, 0, [scp_test_signal(0, 10)], False, flag=1)
        data = create_scp_ecg_binary(lead_codes=[1], samples_per_lead=10, sections={6: sec6})
        with pytest.raises(UnsupportedFormatError, match="bimodal"):
            SCPECGParser().parse(_write(tmp_path, data))

    def test_bimodal_byte_is_reserved_in_version_3(self, tmp_path: Path):
        sec6 = scp_signal_data(1000, 2000, 0, [scp_test_signal(0, 10)], False, flag=1)
        data = create_scp_ecg_binary(
            lead_codes=[1], samples_per_lead=10, sections={6: sec6}, protocol=30,
        )
        record = _parse(tmp_path, data)
        assert record.file_format.version == "3.0"
        np.testing.assert_array_equal(record.leads[0].samples, scp_test_signal(0, 10))


# ---------------------------------------------------------------------------
# Measurements and statements
# ---------------------------------------------------------------------------


def _section7(spikes: bool = True) -> bytes:
    body = struct.pack("<BBHH", 1, 2 if spikes else 0, 857, 29999)
    body += struct.pack("<5H3h", 100, 200, 260, 350, 650, 45, 999, -999)
    if spikes:
        body += struct.pack("<Hh", 1200, 900) + struct.pack("<Hh", 2050, 850)
        body += bytes([3, 1]) + struct.pack("<HH", 0, 500)
        body += bytes([2, 1]) + struct.pack("<HH", 3, 500)
    body += struct.pack("<H", 2) + bytes([0, 0])  # QRS types
    body += struct.pack("<3HB", 72, 70, 410, 1) + struct.pack("<H", 0)
    return body


def _statements(flag: int, lines: list[bytes], typed: bool = False) -> bytes:
    body = bytes([flag]) + struct.pack("<HBB", 2024, 3, 10) + bytes([9, 15, 0]) + bytes([len(lines)])
    for i, line in enumerate(lines, 1):
        text = (b"\x02" + line) if typed else line
        body += bytes([i]) + struct.pack("<H", len(text)) + text
    return body


class TestMeasurementsAndStatements:
    def test_section7(self, tmp_path: Path):
        record = _parse(tmp_path, create_scp_ecg_binary(sections={7: _section7()}))
        m = record.measurements
        assert m.rr_interval == 857
        assert m.heart_rate == 72  # stored ventricular rate
        # Section 7 stores wave boundaries: intervals are not computed from them
        assert (m.pr_interval, m.qrs_duration, m.qt_interval) == (None, None, None)
        names = ("p_onset", "p_offset", "qrs_onset", "qrs_offset", "t_offset")
        assert {n: record.annotations[f"median_{n}_ms"] for n in names} == {
            "p_onset": "100", "p_offset": "200", "qrs_onset": "260",
            "qrs_offset": "350", "t_offset": "650"}
        assert m.qtc_bazett == 410  # formula code 1 = Bazett
        assert m.p_axis == 45
        assert m.qrs_axis is None and m.t_axis is None
        assert m.qrs_count == 2
        assert record.raw_metadata["atrial_rate_bpm"] == 70
        assert "pp_interval_ms" not in record.raw_metadata
        # Detected spikes are reported, not turned into a patient attribute
        assert record.patient.has_pacemaker is None
        spike = record.raw_metadata["pacemaker_spikes"][1]
        assert spike == {
            "time_ms": 2050, "amplitude_uv": 850, "type": "ventricular",
            "source": "internal", "qrs_index": 3, "pulse_width_us": 500,
        }

    def test_heart_rate_not_computed_from_rr(self, tmp_path: Path):
        body = struct.pack("<BBHH", 1, 0, 1000, 1000) + struct.pack("<5H3h", *([29999] * 5), 999, 60, 999)
        record = _parse(tmp_path, create_scp_ecg_binary(sections={7: body}))
        assert record.measurements.rr_interval == 1000
        assert record.measurements.heart_rate is None
        assert record.measurements.pr_interval is None
        assert record.measurements.qrs_axis == 60
        assert record.patient.has_pacemaker is None

    def test_machine_statements(self, tmp_path: Path):
        data = create_scp_ecg_binary(sections={8: _statements(0, [b"Sinus rhythm\x00", b"Normal ECG\x00"])})
        interp = _parse(tmp_path, data).interpretation
        assert interp.source == "machine"
        assert interp.statements == [("Sinus rhythm", ""), ("Normal ECG", "")]
        assert interp.interpreter == ""

    def test_confirmed_statements_and_machine_annotation(self, tmp_path: Path):
        sec1 = _section1(scp_tag(21, b"Dr Heart\x00"))
        data = create_scp_ecg_binary(section1=sec1, sections={
            8: _statements(1, [b"Sinus bradycardia\x00"]),
            11: _statements(0, [b"SR\x00Sinus rhythm\x00"], typed=True),
        })
        record = _parse(tmp_path, data)
        interp = record.interpretation
        assert interp.source == "confirmed"
        assert interp.statements == [("Sinus bradycardia", "")]
        assert interp.interpreter == "Dr Heart"
        assert interp.interpretation_date == datetime(2024, 3, 10, 9, 15, 0)
        assert record.annotations["machine_interpretation"] == "SR Sinus rhythm"

    def test_overread_source(self, tmp_path: Path):
        data = create_scp_ecg_binary(sections={8: _statements(2, [b"Overread\x00"])})
        assert _parse(tmp_path, data).interpretation.source == "overread"

    def test_section10_lead_measurements(self, tmp_path: Path):
        values = [34, 266, 88, 296] + [29999] * 5 + [-120, 900]
        block = struct.pack("<HH", 1, 2 * len(values)) + struct.pack(f"<{len(values)}h", *values)
        body = struct.pack("<HH", 1, 0) + block
        record = _parse(tmp_path, create_scp_ecg_binary(sections={10: body}))
        lead_i = record.raw_metadata["lead_measurements"]["I"]
        assert lead_i["pr_interval"] == 266
        assert lead_i["q_amplitude"] == -120
        assert "q_duration" not in lead_i


# ---------------------------------------------------------------------------
# CRC and corrupt input
# ---------------------------------------------------------------------------


class TestIntegrity:
    def test_crc_mismatch_warns(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary())
        data[-1] ^= 0x01  # last sample of Section 6
        path = _write(tmp_path, bytes(data))
        with pytest.warns(ChecksumWarning, match="Section 6") as caught:
            record = SCPECGParser().parse(path)
        assert caught[0].filename == __file__
        assert record.raw_metadata["checksum_valid"] is False

    def test_truncated_file(self, tmp_path: Path):
        data = create_scp_ecg_binary()
        with pytest.raises(CorruptedFileError, match="declares"):
            SCPECGParser().parse(_write(tmp_path, data[: len(data) // 2]))

    def test_tiny_file(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError):
            SCPECGParser().parse(_write(tmp_path, create_scp_ecg_binary()[:20]))

    def test_pointer_outside_record(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary())
        struct.pack_into("<I", data, 22 + 10 + 6, 10 ** 6)  # Section 1 index
        with pytest.raises(CorruptedFileError, match="outside"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_section_id_mismatch(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary())
        struct.pack_into("<H", data, _section_offset(bytes(data), 3) + 2, 4)
        with pytest.raises(CorruptedFileError, match="header says 4"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_zero_sample_interval(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary())
        struct.pack_into("<H", data, _section_offset(bytes(data), 6) + 18, 0)
        with pytest.raises(CorruptedFileError, match="interval is 0"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_invalid_lead_range(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary())
        struct.pack_into("<I", data, _section_offset(bytes(data), 3) + 18 + 4, 0)
        with pytest.raises(CorruptedFileError, match="sample range"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_lead_count_exceeds_definitions(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary())
        data[_section_offset(bytes(data), 3) + 16] = 200
        with pytest.raises(CorruptedFileError, match="declares 200 leads"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_unknown_difference_code(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary())
        data[_section_offset(bytes(data), 6) + 20] = 7
        with pytest.raises(CorruptedFileError, match="difference encoding 7"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_short_lead_data(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary(samples_per_lead=10))
        struct.pack_into("<I", data, _section_offset(bytes(data), 3) + 18 + 4, 11)
        with pytest.raises(CorruptedFileError, match="expected 11"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_short_huffman_stream(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary(samples_per_lead=50, huffman=True))
        struct.pack_into("<I", data, _section_offset(bytes(data), 3) + 18 + 4, 400)
        with pytest.raises(CorruptedFileError, match="Huffman data ends"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_missing_rhythm_section(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary())
        struct.pack_into("<I", data, 22 + 60 + 2, 0)  # Section 6 pointer length
        with pytest.raises(CorruptedFileError, match="Section 6"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_huffman_encoder_matches_default_table(self):
        assert scp_huffman_encode([0, 1, -1]) == bytes([0b01001010])


def test_sample_interval_kept(tmp_path: Path):
    record = _parse(tmp_path, create_scp_ecg_binary())
    interval = record.raw_metadata["rhythm_sample_interval_us"]
    assert record.leads[0].sampling_rate == round(1_000_000 / interval)


# ---------------------------------------------------------------------------
# Language support code (Section 1 tag 14 byte 17, EN 1064:2005+A1 table as
# implemented by BioSig decode_scp_text), SCP-ECG v3 and edge cases
# ---------------------------------------------------------------------------


def _tag14(language: int) -> bytes:
    """Acquiring device ID: 36 fixed bytes then five NUL-terminated strings."""
    value = bytearray(36)
    value[14] = 20  # protocol revision 2.0
    value[16] = language
    return scp_tag(14, bytes(value) + b"\x00" * 5)


class TestTextEncodings:
    @pytest.mark.parametrize("language,raw,expected", [
        (0x37, "Müller-Łukasz".encode("utf-8"), "Müller-Łukasz"),
        (0x03, "Łódź".encode("iso8859_2"), "Łódź"),
        (0x07, "Ab".encode("utf-16-le") + b"\x00\x00", "Ab"),
        (0x07, "Łó".encode("utf-16"), "Łó"),  # with a byte order mark
        (0x0F, "ｱｲｳ".encode("shift_jis"), "ｱｲｳ"),  # JIS X0201 half-width katakana
    ])
    def test_language_codes(self, tmp_path: Path, language: int, raw: bytes, expected: str):
        sec1 = _section1(_tag14(language), scp_tag(0, raw + b"\x00\x00"))
        assert _parse(tmp_path, create_scp_ecg_binary(section1=sec1)).patient.last_name == expected

    def test_wide_statements(self, tmp_path: Path):
        text = "SR".encode("utf-16-le") + b"\x00\x00" + "Normal".encode("utf-16-le") + b"\x00\x00"
        sec8 = bytes([0]) + struct.pack("<HBB", 2024, 3, 10) + bytes([9, 15, 0, 1])
        sec8 += bytes([1]) + struct.pack("<H", len(text)) + text
        data = create_scp_ecg_binary(section1=_section1(_tag14(0x07)), sections={8: sec8})
        assert _parse(tmp_path, data).interpretation.statements == [("SR Normal", "")]


class TestEdgeCases:
    def test_time_without_date_is_kept_raw(self, tmp_path: Path):
        sec1 = _section1(scp_tag(26, bytes([10, 20, 30])))
        record = _parse(tmp_path, create_scp_ecg_binary(section1=sec1))
        assert record.recording.date is None
        assert record.raw_metadata["acquisition_time"] == "10:20:30"

    def test_long_term_section12_is_unsupported(self, tmp_path: Path):
        data = bytearray(create_scp_ecg_binary(protocol=30, sections={12: bytes(80)}))
        sec0_len = struct.unpack_from("<I", data, 10)[0]
        for pos in range(22, 6 + sec0_len, 10):
            if struct.unpack_from("<H", data, pos)[0] == 6:
                struct.pack_into("<II", data, pos + 2, 0, 0)  # no Section 6
        with pytest.raises(UnsupportedFormatError, match="Section 12"):
            SCPECGParser().parse(_write(tmp_path, _fix_crcs(data)))

    def test_section12_next_to_section6_is_not_a_vendor_section(self, tmp_path: Path):
        data = create_scp_ecg_binary(protocol=30, sections={12: bytes(80)})
        with pytest.warns(UserWarning, match="Section 12"):
            record = SCPECGParser().parse(_write(tmp_path, data))
        assert record.raw_metadata["section12_length"] == 96
        assert "vendor_sections" not in record.raw_metadata

    def test_subtraction_keeps_the_stored_multiplier(self, tmp_path: Path):
        data, _, _ = _reference_record(subtract=True, median_avm=1500)
        lead = _parse(tmp_path, data).leads[0]
        assert lead.resolution == 0.5  # gcd(1000, 1500) nV
        assert lead.adc_resolution == 1000.0 and lead.adc_resolution_unit == "nV"

    def test_reference_beat_shorter_than_a_sample(self, tmp_path: Path):
        sec4 = struct.pack("<HHH", 1, 1, 0)  # 1 ms at 2000 us per sample
        sec5 = scp_signal_data(1000, 2000, 0, [[1], [2]], huffman=False)
        data = create_scp_ecg_binary(sections={4: sec4, 5: sec5})
        path = _write(tmp_path, data)
        with pytest.warns(UserWarning, match="shorter than one sample"):
            record = SCPECGParser().parse(path)
        assert record.median_beats == []
        assert len(record.leads[0].samples) == 500
