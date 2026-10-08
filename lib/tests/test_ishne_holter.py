"""Tests for the ISHNE Holter parser."""

from __future__ import annotations

import json
import struct
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.ishne_holter import ISHNEHolterParser
from tests.conftest import create_ishne_binary


class TestISHNEHolterParser:
    def test_parse_returns_ecg_record(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.source_format == "ishne_holter"

    def test_patient_name(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert "Test" in record.patient.first_name
        assert "Ecg" in record.patient.last_name

    def test_patient_sex(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1980

    def test_recording_sampling_rate(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.recording.acquisition.signal.sampling_rate == 200

    def test_recording_date(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023

    def test_lead_count(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        labels = [l.label for l in record.leads]
        assert "I" in labels
        assert "II" in labels

    def test_lead_sampling_rate(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        for lead in record.leads:
            assert lead.sampling_rate == 200

    def test_lead_samples_are_float(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64

    def test_lead_units_and_is_raw(self, ishne_file: Path):
        """Fixture has ampl_res=1000 nV → resolution=1.0 uV/count → already physical."""
        record = ISHNEHolterParser().parse(ishne_file)
        for lead in record.leads:
            assert lead.resolution_unit == "uV"
            assert lead.units == "uV"
            assert lead.resolution == 1.0
            assert lead.adc_resolution == 1000.0
            assert lead.adc_resolution_unit == "nV"
            assert lead.is_raw is False

    def test_recording_duration(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() > 0

    def test_corrupted_file_too_small(self, tmp_path: Path):
        f = tmp_path / "tiny.ecg"
        f.write_bytes(b"ISHNE1.0" + b"\x00" * 10)
        with pytest.raises(Exception):
            ISHNEHolterParser().parse(f)

    def test_to_dict_unified_schema(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] == "ishne_holter"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, ishne_file: Path):
        fp = FileParser()
        record = fp.parse(ishne_file)
        assert record.source_format == "ishne_holter"


# ---------------------------------------------------------------------------
# Header edge cases
# ---------------------------------------------------------------------------

def _crc(data: bytearray) -> None:
    """Recompute the checksum over byte 10 up to the ECG block."""
    from crccheck.crc import Crc16CcittFalse
    var_size = struct.unpack_from("<i", data, 10)[0]
    crc = Crc16CcittFalse.calc(bytes(data[10:522 + var_size]))
    struct.pack_into("<H", data, 8, crc)


def _build(tmp_path: Path, var_block: bytes = b"", crc: bool = True, **fields) -> Path:
    """Write an ISHNE file from the conftest builder with patched header fields.

    ``fields`` maps a file offset to a (struct format, value) pair.
    """
    data = bytearray(create_ishne_binary(nleads=fields.pop("nleads", 2),
                                         samples_per_lead=fields.pop("samples", 400)))
    if var_block:
        data[522:522] = var_block
        struct.pack_into("<i", data, 10, len(var_block))
        struct.pack_into("<i", data, 22, 522 + len(var_block))
    for offset, (fmt, value) in fields.pop("at", {}).items():
        if isinstance(value, bytes):
            data[offset:offset + len(value)] = value
        else:
            struct.pack_into(fmt, data, offset, *(value if isinstance(value, tuple) else (value,)))
    if crc:
        _crc(data)
    trim = fields.pop("trim", 0)
    if trim:
        data = data[:-trim]
    p = tmp_path / "edge.ecg"
    p.write_bytes(bytes(data))
    return p


def _parse(path: Path) -> tuple[ECGRecord, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        record = ISHNEHolterParser().parse(path)
    return record, list(w)


class TestISHNEHolterSignal:
    def test_three_lead_demultiplexing(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, nleads=3, samples=10))
        for i, lead in enumerate(record.leads):
            assert lead.samples[:3].tolist() == [(i + 1) * 100, (i + 1) * 100 + 1, (i + 1) * 100 + 2]

    def test_incomplete_frame_warns(self, tmp_path: Path):
        record, w = _parse(_build(tmp_path, trim=2))
        assert len(record.leads[0].samples) == 399
        assert any("incomplete final frame" in str(x.message) for x in w)

    def test_frame_boundary_truncation_reports_counts(self, tmp_path: Path):
        record, w = _parse(_build(tmp_path, trim=4))
        assert not w
        assert record.raw_metadata["ecg_size"] == 400
        assert record.raw_metadata["samples_per_lead"] == 399

    def test_warnings_point_at_caller(self, tmp_path: Path):
        _, w = _parse(_build(tmp_path, trim=2))
        assert w and all(x.filename == __file__ for x in w)

    def test_fault_samples_counted_not_changed(self, tmp_path: Path):
        # First sample of lead 2 sits at byte 524
        record, _ = _parse(_build(tmp_path, at={524: ("<h", -32768)}))
        assert record.raw_metadata["lead_fault_samples"] == [0, 1]
        assert record.leads[1].samples[0] == -32768


class TestISHNEHolterHeaderValidation:
    def test_negative_var_block_size(self, tmp_path: Path):
        p = _build(tmp_path, crc=False, at={10: ("<i", -5)})
        with pytest.raises(CorruptedFileError, match="variable block size is negative"):
            ISHNEHolterParser().parse(p)

    def test_var_block_past_eof(self, tmp_path: Path):
        p = _build(tmp_path, crc=False, at={10: ("<i", 10**7)})
        with pytest.raises(CorruptedFileError, match="past the end of the file"):
            ISHNEHolterParser().parse(p)

    @pytest.mark.parametrize("offset", [0, -5, 10**7])
    def test_bad_ecg_offset_falls_back(self, tmp_path: Path, offset: int):
        record, w = _parse(_build(tmp_path, at={22: ("<i", offset)}))
        assert any("ECG block offset" in str(x.message) for x in w)
        assert len(record.leads[0].samples) == 400
        assert record.leads[0].samples[0] == 100

    def test_bad_var_block_offset_warns(self, tmp_path: Path):
        _, w = _parse(_build(tmp_path, at={18: ("<i", 0)}))
        assert any("variable block offset" in str(x.message) for x in w)

    @pytest.mark.parametrize("sr", [0, -200])
    def test_invalid_sampling_rate(self, tmp_path: Path, sr: int):
        p = _build(tmp_path, at={272: ("<h", sr)})
        with pytest.raises(CorruptedFileError, match="sampling rate"):
            ISHNEHolterParser().parse(p)

    def test_invalid_lead_count(self, tmp_path: Path):
        p = _build(tmp_path, at={156: ("<h", 13)})
        with pytest.raises(CorruptedFileError, match="lead count"):
            ISHNEHolterParser().parse(p)

    def test_checksum_mismatch_warns(self, tmp_path: Path):
        p = _build(tmp_path, crc=False, at={28: ("", b"Changed")})
        _, w = _parse(p)
        assert any("checksum mismatch" in str(x.message) for x in w)

    def test_checksum_byte_swapped_accepted(self, tmp_path: Path):
        p = _build(tmp_path)
        data = bytearray(p.read_bytes())
        data[8], data[9] = data[9], data[8]
        p.write_bytes(bytes(data))
        _, w = _parse(p)
        assert not w

    def test_invalid_start_time_drops_date(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, at={150: ("<hhh", (25, 0, 0))}))
        assert record.recording.date is None


class TestISHNEHolterLeadLabels:
    def test_generic_codes_numbered(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, at={158: ("<hh", (1, 0))}))
        assert [l.label for l in record.leads] == ["Lead 1", "Lead 2"]

    def test_duplicate_codes_suffixed(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, nleads=3, at={158: ("<hhh", (6, 6, 6))}))
        assert [l.label for l in record.leads] == ["II", "II_2", "II_3"]

    def test_absent_code_on_stored_lead_warns(self, tmp_path: Path):
        record, w = _parse(_build(tmp_path, at={158: ("<h", -9)}))
        assert record.leads[0].label == "Lead 1"
        assert any("not present" in str(x.message) for x in w)

    def test_lead_quality_description(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, at={182: ("<hh", (1, 3))}))
        assert record.raw_metadata["lead_quality_desc"] == ["good", "frequent noise"]


class TestISHNEHolterHeaderFields:
    def test_file_format(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.file_format.version == "1"
        assert record.file_format.creation_date == date(2023, 12, 1)
        assert record.to_dict()["file_format"] == {"version": "1", "creation_date": "2023-12-01"}

    def test_race(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, at={130: ("<h", 3)}))
        assert record.patient.race == "Oriental"

    @pytest.mark.parametrize("code, has, kind", [
        (0, False, ""),
        (1, True, "unknown type"),
        (5, True, "dual chamber bipolar"),
        (-9, None, ""),
    ])
    def test_pacemaker(self, tmp_path: Path, code: int, has: bool | None, kind: str):
        record, _ = _parse(_build(tmp_path, at={230: ("<h", code)}))
        assert record.patient.has_pacemaker is has
        assert record.patient.pacemaker_type == kind

    def test_recorder_type_analog_digital(self, ishne_file: Path):
        record = ISHNEHolterParser().parse(ishne_file)
        assert record.recording.device.acquisition_type == "digital"
        assert record.recording.device.model == ""

    def test_recorder_type_model_name(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, at={232: ("", b"Lifecard CF")}))
        assert record.recording.device.model == "Lifecard CF"
        assert record.recording.device.acquisition_type == ""

    def test_latin1_names_stripped(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, at={68: ("", "Müller  ".encode("latin-1"))}))
        assert record.patient.last_name == "Müller"

    def test_variable_block_comments(self, tmp_path: Path):
        record, w = _parse(_build(tmp_path, var_block="Holter 24h, électrode V5\x00junk".encode("latin-1")))
        assert not w
        assert record.annotations["comments"] == "Holter 24h, électrode V5"
        assert len(record.leads[0].samples) == 400

    def test_proprietary_and_copyright(self, tmp_path: Path):
        record, _ = _parse(_build(tmp_path, at={274: ("", b"Hospital A"), 354: ("", b"(c) 2023")}))
        assert record.raw_metadata["proprietary"] == "Hospital A"
        assert record.raw_metadata["copyright"] == "(c) 2023"
