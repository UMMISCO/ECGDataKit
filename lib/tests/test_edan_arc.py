"""Tests for the EDAN ARC / SE-2012 Holter parser."""

from __future__ import annotations

import json
import struct
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.edan_arc import EDANARCHolterParser
from tests.conftest import create_edan_arc_dat, create_edan_arc_hea


class TestEDANARCDocumentedLayout:
    """patient.hea + ecgraw.dat — the deterministic path."""

    def test_parse_returns_ecg_record(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert isinstance(record, ECGRecord)

    def test_source_format(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.source_format == "edan_arc"

    def test_patient_id(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.patient.patient_id == "EDAN0001"

    def test_patient_height_weight(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.patient.height == 175.0
        assert record.patient.weight == 70.0

    def test_patient_medication(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.patient.medications == ["Aspirin"]

    def test_recording_sampling_rate(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.recording.acquisition.signal.sampling_rate == 200

    def test_recording_date(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023
        assert record.recording.date.month == 12
        assert record.recording.date.day == 1

    def test_recording_duration_from_timestamps(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() == pytest.approx(5.0)
        assert record.recording.end_date - record.recording.date == record.recording.duration

    def test_lead_count(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert len(record.leads) == 3

    def test_lead_labels(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        labels = [lead.label for lead in record.leads]
        assert labels == ["I", "II", "III"]

    def test_lead_samples_are_float(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) == 1000

    def test_lead_signal_centred_around_zero(self, edan_arc_dir: Path):
        """ADC zero = 16384; decoded samples should be small signed values."""
        record = EDANARCHolterParser().parse(edan_arc_dir)
        for ch, lead in enumerate(record.leads):
            expected_min = (ch + 1) * 10
            expected_max = expected_min + 49
            assert lead.samples.min() == pytest.approx(expected_min)
            assert lead.samples.max() == pytest.approx(expected_max)

    def test_signal_offset_metadata(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        sig = record.recording.acquisition.signal
        assert sig.signal_offset == 16384
        assert sig.bits_per_sample == 16
        assert sig.data_encoding == "uint16"
        assert sig.number_channels_allocated == 3
        assert sig.number_channels_valid == 3

    def test_device_manufacturer(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.recording.device.manufacturer == "EDAN"
        # Offset 2304 is documented as the recorder ID
        assert record.recording.device.serial_number == "SE2012"
        assert record.recording.device.software_version == "1.0a"
        assert record.recording.device.department == "Cardiology"

    def test_lowpass_filter(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.recording.acquisition.filters.lowpass == 75.0

    def test_can_parse_detects_patient_hea(self, edan_arc_dir: Path):
        header = edan_arc_dir.read_bytes()[:4096]
        assert EDANARCHolterParser.can_parse(edan_arc_dir, header) is True

    def test_can_parse_rejects_orphan_hea(self, tmp_path: Path):
        """patient.hea without ecgraw.dat sibling must NOT be claimed."""
        p = tmp_path / "patient.hea"
        p.write_bytes(b"\x00" * 4096)
        header = p.read_bytes()[:4096]
        assert EDANARCHolterParser.can_parse(p, header) is False

    def test_to_dict_unified_schema(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        parsed = json.loads(record.to_json())
        assert parsed["source_format"] == "edan_arc"
        assert len(parsed["leads"]) == 3

    def test_auto_detection_via_file_parser(self, edan_arc_dir: Path):
        record = FileParser().parse(edan_arc_dir)
        assert record.source_format == "edan_arc"

    def test_corrupted_header_too_small(self, tmp_path: Path):
        """A 64-byte truncated header should be rejected."""
        hea = tmp_path / "patient.hea"
        dat = tmp_path / "ecgraw.dat"
        hea.write_bytes(b"\x00" * 32)
        dat.write_bytes(b"\x00" * 100)
        with pytest.raises(CorruptedFileError):
            EDANARCHolterParser().parse(hea)


class TestEDANARCArchive:
    """`.arc` archive — best-effort heuristic path."""

    def test_parse_returns_ecg_record(self, edan_arc_archive: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = EDANARCHolterParser().parse(edan_arc_archive)
        assert isinstance(record, ECGRecord)

    def test_source_format_marks_archive(self, edan_arc_archive: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = EDANARCHolterParser().parse(edan_arc_archive)
        assert record.source_format == "edan_arc_archive"
        assert record.raw_metadata.get("arc_heuristic") is True

    def test_parse_emits_warning(self, edan_arc_archive: Path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EDANARCHolterParser().parse(edan_arc_archive)
        assert any(
            issubclass(w.category, UserWarning)
            and "experimental" in str(w.message)
            for w in caught
        )

    def test_lead_count_matches_header(self, edan_arc_archive: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = EDANARCHolterParser().parse(edan_arc_archive)
        assert len(record.leads) == 3

    def test_signal_data_matches_documented_layout(
        self, edan_arc_archive: Path, edan_arc_dir: Path,
    ):
        """Heuristic .arc parse should recover the same channel layout and
        ECG amplitude range as the documented parse.  Exact byte-level
        equality is not asserted: the wrapper format is unknown, so the
        probe may land a few bytes off the true payload start."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            arc_rec = EDANARCHolterParser().parse(edan_arc_archive)
        doc_rec = EDANARCHolterParser().parse(edan_arc_dir)
        assert len(arc_rec.leads) == len(doc_rec.leads)
        for arc_lead, doc_lead in zip(arc_rec.leads, doc_rec.leads):
            assert arc_lead.label == doc_lead.label
            assert arc_lead.sampling_rate == doc_lead.sampling_rate
            # Both should be small signed values centred around 0 after
            # ADC-zero subtraction (synthetic fixture stays under 100).
            assert abs(arc_lead.samples.mean()) < 200
            assert arc_lead.samples.std() < 200

    def test_can_parse_detects_arc_extension(self, edan_arc_archive: Path):
        header = edan_arc_archive.read_bytes()[:4096]
        assert EDANARCHolterParser.can_parse(edan_arc_archive, header) is True

    def test_corrupted_arc_without_header(self, tmp_path: Path):
        """A .arc file with no recognizable patient.hea must raise."""
        p = tmp_path / "junk.arc"
        p.write_bytes(b"\x00" * 8192)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(CorruptedFileError):
                EDANARCHolterParser().parse(p)


class TestNeutralHolterArc:
    """NEUTRAL HOLTER RECORDING `.arc` — reverse-engineered decoder."""

    def test_parses_reverse_engineered(self, neutral_holter_arc_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = EDANARCHolterParser().parse(neutral_holter_arc_file)
        assert isinstance(record, ECGRecord)
        assert record.source_format == "neutral_holter_arc"

    def test_emits_reverse_engineered_warning(self, neutral_holter_arc_file: Path):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EDANARCHolterParser().parse(neutral_holter_arc_file)
        assert any(
            issubclass(w.category, UserWarning)
            and "reverse-engineered" in str(w.message)
            for w in caught
        )

    def test_decodes_three_channels(self, neutral_holter_arc_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = EDANARCHolterParser().parse(neutral_holter_arc_file)
        assert len(record.leads) == 3
        # The std-jump scan rounds to a 4-KB block boundary, so a few
        # hundred samples may be trimmed at the very end — assert we
        # recovered at least 80% of the synthetic 10s × 250Hz signal.
        for lead in record.leads:
            assert lead.sampling_rate == 250
            assert lead.samples.dtype == np.float64
            assert 0.8 * 10 * 250 <= len(lead.samples) <= 10 * 250

    def test_picks_up_filename_timestamp(self, neutral_holter_arc_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = EDANARCHolterParser().parse(neutral_holter_arc_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2026
        assert record.recording.date.month == 5
        assert record.recording.date.day == 6

    def test_extracts_session_uuid_and_filename(self, neutral_holter_arc_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = EDANARCHolterParser().parse(neutral_holter_arc_file)
        assert record.raw_metadata["arc_variant"] == "neutral_holter"
        assert record.raw_metadata["reverse_engineered"] is True
        assert record.raw_metadata["session_uuid"] == "6a0c3d93-3b15"
        assert record.raw_metadata["embedded_filename"] == "patientdata.dat"

    def test_payload_boundary_found(self, neutral_holter_arc_file: Path):
        """The std-jump scan must land within one scan-block of the
        synthetic ECG end (10s × 250Hz × 3ch × 2B = 15000 bytes)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = EDANARCHolterParser().parse(neutral_holter_arc_file)
        assert record.raw_metadata["ecg_payload_start"] == 0x1000
        # The scan rounds down to a 4-KB block boundary, so end_offset is
        # the largest block-aligned position that still contains pure ECG.
        # Expected: 0x1000 + 3 full 4-KB blocks = 0x4000.
        end = record.raw_metadata["ecg_payload_end"]
        assert 0x1000 + 8000 <= end <= 0x1000 + 15000 + 4096

    def test_auto_detection_via_file_parser(self, neutral_holter_arc_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = FileParser().parse(neutral_holter_arc_file)
        assert record.source_format == "neutral_holter_arc"

    def test_archive_auto_detection_via_file_parser(self, edan_arc_archive: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = FileParser().parse(edan_arc_archive)
        assert record.source_format == "edan_arc_archive"

def _write_pair(directory: Path, hea: bytes, dat: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "patient.hea").write_bytes(hea)
    (directory / "ecgraw.dat").write_bytes(dat)
    return directory / "patient.hea"


def _parse_quiet(path: Path) -> ECGRecord:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return EDANARCHolterParser().parse(path)


def _expected_channel(ch: int, n: int) -> np.ndarray:
    """Signed samples written by create_edan_arc_dat for channel *ch*."""
    return (ch + 1) * 10 + (np.arange(n) % 50).astype(np.float64)


class TestEDANChannelLayout:
    """ecgraw.dat stride: header channel count vs 12 always stored."""

    def test_compact_layout_exact_values(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        for ch, lead in enumerate(record.leads):
            np.testing.assert_array_equal(lead.samples, _expected_channel(ch, 1000))
        assert record.raw_metadata["stored_channel_count"] == 3

    def test_twelve_stored_zero_padding(self, tmp_path: Path):
        dat = create_edan_arc_dat(3, 1000, stored_channels=12, pad_value=0)
        hea_path = _write_pair(tmp_path, create_edan_arc_hea(), dat)
        record = EDANARCHolterParser().parse(hea_path)
        assert [lead.label for lead in record.leads] == ["I", "II", "III"]
        for ch, lead in enumerate(record.leads):
            np.testing.assert_array_equal(lead.samples, _expected_channel(ch, 1000))
        sig = record.recording.acquisition.signal
        assert sig.number_channels_allocated == 12
        assert sig.number_channels_valid == 3

    def test_twelve_stored_adc_zero_padding_by_duration(self, tmp_path: Path):
        dat = create_edan_arc_dat(3, 1000, stored_channels=12, pad_value=16384)
        hea_path = _write_pair(tmp_path, create_edan_arc_hea(), dat)
        record = EDANARCHolterParser().parse(hea_path)
        assert record.raw_metadata["channel_layout_basis"] == "header duration"
        np.testing.assert_array_equal(record.leads[2].samples, _expected_channel(2, 1000))

    def test_twelve_stored_without_timestamps(self, tmp_path: Path):
        hea = create_edan_arc_hea(start_epoch=0, end_epoch=0)
        dat = create_edan_arc_dat(3, 1000, stored_channels=12, pad_value=16384)
        record = _parse_quiet(_write_pair(tmp_path, hea, dat))
        assert record.raw_metadata["stored_channel_count"] == 12
        np.testing.assert_array_equal(record.leads[1].samples, _expected_channel(1, 1000))
        assert record.recording.date is None

    def test_compact_layout_without_timestamps(self, tmp_path: Path):
        hea = create_edan_arc_hea(start_epoch=0, end_epoch=0)
        record = _parse_quiet(_write_pair(tmp_path, hea, create_edan_arc_dat()))
        assert record.raw_metadata["stored_channel_count"] == 3
        np.testing.assert_array_equal(record.leads[0].samples, _expected_channel(0, 1000))

    def test_twelve_channel_header(self, tmp_path: Path):
        labels = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
        hea = create_edan_arc_hea(channel_count=12, lead_labels=labels)
        record = EDANARCHolterParser().parse(_write_pair(tmp_path, hea, create_edan_arc_dat(12, 1000)))
        assert [lead.label for lead in record.leads][3:6] == ["aVR", "aVL", "aVF"]
        np.testing.assert_array_equal(record.leads[11].samples, _expected_channel(11, 1000))

    def test_no_plausible_layout_raises(self, tmp_path: Path):
        dat = np.zeros(3000, dtype="<u2").tobytes()
        hea_path = _write_pair(tmp_path, create_edan_arc_hea(), dat)
        with pytest.raises(CorruptedFileError, match="ADC zero"):
            EDANARCHolterParser().parse(hea_path)


class TestEDANRobustness:

    def test_odd_byte_count(self, tmp_path: Path):
        hea_path = _write_pair(tmp_path, create_edan_arc_hea(), create_edan_arc_dat() + b"\x01")
        with pytest.warns(UserWarning, match="odd byte count"):
            record = EDANARCHolterParser().parse(hea_path)
        assert len(record.leads[0].samples) == 1000

    def test_partial_trailing_frame(self, tmp_path: Path):
        dat = create_edan_arc_dat() + struct.pack("<H", 16384)
        hea_path = _write_pair(tmp_path, create_edan_arc_hea(), dat)
        with pytest.warns(UserWarning, match="partial frame"):
            record = EDANARCHolterParser().parse(hea_path)
        assert len(record.leads[0].samples) == 1000

    def test_header_without_labels_raises(self, tmp_path: Path):
        hea_path = _write_pair(tmp_path, create_edan_arc_hea()[:100], create_edan_arc_dat())
        with pytest.raises(CorruptedFileError, match="truncated"):
            EDANARCHolterParser().parse(hea_path)

    def test_truncated_header_warns(self, tmp_path: Path):
        hea_path = _write_pair(tmp_path, create_edan_arc_hea()[:2000], create_edan_arc_dat())
        with pytest.warns(UserWarning, match="truncated"):
            record = EDANARCHolterParser().parse(hea_path)
        assert record.patient.last_name == ""
        assert [lead.label for lead in record.leads] == ["I", "II", "III"]

    @pytest.mark.parametrize("rate", [0, -5, 20000])
    def test_bad_sampling_rate(self, tmp_path: Path, rate: int):
        hea_path = _write_pair(tmp_path, create_edan_arc_hea(sampling_rate=rate), create_edan_arc_dat())
        with pytest.raises(CorruptedFileError, match="sampling rate"):
            EDANARCHolterParser().parse(hea_path)

    def test_end_before_start(self, tmp_path: Path):
        hea = create_edan_arc_hea(end_epoch=1_701_421_800 - 100)
        with pytest.warns(UserWarning, match="not after the start"):
            record = EDANARCHolterParser().parse(_write_pair(tmp_path, hea, create_edan_arc_dat()))
        assert record.recording.end_date is None  # unusable header end is not replaced

    def test_implausible_start_epoch(self, tmp_path: Path):
        hea = create_edan_arc_hea(start_epoch=0xFFFFFFFF)
        record = _parse_quiet(_write_pair(tmp_path, hea, create_edan_arc_dat()))
        assert record.recording.date is None
        assert record.recording.end_date is None
        assert record.recording.duration == timedelta(seconds=5)

    def test_duration_mismatch_warns(self, tmp_path: Path):
        hea = create_edan_arc_hea(end_epoch=1_701_421_800 + 3600)
        with pytest.warns(UserWarning, match="duration is taken from the signal"):
            record = EDANARCHolterParser().parse(_write_pair(tmp_path, hea, create_edan_arc_dat()))
        assert record.recording.duration == timedelta(seconds=5)

    def test_epoch_timezone_flagged(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.recording.date == datetime(2023, 12, 1, 9, 10, 0)
        assert "unverified" in record.raw_metadata["epoch_timezone"]

    def test_duplicate_and_nonstandard_labels(self, tmp_path: Path):
        hea = create_edan_arc_hea(lead_labels=("II", "II", "avf"))
        record = EDANARCHolterParser().parse(_write_pair(tmp_path, hea, create_edan_arc_dat()))
        assert [lead.label for lead in record.leads] == ["II", "II_2", "aVF"]

    def test_empty_label_falls_back(self, tmp_path: Path):
        hea = create_edan_arc_hea(lead_labels=("I", "", "III"))
        record = EDANARCHolterParser().parse(_write_pair(tmp_path, hea, create_edan_arc_dat()))
        assert [lead.label for lead in record.leads] == ["I", "Ch2", "III"]


class TestEDANText:

    def test_gb18030_name(self, tmp_path: Path):
        name = "张三"
        hea = create_edan_arc_hea(patient_name=name.encode("gb18030"))
        record = EDANARCHolterParser().parse(_write_pair(tmp_path, hea, create_edan_arc_dat()))
        assert record.patient.last_name == name
        assert record.patient.first_name == ""

    def test_latin1_fallback_keeps_bytes(self, tmp_path: Path):
        # 0xFF is invalid in GB18030 and UTF-8: decoded as Latin-1
        hea = create_edan_arc_hea(patient_name=b"Jos\xe9 \xff")
        record = EDANARCHolterParser().parse(_write_pair(tmp_path, hea, create_edan_arc_dat()))
        assert record.raw_metadata["patient_name"] == "Jos\xe9 \xff"

    def test_western_name_split(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.patient.first_name == "Edan"
        assert record.patient.last_name == "TestPatient"

    def test_clinical_fields(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert record.patient.clinical_history == "Sinus rhythm\nHypertension"
        assert record.recording.referring_physician == "Dr Liu"
        assert record.recording.technician == "TechA"
        assert record.raw_metadata["telephone"] == "555-1234567"
        assert record.raw_metadata["dft_filter"] == "50Hz"


class TestEDANEntryPoints:

    def test_directory_input(self, edan_arc_dir: Path):
        record = FileParser().parse(edan_arc_dir.parent, auto_scale=False)
        assert record.source_format == "edan_arc"
        assert record.leads[0].samples[0] == 10.0

    def test_ecgraw_dat_input(self, edan_arc_dir: Path):
        record = FileParser().parse(edan_arc_dir.parent / "ecgraw.dat", auto_scale=False)
        assert record.source_format == "edan_arc"
        assert len(record.leads) == 3

    def test_uppercase_names(self, tmp_path: Path):
        (tmp_path / "PATIENT.HEA").write_bytes(create_edan_arc_hea())
        (tmp_path / "ECGRAW.DAT").write_bytes(create_edan_arc_dat())
        record = EDANARCHolterParser().parse(tmp_path / "PATIENT.HEA")
        assert len(record.leads[0].samples) == 1000

    def test_orphan_dat_not_claimed(self, tmp_path: Path):
        p = tmp_path / "ecgraw.dat"
        p.write_bytes(create_edan_arc_dat())
        assert EDANARCHolterParser.can_parse(p, p.read_bytes()[:4096]) is False

    def test_arc_without_signature_not_claimed(self, tmp_path: Path):
        p = tmp_path / "other.arc"
        p.write_bytes(b"\x01\x02" * 4096)
        assert EDANARCHolterParser.can_parse(p, p.read_bytes()[:4096]) is False

    def test_raw_counts_without_auto_scale(self, edan_arc_dir: Path):
        with pytest.warns(UserWarning, match="raw ADC"):
            record = FileParser().parse(edan_arc_dir, auto_scale=False)
        lead = record.leads[0]
        assert lead.is_raw and lead.units == ""
        np.testing.assert_array_equal(lead.samples, _expected_channel(0, 1000))


class TestEDANArcHeuristics:

    def test_arc_without_dat_marker_is_frame_aligned(self, tmp_path: Path):
        """Without an ecgraw.dat marker the payload follows the 3672-byte header."""
        p = tmp_path / "nomarker.arc"
        p.write_bytes(b"patient.hea\x00" + create_edan_arc_hea() + create_edan_arc_dat())
        record = _parse_quiet(p)
        for ch, lead in enumerate(record.leads):
            np.testing.assert_array_equal(lead.samples, _expected_channel(ch, 1000))

    def test_arc_with_markers_exact_values(self, edan_arc_archive: Path):
        record = _parse_quiet(edan_arc_archive)
        for ch, lead in enumerate(record.leads):
            np.testing.assert_array_equal(lead.samples, _expected_channel(ch, 1000))


def _neutral_arc(signal: np.ndarray, index_gap: int = 1024, index_ptr: int | None = None) -> bytes:
    """NEUTRAL HOLTER .arc with *signal* (frames x 3), zero gap, then beat-like noise."""
    header = bytearray(0x1000)
    struct.pack_into("<I", header, 0, 3)
    header[4:32] = b"##NEUTRAL HOLTER RECORDING##"
    payload = signal.astype("<i2").tobytes()
    tail = np.random.default_rng(1).integers(-20000, 20000, size=8192, dtype=np.int16).tobytes()
    gap = b"\x00" * index_gap
    if index_ptr is None:
        index_ptr = 0x1000 + len(payload) + len(gap)
    struct.pack_into("<I", header, 0x56, index_ptr)
    return bytes(header) + payload + gap + tail


class TestNeutralHolterEnd:
    def _signal(self, seconds: int = 60) -> np.ndarray:
        t = np.arange(seconds * 250) / 250
        sig = np.round(20 * np.sin(2 * np.pi * 1.2 * t))[:, None].repeat(3, 1)
        sig[:, 1] += 5
        return sig

    def _parse(self, tmp_path: Path, blob: bytes):
        p = tmp_path / "DT-01_02_2025-10_00_00.arc"
        p.write_bytes(blob)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record = EDANARCHolterParser().parse(p)
        return record, [str(w.message) for w in caught]

    def test_artifact_inside_ecg_is_kept(self, tmp_path: Path):
        sig = self._signal()
        sig[20 * 250:24 * 250] += np.random.default_rng(0).normal(0, 1500, size=(1000, 3))
        record, messages = self._parse(tmp_path, _neutral_arc(sig))
        np.testing.assert_array_equal(record.leads[0].samples, sig[:, 0].astype("<i2"))
        assert any("high-amplitude" in m for m in messages)
        assert record.raw_metadata["high_amplitude_blocks_kept"] > 0

    def test_exact_end_and_zero_gap_dropped(self, tmp_path: Path):
        sig = self._signal(10)
        record, _ = self._parse(tmp_path, _neutral_arc(sig))
        assert len(record.leads[0].samples) == 2500
        assert record.raw_metadata["trailing_zero_frames_dropped"] == 1024 // 6

    def test_invalid_index_pointer_warns(self, tmp_path: Path):
        sig = self._signal(10)
        record, messages = self._parse(tmp_path, _neutral_arc(sig, index_ptr=0xFFFFFFF0))
        assert any("index pointer" in m for m in messages)
        assert len(record.leads[0].samples) == 2500
        assert record.raw_metadata["date_source"] == "filename"


def test_neutral_end_detected_to_the_frame(tmp_path: Path):
    # ECG of 2503 frames (not a multiple of 16) followed directly by beat records
    import struct
    from tests.conftest import create_neutral_holter_arc

    _, base = create_neutral_holter_arc(duration_s=1)
    header = bytearray(base[:0x1000])
    frames = 2503
    t = np.arange(frames) / 250
    ecg = np.stack([(20 * np.sin(2 * np.pi * 1.2 * t + ch * np.pi / 4) + 3).astype("<i2")
                    for ch in range(3)], axis=1)
    tail = np.random.default_rng(0).integers(-20000, 20000, 8192).astype("<i2").tobytes()
    struct.pack_into("<I", header, 0x56, 0x1000 + ecg.nbytes + len(tail))
    path = tmp_path / "DT-06_05_2026-11_38_39.arc"
    path.write_bytes(bytes(header) + ecg.tobytes() + tail)
    with pytest.warns(UserWarning):
        record = EDANARCHolterParser().parse(path)
    assert len(record.leads[0].samples) == frames
    np.testing.assert_array_equal(record.leads[2].samples, ecg[:, 2])


def test_neutral_short_recording_records_in_last_partial_block(tmp_path: Path):
    # 683 frames of ECG: the beat records start inside the final, partial 4 KB block
    from tests.conftest import create_neutral_holter_arc

    _, base = create_neutral_holter_arc(duration_s=1)
    header = bytearray(base[:0x1000])
    frames = 683
    t = np.arange(frames) / 250
    ecg = np.stack([(20 * np.sin(2 * np.pi * 1.2 * t + ch * np.pi / 4) + 3).astype("<i2")
                    for ch in range(3)], axis=1)
    tail = np.random.default_rng(1).integers(-20000, 20000, 750).astype("<i2").tobytes()
    struct.pack_into("<I", header, 0x56, 0x1000 + ecg.nbytes + len(tail))
    path = tmp_path / "DT-06_05_2026-11_38_39.arc"
    path.write_bytes(bytes(header) + ecg.tobytes() + tail)
    with pytest.warns(UserWarning):
        record = EDANARCHolterParser().parse(path)
    assert len(record.leads[0].samples) == frames
