"""Tests for the EDAN ARC / SE-2012 Holter parser."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.edan_arc import EDANARCHolterParser


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
        assert record.recording.duration.total_seconds() == pytest.approx(10.0)

    def test_lead_count(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        assert len(record.leads) == 3

    def test_lead_labels(self, edan_arc_dir: Path):
        record = EDANARCHolterParser().parse(edan_arc_dir)
        labels = [l.label for l in record.leads]
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
        assert record.recording.device.model == "SE2012"

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
            "source_format", "patient", "recording",
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
            and "best-effort" in str(w.message)
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
        """Heuristic .arc parse should match the documented-layout parse."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            arc_rec = EDANARCHolterParser().parse(edan_arc_archive)
        doc_rec = EDANARCHolterParser().parse(edan_arc_dir)
        for arc_lead, doc_lead in zip(arc_rec.leads, doc_rec.leads):
            np.testing.assert_array_equal(arc_lead.samples, doc_lead.samples)
            assert arc_lead.label == doc_lead.label

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

    def test_neutral_holter_archive_rejected_cleanly(self, tmp_path: Path):
        """`##NEUTRAL HOLTER RECORDING##` archives must be refused with a
        clear message rather than fingerprint-scanned into garbage."""
        p = tmp_path / "neutral.arc"
        body = bytearray(b"\x03\x00\x00\x00")
        body.extend(b"##NEUTRAL HOLTER RECORDING##")
        body.extend(b"\x00" * (8192 - len(body)))
        p.write_bytes(bytes(body))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(CorruptedFileError, match="NEUTRAL HOLTER"):
                EDANARCHolterParser().parse(p)

    def test_auto_detection_via_file_parser(self, edan_arc_archive: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = FileParser().parse(edan_arc_archive)
        assert record.source_format == "edan_arc_archive"