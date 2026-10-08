"""Tests for the EDF/EDF+ parser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.edf import EDFParser


class TestEDFParser:
    def test_parse_returns_ecg_record(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert isinstance(record, ECGRecord)

    def test_source_format_edfplus(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.source_format == "edf_plus"

    def test_patient_id(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.patient.patient_id == "P001"

    def test_patient_sex(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.patient.sex == "M"

    def test_patient_birth_date(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.patient.birth_date is not None
        assert record.patient.birth_date.year == 1980

    def test_recording_date(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.recording.date is not None
        assert record.recording.date.year == 2023
        assert record.recording.date.month == 12

    def test_recording_sampling_rate(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.recording.acquisition.signal.sampling_rate == 500

    def test_recording_duration(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert record.recording.duration is not None
        assert record.recording.duration.total_seconds() > 0

    def test_lead_count(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        assert len(record.leads) == 2

    def test_lead_labels(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        labels = [l.label for l in record.leads]
        assert labels == ["I", "II"]

    def test_lead_samples_are_float(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        for lead in record.leads:
            assert lead.samples.dtype == np.float64
            assert len(lead.samples) > 0

    def test_lead_sampling_rate(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        for lead in record.leads:
            assert lead.sampling_rate == 500

    def test_lead_units_and_is_raw(self, edf_file: Path):
        """Fixture has gain≈9.77e-5 (phys 6.4 / dig 65535) → raw ADC."""
        record = EDFParser().parse(edf_file)
        for lead in record.leads:
            assert lead.resolution_unit == "mV"
            assert lead.units == ""
            assert lead.resolution != 1.0
            assert lead.is_raw is True

    def test_corrupted_file_too_small(self, tmp_path: Path):
        f = tmp_path / "tiny.edf"
        f.write_bytes(b"0       " + b"\x00" * 10)
        with pytest.raises(Exception):
            EDFParser().parse(f)

    def test_to_dict_unified_schema(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        d = record.to_dict()
        assert set(d.keys()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, edf_file: Path):
        record = EDFParser().parse(edf_file)
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["source_format"] in ("edf", "edf_plus")
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, edf_file: Path):
        fp = FileParser()
        record = fp.parse(edf_file)
        assert record.source_format in ("edf", "edf_plus")


# ---------------------------------------------------------------------------
# Spec-based builder (https://www.edfplus.info/specs/edf.html, edfplus.html)
# ---------------------------------------------------------------------------

import struct
from datetime import datetime

from ecgdatakit.exceptions import CorruptedFileError, UnsupportedFormatError


def _sig(label="ECG I", spr=4, dim="uV", pmin=-3200, pmax=3200, dmin=-32768,
         dmax=32767, prefilter="", data=None):
    return dict(label=label, spr=spr, dim=dim, pmin=pmin, pmax=pmax, dmin=dmin,
                dmax=dmax, prefilter=prefilter, data=data)


def _ann(spr=30, tals=None):
    """Annotation signal; *tals* maps record index to extra TAL bytes."""
    tals = tals or {}

    def data(rec, onset):
        block = f"+{onset:g}\x14\x14\x00".encode() + tals.get(rec, b"")
        return block.ljust(spr * 2, b"\x00")

    return dict(label="EDF Annotations", spr=spr, dim="", pmin=-1, pmax=1,
                dmin=-32768, dmax=32767, prefilter="", data=data)


def build_edf(signals, nrec=2, duration="1", reserved="EDF+C",
              patient="P1 M 15-JUN-1980 Doe,_John",
              recording="Startdate 01-DEC-2023 PSG-12 Tech_A Cardio_X",
              date="01.12.23", time="10.30.00", nrec_field=None, header_bytes=None,
              onsets=None):
    def f(value, n):
        b = value if isinstance(value, bytes) else str(value).encode("latin-1")
        return b.ljust(n, b" ")[:n]

    ns = len(signals)
    hdr = (b"0       " + f(patient, 80) + f(recording, 80) + f(date, 8) + f(time, 8)
           + f(header_bytes or 256 * (ns + 1), 8) + f(reserved, 44)
           + f(nrec if nrec_field is None else nrec_field, 8) + f(duration, 8) + f(ns, 4))
    for key, n in [("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8),
                   ("pmax", 8), ("dmin", 8), ("dmax", 8), ("prefilter", 80),
                   ("spr", 8), ("reserved", 32)]:
        for s in signals:
            hdr += f(s.get(key, "AgAgCl" if key == "transducer" else ""), n)
    body = b""
    for rec in range(nrec):
        for s in signals:
            if s["label"] == "EDF Annotations":
                onset = onsets[rec] if onsets else rec * float(duration)
                body += s["data"](rec, onset)
            elif s["data"] is not None:
                body += struct.pack(f"<{s['spr']}h", *s["data"][rec])
            else:
                body += struct.pack(f"<{s['spr']}h", *[rec * 100 + k for k in range(s["spr"])])
    return hdr + body


def _parse(tmp_path: Path, data: bytes):
    p = tmp_path / "t.edf"
    p.write_bytes(data)
    return EDFParser().parse(p)


class TestEDFSignals:
    def test_scaling_formula(self, tmp_path: Path):
        sig = _sig(pmin=-5, pmax=5, dmin=-2048, dmax=2047, dim="mV", spr=2,
                   data=[[-2048, 2047], [0, 1]])
        lead = _parse(tmp_path, build_edf([sig])).leads[0]
        assert lead.samples.tolist() == [-2048, 2047, 0, 1]
        phys = lead.samples * lead.resolution + lead.offset
        assert phys[0] == pytest.approx(-5) and phys[1] == pytest.approx(5)
        assert lead.resolution == pytest.approx(10 / 4095)
        assert lead.resolution_unit == "mV" and lead.adc_resolution_unit == "mV"
        assert lead.annotations["digital_minimum"] == "-2048"

    def test_mixed_rates_and_annotation_excluded(self, tmp_path: Path):
        sigs = [_sig("ECG II", spr=4), _ann(), _sig("Resp", spr=1, dim="%", pmin=0, pmax=100)]
        record = _parse(tmp_path, build_edf(sigs, nrec=3))
        ii, resp = record.leads
        assert ii.label == "II" and ii.sampling_rate == 4
        assert ii.samples.tolist() == [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203]
        assert resp.sampling_rate == 1 and resp.samples.tolist() == [0, 100, 200]
        assert resp.resolution_unit == "%"
        assert record.recording.acquisition.signal.sampling_rate == 4

    def test_labels_normalised_and_unique(self, tmp_path: Path):
        sigs = [_sig(n, spr=1) for n in ["ECG I", "ECG aVR", "ECG V1", "ECG V1"]]
        record = _parse(tmp_path, build_edf(sigs))
        assert [l.label for l in record.leads] == ["I", "aVR", "V1", "V1_2"]

    def test_rate_is_rounded(self, tmp_path: Path):
        record = _parse(tmp_path, build_edf([_sig(spr=7)], duration="0.07"))
        assert record.leads[0].sampling_rate == 100
        record = _parse(tmp_path, build_edf([_sig(spr=1)], duration="2"))
        assert record.raw_metadata["exact_sampling_rates"] == {"I": 0.5}

    def test_invalid_digital_range_left_uncalibrated(self, tmp_path: Path):
        with pytest.warns(UserWarning, match="invalid range"):
            record = _parse(tmp_path, build_edf([_sig(dmin=100, dmax=-100)]))
        lead = record.leads[0]
        assert lead.is_raw and lead.resolution_unit == "" and lead.resolution == 1.0

    def test_auto_scale_gives_mv(self, tmp_path: Path):
        sig = _sig(pmin=-3276.8, pmax=3276.7, dmin=-32768, dmax=32767, dim="uV",
                   spr=1, data=[[1000], [-1000]])
        p = tmp_path / "s.edf"
        p.write_bytes(build_edf([sig]))
        scaled = FileParser().parse(p)
        assert scaled.leads[0].units == "mV"
        np.testing.assert_allclose(scaled.leads[0].samples, [0.1, -0.1])
        with pytest.warns(UserWarning, match="raw ADC"):
            raw = FileParser().parse(p, auto_scale=False)
        assert raw.leads[0].samples.tolist() == [1000, -1000]


class TestEDFRecords:
    def test_num_records_minus_one(self, tmp_path: Path):
        with pytest.warns(UserWarning, match="-1"):
            record = _parse(tmp_path, build_edf([_sig()], nrec=3, nrec_field=-1))
        assert record.raw_metadata["num_records"] == 3
        assert len(record.leads[0].samples) == 12

    def test_truncated_file_is_trimmed(self, tmp_path: Path):
        data = build_edf([_sig(), _sig("ECG II")], nrec=3)[:-10]
        with pytest.warns(UserWarning, match="truncated"):
            record = _parse(tmp_path, data)
        assert record.leads[0].samples.tolist() == [0, 1, 2, 3, 100, 101, 102, 103]
        assert record.recording.duration.total_seconds() == 2

    def test_no_complete_record_raises(self, tmp_path: Path):
        data = build_edf([_sig()], nrec=1)[:-2]
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, data)

    def test_header_size_mismatch_uses_spec_size(self, tmp_path: Path):
        with pytest.warns(UserWarning, match="header size"):
            record = _parse(tmp_path, build_edf([_sig()], header_bytes=100))
        assert record.leads[0].samples[:4].tolist() == [0, 1, 2, 3]

    @pytest.mark.parametrize("kwargs", [
        {"nrec_field": "abc"},
        {"duration": "x"},
        {"signals": [_sig(pmin="x")]},
        {"signals": [_sig(spr=0)]},
        {"duration": "0"},
    ])
    def test_corrupt_header_raises(self, tmp_path: Path, kwargs):
        kwargs = dict(kwargs)
        signals = kwargs.pop("signals", [_sig()])
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, build_edf(signals, **kwargs))

    def test_truncated_signal_header_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, build_edf([_sig()])[:300])

    def test_bdf_unsupported(self, tmp_path: Path):
        with pytest.raises(UnsupportedFormatError):
            _parse(tmp_path, b"\xffBIOSEMI" + build_edf([_sig()])[8:])


class TestEDFAnnotations:
    def test_utf8_tal_in_record_annotations(self, tmp_path: Path):
        tal = "+0.5\x150.25\x14Café\x14second\x14\x00+1\x14note\x14\x00".encode("utf-8")
        record = _parse(tmp_path, build_edf([_sig(), _ann(tals={0: tal})]))
        assert record.raw_metadata["edf_annotations"] == [
            {"onset": 0.5, "duration": 0.25, "text": "Café"},
            {"onset": 0.5, "duration": 0.25, "text": "second"},
            {"onset": 1.0, "duration": None, "text": "note"},
        ]
        assert record.annotations["edf_annotations"] == (
            "+0.5\t0.25\tCafé\n+0.5\t0.25\tsecond\n+1\t\tnote")
        assert "Café" in json.loads(record.to_json())["annotations"]["edf_annotations"]

    def test_edfplus_d_gaps(self, tmp_path: Path):
        data = build_edf([_sig(), _ann()], nrec=3, reserved="EDF+D", onsets=[0, 1, 10])
        with pytest.warns(UserWarning, match="gaps"):
            record = _parse(tmp_path, data)
        assert record.file_format.version == "EDF+D"
        assert record.raw_metadata["record_onsets"] == [0, 1, 10]
        assert record.recording.duration.total_seconds() == 3
        assert record.recording.end_date is None  # EDF stores no end time

    def test_subsecond_start(self, tmp_path: Path):
        record = _parse(tmp_path, build_edf([_sig(), _ann()], nrec=2, onsets=[0.25, 1.25]))
        assert record.recording.date == datetime(2023, 12, 1, 10, 30, 0, 250000)


class TestEDFIdentification:
    def test_edfplus_fields(self, tmp_path: Path):
        record = _parse(tmp_path, build_edf([_sig()]))
        assert record.patient.patient_id == "P1"
        assert (record.patient.last_name, record.patient.first_name) == ("Doe", "John")
        assert record.patient.birth_date == datetime(1980, 6, 15)
        assert record.patient.age == 43
        assert record.recording.technician == "Tech A"
        assert record.recording.device.model == "Cardio X"
        assert record.raw_metadata["admin_code"] == "PSG-12"
        assert record.file_format.version == "EDF+C"

    def test_latin1_header_text(self, tmp_path: Path):
        record = _parse(tmp_path, build_edf([_sig()], patient="P1 F X M\xfcller_J\xfcrgen"))
        assert record.patient.last_name == "Müller Jürgen"

    def test_year_after_2084_from_startdate(self, tmp_path: Path):
        record = _parse(tmp_path, build_edf(
            [_sig()], date="01.12.yy", recording="Startdate 01-DEC-2090 X X X"))
        assert record.recording.date == datetime(2090, 12, 1, 10, 30)

    def test_plain_edf_clipping_year(self, tmp_path: Path):
        record = _parse(tmp_path, build_edf(
            [_sig()], reserved="", date="31.12.85", patient="free text"))
        assert record.recording.date == datetime(1985, 12, 31, 10, 30)
        assert record.patient.patient_id == "free text"
        assert record.file_format.version == "EDF"
        assert record.source_format == "edf"

    def test_prefiltering(self, tmp_path: Path):
        sigs = [_sig(prefilter="HP:0.05Hz LP:150Hz N:50Hz"),
                _sig("ECG II", prefilter="HP:0.05Hz LP:150Hz N:50Hz")]
        filters = _parse(tmp_path, build_edf(sigs)).recording.acquisition.filters
        assert (filters.highpass, filters.lowpass, filters.notch) == (0.05, 150.0, 50.0)
