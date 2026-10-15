"""Tests for the XLE ECG parser (experimental), on synthetic files."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.xle import XLEParser
from tests.test_anonymize import ALL_FORMATS, _any_format_file

HEADER = 0x40000
FIELDS = (("Given name", 2673), ("Family name", 2712), ("ID", 2761), ("Birth Date", 2793),
          ("Sex", 2840), ("History", 2876), ("Comment", 2916))
DATE_OFFSET, TIME_OFFSET = 182972, 182982
QRS_OFFSET, COUNTS_OFFSET, AVERAGE_OFFSET = 0xB8A, 0xB94, 0xB9C
CR_LEADS = ["I", "II", "CR1", "CR2", "CR3", "CR4", "CR5", "CR6"]
V_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
TWELVE_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _swap16(data: bytes) -> bytes:
    out = bytearray(data)
    out[0::2], out[1::2] = data[1::2], data[0::2]
    return bytes(out)


def _int32_words(values: np.ndarray) -> bytes:
    """32-bit values as two 16-bit little-endian words, high word first."""
    values = np.asarray(values, dtype=np.int64).ravel()
    words = np.empty(2 * values.size, dtype="<u2")
    words[0::2] = (values >> 16) & 0xFFFF
    words[1::2] = values & 0xFFFF
    return words.tobytes()


def average_beat() -> np.ndarray:
    """5000 samples x 8 of an averaged beat."""
    t = np.arange(5000)
    return np.stack([(k + 1) * 300 * np.exp(-((t - 2000) / 60.0) ** 2) - 50 * k
                     for k in range(8)], axis=1).round()


def average_table(beats: int = 40, rows: int = 5000) -> np.ndarray:
    """5000 rows x 9 as stored: on 8 channels the change from the previous
    row of the sum of *beats* copies of :func:`average_beat`, then the number
    of beats summed (half of them in the last 1000 rows, none after *rows*)."""
    sums = average_beat() * beats
    changes = np.diff(sums, axis=0, prepend=0)
    counts = np.where(np.arange(5000) < 4000, beats, beats // 2)
    counts[rows:] = 0
    return np.column_stack([changes, counts]).astype(np.int64)


def make_xle(path: Path, signal: np.ndarray, values: dict[str, str] | None = None,
             date: str = "", time: str = "", average: np.ndarray | None = None,
             qrs: tuple[int, int] = (380, 470), detected: int = 45) -> Path:
    """Write an XLE-like file: patient labels and values in a byte-swapped
    header with the recording *date* and *time* and the representative beat
    *average* (see :func:`average_table`), then *signal* (frames x 8) stored
    as int16 modulo 65536."""
    values = values or {}
    text = bytearray(3200)
    for label, off in FIELDS:
        entry = f"{label}: {values.get(label, '')}".encode("latin-1")
        text[off:off + len(entry)] = entry
    header = bytearray(HEADER)
    header[:3200] = _swap16(bytes(text))
    header[DATE_OFFSET:DATE_OFFSET + 8] = _swap16(date.encode("latin-1").ljust(8, b"\x00"))
    header[TIME_OFFSET:TIME_OFFSET + 6] = _swap16(time.encode("latin-1").ljust(6, b"\x00"))
    if average is not None:
        header[QRS_OFFSET:QRS_OFFSET + 8] = np.array(
            [qrs[0], qrs[1], 5 * qrs[0], 5 * qrs[1]], dtype="<u2").tobytes()
        header[COUNTS_OFFSET:COUNTS_OFFSET + 8] = _int32_words([detected, average[0, 8]])
        table = _int32_words(average)
        header[AVERAGE_OFFSET:AVERAGE_OFFSET + len(table)] = table
    stored = ((np.asarray(signal, dtype=np.int64) + 32768) % 65536 - 32768).astype("<i2")
    path.write_bytes(bytes(header) + stored.tobytes())
    return path


def ecg_like(frames: int = 3000, drift: int = 0) -> np.ndarray:
    """8 channels of beats every 750 samples on a baseline drifting by *drift*."""
    t = np.arange(frames)
    beat = 1500 * np.exp(-(((t % 750) - 375) / 15.0) ** 2)
    base = np.linspace(0, drift, frames)
    return np.stack([(k + 1) * beat / 4 + base + 1000 * k for k in range(8)], axis=1).round()


def parse(path: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FileParser().parse(path, **kwargs)


def caught_messages(func, *args, **kwargs) -> tuple[object, list[str]]:
    """Result of *func* and the messages of the warnings it issued."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def has(messages: list[str], text: str) -> bool:
    return any(text in m for m in messages)


EXPERIMENTAL = "XLE support is experimental"
SCALE = "XLE: the voltage scale is not stored"
AVERAGE = "XLE representative beat"


class TestXLE:
    def test_detected_and_read(self, tmp_path):
        signal = ecg_like()
        path = make_xle(tmp_path / "rec.xle", signal)
        record = parse(path, auto_scale=False)
        assert record.file_format.parser == "XLEParser"
        assert [lead.label for lead in record.leads] == CR_LEADS
        assert all(lead.sampling_rate == 1000 for lead in record.leads)
        assert record.recording.duration == timedelta(seconds=3)
        for k, lead in enumerate(record.leads):
            np.testing.assert_array_equal(lead.samples, signal[:, k])

    def test_wraps_undone(self, tmp_path):
        # the baseline drifts by 150 000 counts: stored values wrap twice
        signal = ecg_like(frames=20000, drift=150_000)
        path = make_xle(tmp_path / "rec.xle", signal)
        record = parse(path, auto_scale=False)
        for k, lead in enumerate(record.leads):
            np.testing.assert_array_equal(lead.samples - lead.samples[0],
                                          signal[:, k] - signal[0, k])
        assert record.raw_metadata["wraps_undone"] > 0

    def test_converted_to_mv_with_a_warning(self, tmp_path):
        signal = ecg_like()
        path = make_xle(tmp_path / "rec.xle", signal)
        record, messages = caught_messages(FileParser().parse, path)   # auto_scale=True
        assert has(messages, EXPERIMENTAL) and has(messages, SCALE)
        for k, lead in enumerate(record.leads):
            assert not lead.is_raw and lead.units == "mV"
            np.testing.assert_allclose(lead.samples, signal[:, k] * 0.9375 / 1000)

    def test_raw_counts_keep_the_scale(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like())
        record, messages = caught_messages(FileParser().parse, path, auto_scale=False)
        # no conversion: the format warning only, not the scale warning
        assert has(messages, EXPERIMENTAL) and not has(messages, SCALE)
        lead = record.leads[0]
        assert lead.is_raw and lead.units == ""
        assert (lead.resolution, lead.resolution_unit) == (0.9375, "uV")

    def test_scale_warning_on_manual_conversion(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like())
        record = parse(path, auto_scale=False)
        physical, messages = caught_messages(record.to_physical)
        assert has(messages, SCALE)
        # already converted: no second warning
        _, again = caught_messages(physical.to_physical)
        assert not has(again, SCALE)

    def test_experimental_warning(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like())
        with pytest.warns(UserWarning, match="XLE support is experimental"):
            XLEParser().parse(path)

    def test_no_wilson_correction_by_default(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            XLEParser().parse(path)
        assert not any("Wilson" in str(w.message) for w in caught)

    def test_wilson_correction(self, tmp_path):
        signal = ecg_like()
        path = make_xle(tmp_path / "rec.xle", signal)
        with pytest.warns(UserWarning, match="Wilson correction applied"):
            record = XLEParser(wilson_correction=True).parse(path)
        assert [lead.label for lead in record.leads] == V_LEADS
        wct = (signal[:, 0] + signal[:, 1]) / 3
        for k, lead in enumerate(record.leads):
            expected = signal[:, k] - wct if k >= 2 else signal[:, k]
            np.testing.assert_allclose(lead.samples, expected)
        assert record.raw_metadata["wilson_correction"] is True
        assert record.raw_metadata["chest_lead_reference"] == "Wilson central terminal"
        # converted with the same scale
        with pytest.warns(UserWarning, match=SCALE):
            mv = record.to_physical().convert_units("mV")
        np.testing.assert_allclose(mv.leads[2].samples, (signal[:, 2] - wct) * 0.9375 / 1000)

    def test_inferred_values_are_marked(self, tmp_path):
        record = parse(make_xle(tmp_path / "rec.xle", ecg_like()))
        meta = record.raw_metadata
        assert meta["reverse_engineered"] is True
        assert meta["sampling_rate_stated"] is False
        assert meta["resolution_stated"] is False
        assert meta["lead_order_stated"] is False
        assert meta["wilson_correction"] is False
        assert meta["chest_lead_reference"] == "right arm"
        assert meta["computed_leads"] == []

    def test_compute_missing_leads(self, tmp_path):
        signal = ecg_like()
        path = make_xle(tmp_path / "rec.xle", signal)
        record, messages = caught_messages(
            XLEParser(wilson_correction=True, compute_missing_leads=True).parse, path)
        assert has(messages, "computed from leads I and II")
        leads = {lead.label: lead.samples for lead in record.leads}
        assert list(leads) == TWELVE_LEADS
        i, ii = signal[:, 0], signal[:, 1]
        np.testing.assert_allclose(leads["III"], ii - i)
        np.testing.assert_allclose(leads["aVR"], -(i + ii) / 2)
        np.testing.assert_allclose(leads["aVL"], i - ii / 2)
        np.testing.assert_allclose(leads["aVF"], ii - i / 2)
        # stored limb leads unchanged
        np.testing.assert_array_equal(leads["I"], i)
        np.testing.assert_array_equal(leads["II"], ii)
        assert record.raw_metadata["computed_leads"] == ["III", "aVR", "aVL", "aVF"]
        # computed leads are converted with the same scale
        mv = record.to_physical().convert_units("mV")
        np.testing.assert_allclose(mv.leads[2].samples, (ii - i) * 0.9375 / 1000)

    def test_missing_leads_require_wilson(self):
        with pytest.raises(ValueError, match="requires wilson_correction=True"):
            XLEParser(compute_missing_leads=True)

    def test_standard_twelve_leads(self, tmp_path):
        signal = ecg_like()
        path = make_xle(tmp_path / "rec.xle", signal)
        record, messages = caught_messages(
            XLEParser(wilson_correction=True, compute_missing_leads=True).parse, path)
        assert [lead.label for lead in record.leads] == TWELVE_LEADS
        assert has(messages, "Wilson correction applied")
        assert has(messages, "computed from leads I and II")
        wct = (signal[:, 0] + signal[:, 1]) / 3
        np.testing.assert_allclose(record.leads[6].samples, signal[:, 2] - wct)
        # Einthoven: I + III = II
        np.testing.assert_allclose(record.leads[0].samples + record.leads[2].samples,
                                   record.leads[1].samples)
        # Goldberger: aVR + aVL + aVF = 0
        np.testing.assert_allclose(
            record.leads[3].samples + record.leads[4].samples + record.leads[5].samples, 0,
            atol=1e-9)

    @pytest.mark.parametrize("date, time, expected", [
        ("27.04.99", "08:15", datetime(1999, 4, 27, 8, 15)),     # dots: day first
        ("04/27/01", "17:05", datetime(2001, 4, 27, 17, 5)),     # slashes: month first
        ("05.06.02", "", datetime(2002, 6, 5)),
        ("", "", None),
        ("31.13.99", "10:00", None),
    ])
    def test_recording_date(self, tmp_path, date, time, expected):
        record = parse(make_xle(tmp_path / "rec.xle", ecg_like(), date=date, time=time))
        assert record.recording.date == expected
        assert record.raw_metadata["recording_date_text"] == date
        assert record.raw_metadata["recording_time_text"] == time


class TestRepresentativeBeat:
    def test_none_without_average(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like())
        record, messages = caught_messages(XLEParser().parse, path)
        assert record.median_beats == []
        assert not has(messages, AVERAGE)
        assert record.annotations == {}
        assert record.measurements.qrs_count is None
        assert "median_scale_stated" not in record.raw_metadata

    def test_beat_rebuilt_and_averaged(self, tmp_path):
        table = average_table()
        path = make_xle(tmp_path / "rec.xle", ecg_like(), average=table)
        record, messages = caught_messages(XLEParser().parse, path)
        assert has(messages, AVERAGE)
        assert [beat.label for beat in record.median_beats] == CR_LEADS
        for k, beat in enumerate(record.median_beats):
            assert beat.sampling_rate == 5000
            np.testing.assert_allclose(beat.samples, average_beat()[:, k])
        assert record.raw_metadata["median_beat_counts"] == table[:, 8].tolist()
        assert record.annotations == {"median_qrs_onset_ms": "380",
                                      "median_qrs_offset_ms": "470",
                                      "median_beats_averaged": "40"}
        assert record.measurements.qrs_count == 45
        meta = record.raw_metadata
        assert meta["median_sampling_rate_stated"] is False
        assert meta["median_scale_stated"] is False
        assert meta["median_lead_order_stated"] is False

    def test_ends_where_no_beat_is_summed(self, tmp_path):
        table = average_table(rows=4500)
        table[4500:, :8] = 123_456_789   # rows after the end are not read
        path = make_xle(tmp_path / "rec.xle", ecg_like(), average=table)
        record = parse(path, auto_scale=False)
        for k, beat in enumerate(record.median_beats):
            np.testing.assert_allclose(beat.samples, average_beat()[:4500, k])
        assert len(record.raw_metadata["median_beat_counts"]) == 4500

    def test_converted_with_the_beat_scale(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like(), average=average_table())
        record, messages = caught_messages(FileParser().parse, path)   # auto_scale=True
        assert has(messages, SCALE) and has(messages, "0.9375 / 1.6 uV per count")
        beat = record.median_beats[0]
        assert not beat.is_raw and beat.units == "mV"
        np.testing.assert_allclose(beat.samples, average_beat()[:, 0] * 0.9375 / 1.6 / 1000)
        assert record.raw_metadata["median_scale_factor"] == 1.6

    def test_raw_counts_keep_the_beat_scale(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like(), average=average_table())
        record = parse(path, auto_scale=False)
        beat = record.median_beats[0]
        assert beat.is_raw and beat.units == ""
        assert (beat.resolution, beat.resolution_unit) == (0.9375 / 1.6, "uV")
        np.testing.assert_allclose(beat.samples, average_beat()[:, 0])
        _, messages = caught_messages(record.to_physical)
        assert has(messages, "0.9375 / 1.6 uV per count")

    def test_no_beat_scale_note_without_beat(self, tmp_path):
        record, messages = caught_messages(
            FileParser().parse, make_xle(tmp_path / "rec.xle", ecg_like()))
        assert has(messages, SCALE) and not has(messages, "1.6")

    def test_wilson_and_missing_leads(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like(), average=average_table())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = XLEParser(wilson_correction=True, compute_missing_leads=True).parse(path)
        beats = {beat.label: beat.samples for beat in record.median_beats}
        assert list(beats) == TWELVE_LEADS
        mean = average_beat()
        i, ii = mean[:, 0], mean[:, 1]
        np.testing.assert_allclose(beats["V1"], mean[:, 2] - (i + ii) / 3)
        np.testing.assert_allclose(beats["III"], ii - i)
        np.testing.assert_allclose(beats["aVR"], -(i + ii) / 2)
        np.testing.assert_allclose(beats["aVL"], i - ii / 2)
        np.testing.assert_allclose(beats["aVF"], ii - i / 2)

    def test_wilson_only(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like(), average=average_table())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = XLEParser(wilson_correction=True).parse(path)
        assert [beat.label for beat in record.median_beats] == V_LEADS


@pytest.mark.parametrize("name", ALL_FORMATS)
def test_xle_warnings_only_for_xle(name, request, tmp_path):
    """FileParser is shared: the XLE warnings must not reach other formats."""
    path = _any_format_file(name, request, tmp_path)[0]
    record, messages = caught_messages(FileParser().parse, path)
    _, manual = caught_messages(parse(path, auto_scale=False).to_physical)
    is_xle = name == "xle"
    assert has(messages, EXPERIMENTAL) == is_xle
    assert has(messages, SCALE) == is_xle
    assert has(manual, SCALE) == is_xle
    assert not has(messages, AVERAGE) or is_xle
    assert ("scale_warning" in record.raw_metadata) == is_xle

    def test_patient_fields(self, tmp_path):
        values = {"Given name": "Zoe", "Family name": "Doe", "ID": "SUBJ-0042",
                  "Birth Date": "01.02.03", "Sex": "w", "History": "none", "Comment": "test"}
        record = parse(make_xle(tmp_path / "rec.xle", ecg_like(), values))
        p = record.patient
        assert (p.first_name, p.last_name, p.patient_id) == ("Zoe", "Doe", "SUBJ-0042")
        assert p.sex == "F"                            # w: weiblich
        assert p.clinical_history == "none"
        # a free-text birth date is kept as text, never turned into a date
        assert p.birth_date is None
        assert record.raw_metadata["birth_date_text"] == "01.02.03"
        assert record.raw_metadata["comment"] == "test"

    @pytest.mark.parametrize("stored, sex", [("m", "M"), ("f", "F"), ("W", "F"), ("x", "U"), ("", "")])
    def test_sex(self, tmp_path, stored, sex):
        record = parse(make_xle(tmp_path / "rec.xle", ecg_like(), {"Sex": stored}))
        assert record.patient.sex == sex

    def test_empty_fields(self, tmp_path):
        record = parse(make_xle(tmp_path / "rec.xle", ecg_like()))
        p = record.patient
        assert p.first_name == p.last_name == p.patient_id == p.sex == ""

    def test_not_detected_without_labels(self, tmp_path):
        path = tmp_path / "other.xle"
        path.write_bytes(bytes(HEADER) + bytes(16 * 100))
        assert not XLEParser.can_parse(path, path.read_bytes()[:4096])

    def test_not_detected_with_partial_frame(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like())
        path.write_bytes(path.read_bytes() + b"\x00")
        assert not XLEParser.can_parse(path, path.read_bytes()[:4096])

    def test_xml_xle_not_detected(self, tmp_path):
        # .xle is also used by unrelated XML data files
        path = tmp_path / "data.xle"
        path.write_text('<?xml version="1.0"?><Body_xle></Body_xle>')
        assert not XLEParser.can_parse(path, path.read_bytes()[:4096])

    def test_header_only_is_corrupted(self, tmp_path):
        path = make_xle(tmp_path / "rec.xle", ecg_like())
        path.write_bytes(path.read_bytes()[:HEADER])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(CorruptedFileError):
                XLEParser().parse(path)
