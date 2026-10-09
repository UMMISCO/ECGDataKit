"""Tests for FileParser dispatch, auto-scaling, batch parsing and shared helpers."""

from __future__ import annotations

import gc
import json
import os
import time
import warnings
import weakref
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit import ECGRecord, Lead
import ecgdatakit
from ecgdatakit.exceptions import ChecksumWarning, CorruptedFileError, UnsupportedFormatError
from ecgdatakit.parsing import batch as batch_module
from ecgdatakit.parsing.batch import BatchParseWarning, parse_batch
from ecgdatakit.parsing.parsers.ishne_holter import ISHNEHolterParser
from ecgdatakit.parsing.helpers import (
    decode_text,
    fill_signal_summary,
    normalize_lead_label,
    unique_labels,
)
from ecgdatakit.parsing.parser import FileParser
from tests.conftest import create_ishne_binary


class TestHelpers:
    @pytest.mark.parametrize("raw, expected", [
        ("AVR", "aVR"), ("avl", "aVL"), ("ECG I", "I"), ("Lead V1", "V1"),
        ("ii", "II"), ("v6", "V6"), ("Resp", "Resp"), (" III ", "III"),
    ])
    def test_normalize_lead_label(self, raw, expected):
        assert normalize_lead_label(raw) == expected

    def test_unique_labels(self):
        assert unique_labels(["II", "II", "I", "II"]) == ["II", "II_2", "I", "II_3"]

    def test_unique_labels_never_reuse_an_input_label(self):
        assert unique_labels(["II", "II_2", "II"]) == ["II", "II_2", "II_3"]
        out = unique_labels(["a", "a", "a_2", "a_2"])
        assert len(set(out)) == 4 and out[0] == "a" and out[2] == "a_2"

    def test_decode_text_never_loses_bytes(self):
        assert decode_text("José".encode("utf-8")) == "José"
        assert decode_text("Müller".encode("latin-1")) == "Müller"
        assert decode_text(b"\xd5\xc5", "gb18030") == "张"

    def test_fill_signal_summary(self):
        rec = ECGRecord(leads=[
            Lead(label="I", samples=np.zeros(3), sampling_rate=500, resolution=0.0025, resolution_unit="mV"),
            Lead(label="II", samples=np.zeros(3), sampling_rate=500, resolution=2.5, resolution_unit="uV"),
        ])
        fill_signal_summary(rec)
        assert rec.recording.acquisition.signal.sampling_rate == 500
        assert rec.recording.acquisition.signal.resolution == pytest.approx(2.5)


class TestFileParser:
    def test_parsers_sorted_by_priority(self):
        prio = [p.PRIORITY for p in FileParser().parsers]
        assert prio == sorted(prio)

    def test_parser_warnings_point_at_caller(self, tmp_path: Path):
        data = bytearray(create_ishne_binary())
        data[28:35] = b"Changed"  # break the checksum
        p = tmp_path / "x.ecg"
        p.write_bytes(bytes(data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FileParser().parse(p)
        checksum = [x for x in w if issubclass(x.category, ChecksumWarning)]
        assert checksum and all(x.filename == __file__ for x in checksum)

    def test_filtered_warning_stays_silenced(self, tmp_path: Path):
        data = bytearray(create_ishne_binary())
        data[28:35] = b"Changed"
        p = tmp_path / "x.ecg"
        p.write_bytes(bytes(data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=ChecksumWarning)
            FileParser().parse(p)
        assert not any(issubclass(x.category, ChecksumWarning) for x in w)

    def test_decode_errors_become_corrupted_file_error(self, tmp_path: Path):
        p = tmp_path / "bad.xml"
        p.write_text("<AnnotatedECG><component>")
        with pytest.raises(CorruptedFileError):
            FileParser().parse(p)

    def test_auto_scale_false_keeps_raw_counts(self, ishne_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = FileParser().parse(ishne_file, auto_scale=False)
            mv = FileParser().parse(ishne_file)
        lead_raw, lead_mv = raw.leads[0], mv.leads[0]
        # fixture: 1000 nV/count, first sample 100 counts
        assert lead_raw.samples[0] == 100
        assert lead_mv.units == "mV" and not lead_mv.is_raw
        assert lead_mv.samples[0] == pytest.approx(0.1)

    def test_no_raw_warning_when_leads_are_physical(self, ishne_file: Path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FileParser().parse(ishne_file, auto_scale=False)
        assert not any("auto_scale=False" in str(x.message) for x in w)

    def test_auto_scale_nv_unit(self):
        rec = ECGRecord(leads=[Lead(label="I", samples=np.array([2000.0]), sampling_rate=500,
                                    resolution=1.0, resolution_unit="nV", is_raw=True)])
        out = FileParser._auto_scale(rec, "mV")
        assert out.leads[0].units == "mV"
        assert out.leads[0].samples[0] == pytest.approx(0.002)

    def test_auto_scale_warns_on_non_voltage_unit(self):
        rec = ECGRecord(leads=[Lead(label="Resp", samples=np.array([1.0]), sampling_rate=50,
                                    resolution=2.0, resolution_unit="Ohm", is_raw=True)])
        with pytest.warns(UserWarning, match="not in a voltage unit"):
            FileParser._auto_scale(rec, "mV")

    def test_filepath_recorded(self, ishne_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = FileParser().parse(ishne_file)
        assert rec.raw_metadata["filepath"] == str(ishne_file)


class TestJson:
    def test_nan_and_numpy_scalars(self):
        rec = ECGRecord(leads=[Lead(label="I", samples=np.array([1.0, np.nan]),
                                    sampling_rate=np.int64(500))])
        rec.patient.age = np.int64(40)
        out = json.loads(rec.to_json())
        assert out["leads"][0]["samples"] == [1.0, None]
        assert out["patient"]["age"] == 40


class TestBatch:
    def _files(self, tmp_path: Path) -> list[Path]:
        good = tmp_path / "a.ecg"
        good.write_bytes(create_ishne_binary())
        bad = tmp_path / "b.xml"
        bad.write_text("<AnnotatedECG><component>")
        good2 = tmp_path / "c.ecg"
        good2.write_bytes(create_ishne_binary())
        return [good, bad, good2]

    def test_raise_by_default(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError):
            list(parse_batch(self._files(tmp_path), max_workers=1))

    def test_warn_continues(self, tmp_path: Path):
        with pytest.warns(BatchParseWarning, match="b.xml"):
            records = list(parse_batch(self._files(tmp_path), max_workers=1, on_error="warn"))
        assert [Path(r.raw_metadata["filepath"]).name for r in records] == ["a.ecg", "c.ecg"]

    def test_forwards_auto_scale(self, tmp_path: Path):
        files = self._files(tmp_path)[:1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = next(parse_batch(files, max_workers=1, auto_scale=False))
        assert rec.leads[0].samples[0] == 100

    def test_invalid_on_error(self):
        with pytest.raises(ValueError):
            list(parse_batch([], on_error="skip"))


class TestFileParserErrors:
    def test_units_any_case(self, ishne_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = FileParser().parse(ishne_file, units="UV")
        assert rec.leads[0].units == "uV" and rec.leads[0].samples[0] == pytest.approx(100.0)

    def test_unknown_file_is_unsupported_and_value_error(self, tmp_path: Path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"nothing")
        with pytest.raises(UnsupportedFormatError):
            FileParser().parse(p)
        assert issubclass(UnsupportedFormatError, ValueError)

    def test_import_error_not_wrapped(self, ishne_file: Path, monkeypatch):
        def parse(self, path):
            raise ImportError("No module named 'pydicom'")
        monkeypatch.setattr(ISHNEHolterParser, "parse", parse)
        with pytest.raises(ImportError, match="pydicom"):
            FileParser().parse(ishne_file)

    def test_warnings_kept_when_parser_fails(self, ishne_file: Path, monkeypatch):
        def parse(self, path):
            warnings.warn("seen before the failure", UserWarning)
            raise CorruptedFileError("broken")
        monkeypatch.setattr(ISHNEHolterParser, "parse", parse)
        with pytest.warns(UserWarning, match="seen before the failure"):
            with pytest.raises(CorruptedFileError):
                FileParser().parse(ishne_file)

    def test_record_without_samples_raises(self, ishne_file: Path, monkeypatch):
        def parse(self, path):
            return ECGRecord(leads=[Lead(label="I", samples=np.empty(0), sampling_rate=500)])
        monkeypatch.setattr(ISHNEHolterParser, "parse", parse)
        with pytest.raises(CorruptedFileError, match="holds no samples"):
            FileParser().parse(ishne_file)


class TestAutoScaleInPlace:
    def test_scales_owned_arrays_in_place(self):
        samples = np.array([1000.0, 2000.0])
        rec = ECGRecord(leads=[Lead(label="I", samples=samples, sampling_rate=500,
                                    resolution=2.0, resolution_unit="uV", offset=10.0)])
        out = FileParser._auto_scale(rec, "mV")
        assert out.leads[0].samples is samples  # no second copy
        np.testing.assert_allclose(samples, [2.01, 4.01])
        assert out.leads[0].units == "mV" and not out.leads[0].is_raw

    def test_shared_and_foreign_arrays_are_copied(self):
        base = np.array([100.0, 200.0])
        frozen = np.array([5, 6], dtype=np.int16)
        rec = ECGRecord(leads=[
            Lead(label="I", samples=base, sampling_rate=500, resolution=1.0, resolution_unit="mV"),
            Lead(label="II", samples=base[:], sampling_rate=500, resolution=1.0, resolution_unit="mV"),
            Lead(label="III", samples=frozen, sampling_rate=500, resolution=1.0,
                 resolution_unit="uV"),
        ])
        out = FileParser._auto_scale(rec, "uV")
        np.testing.assert_allclose(out.leads[0].samples, [100000.0, 200000.0])
        np.testing.assert_allclose(out.leads[1].samples, [100000.0, 200000.0])
        np.testing.assert_allclose(out.leads[2].samples, [5.0, 6.0])
        assert frozen.tolist() == [5, 6]


_ORIGINAL_PARSE_SINGLE = batch_module._parse_single


def _missing_dependency_single(path, auto_scale, units):
    """Worker result for a format whose optional dependency is not installed."""
    if Path(path).name.startswith("dicom"):
        return None, [], ImportError("No module named 'pydicom'")
    return _ORIGINAL_PARSE_SINGLE(path, auto_scale, units)


def _crash_single(path, auto_scale, units):
    """Worker that dies on files named crash*, like a worker killed when out of memory."""
    if Path(path).name.startswith("crash"):
        os._exit(1)
    return _ORIGINAL_PARSE_SINGLE(path, auto_scale, units)


class TestBatchProcesses:
    def _many(self, tmp_path: Path, n: int) -> list[Path]:
        files = []
        for i in range(n):
            p = tmp_path / f"f{i:02d}.ecg"
            p.write_bytes(create_ishne_binary())
            files.append(p)
        return files

    def test_order_with_several_workers(self, tmp_path: Path):
        files = self._many(tmp_path, 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            names = [Path(r.raw_metadata["filepath"]).name
                     for r in parse_batch(files, max_workers=3)]
        assert names == [f.name for f in files]

    def test_records_freed_after_use(self, tmp_path: Path):
        files = self._many(tmp_path, 8)
        refs = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rec in parse_batch(files, max_workers=2):
                refs.append(weakref.ref(rec))
                del rec
                gc.collect()
        assert sum(r() is not None for r in refs) <= 1

    def test_close_stops_early(self, tmp_path: Path):
        files = self._many(tmp_path, 2) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen = parse_batch(files, max_workers=1)
            next(gen)
            start = time.perf_counter()
            gen.close()
        assert time.perf_counter() - start < 5

    def test_dead_worker_reported_and_rest_parsed(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(batch_module, "_parse_single", _crash_single)
        files = self._many(tmp_path, 4)
        crash = tmp_path / "crash.ecg"
        crash.write_bytes(create_ishne_binary())
        files.insert(2, crash)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            names = [Path(r.raw_metadata["filepath"]).name
                     for r in parse_batch(files, max_workers=2, on_error="warn")]
        assert names == [f.name for f in files if f != crash]
        batch_warnings = [x for x in w if issubclass(x.category, BatchParseWarning)]
        assert len(batch_warnings) == 1 and "crash.ecg" in str(batch_warnings[0].message)

    @pytest.mark.parametrize("executor", ["process", "serial"])
    @pytest.mark.parametrize("on_error", ["warn", "ignore"])
    def test_missing_dependency_always_raised(self, tmp_path: Path, monkeypatch,
                                              executor, on_error):
        monkeypatch.setattr(batch_module, "_parse_single", _missing_dependency_single)
        files = self._many(tmp_path, 3)
        dicom = tmp_path / "dicom.dcm"
        dicom.write_bytes(b"x")
        files.insert(1, dicom)
        names = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(ImportError, match="pydicom"):
                for rec in parse_batch(files, max_workers=2, on_error=on_error,
                                       executor=executor):
                    names.append(Path(rec.raw_metadata["filepath"]).name)
        assert names == [files[0].name]
        assert not any(issubclass(x.category, BatchParseWarning) for x in w)

    def test_serial_executor(self, tmp_path: Path):
        files = self._many(tmp_path, 2)
        bad = tmp_path / "b.xml"
        bad.write_text("<AnnotatedECG><component>")
        with pytest.warns(BatchParseWarning, match="b.xml"):
            records = list(parse_batch([files[0], bad, files[1]], executor="serial",
                                       on_error="warn"))
        assert len(records) == 2
        with pytest.raises(ValueError, match="executor"):
            list(parse_batch(files, executor="thread"))


def test_public_exports():
    assert ecgdatakit.BatchParseWarning is BatchParseWarning
    assert ecgdatakit.derive_is_raw(1.0, 0.0, "") is True
    assert {"BatchParseWarning", "derive_is_raw"} <= set(ecgdatakit.__all__)
