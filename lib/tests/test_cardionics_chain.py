"""Tests for the Cardionics Holter chain parser (.chn + ISHNE chunks)."""

from __future__ import annotations

import binascii
import struct
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.cardionics_chain import CardionicsChainParser
from tests.conftest import create_ishne_binary

RATE = 10
START = datetime(2009, 9, 17, 8, 21, 6)
PREFIX = "090917082106ABC-DE"

# End record of a recorder chain, as written: name "End", then bytes that are not data
END_RECORD = bytes.fromhex(
    "840300009f07000011000900d9070800350031001c03456e6400f4773a8af477"
    "f1abe5772000000010000000000000000000000000000000f03f1d0000001400"
    "d4e1120060e01a0074f91200f088f977781cf477ffffffff3a8af4778e05cd7c"
    "0000140000000000100000003f170e7740e0dc7c1000000010000000a9160e77"
    "100000000f0000000100000000000000186b0e776ae4120001000000016b0e77"
    "6ae412000100000020ea1200000000000000008038ea12000000008000000000"
    "00000000090000002d0039003000300030003000300030003000300030003000"
    "300030003000300030003000ace41200c01ddb020010000000000000b4bff477"
    "bdabe57754020000000000000000000000000000"
)

# End record of a second recorder chain, 13 s after the start of its last chunk
END_RECORD_2 = bytes.fromhex(
    "84030000a91800000b000a00db070e0039003200a903456e640059000d000000"
    "fe01000039863a7e00000000c07459000b00000000020000dce312004d656173"
    "757265732e2e2e0020e412000821240000000000f76a927c94e41200fe6a927c"
    "f8fbfd7f08212400000000001400000001000000cce412000000000000000000"
    "18e51200ca9e73010000000000000000000000000000000018000a0200fcfd7f"
    "00000000780114000000000068e4120028c23a7e8201070008000000e2020a00"
    "0000000001000000b0e41200d5c13a7e8201070008000000e2020a00e9c13a7e"
    "18e51200ca9e73010000000000000000ca9e73010000000080e412005ce51200"
    "f270ef779d70ef77550001011000000010000000"
)


def chain_record(nominal: int, elapsed: int, written: tuple, name: str) -> bytes:
    """One 276-byte record: lengths, write time (day first) and chunk name."""
    data = struct.pack("<II7H", nominal, elapsed, *written) + name.encode("latin-1")
    return data + bytes(276 - len(data))


def recorder_chain(prefix: str = PREFIX) -> bytes:
    """The chain of a 1951 s recording: chunks starting at 900 s and 1800 s, then End."""
    return (chain_record(900, 900, (17, 9, 2009, 8, 36, 18, 515), f"{prefix}_1.ecg")
            + chain_record(900, 1800, (17, 9, 2009, 8, 51, 18, 453), f"{prefix}_2.ecg")
            + END_RECORD)


def ishne_chunk(first: int, samples: int, start: datetime, rate: int = RATE,
                patient_id: bytes = b"P0001") -> bytes:
    """ISHNE file of two leads: lead I counts on from *first*, lead II is its negative."""
    head = bytearray(create_ishne_binary(nleads=2, sr=rate, samples_per_lead=1)[:522])
    struct.pack_into("<i", head, 14, samples)
    struct.pack_into("<hhh", head, 138, start.day, start.month, start.year)
    struct.pack_into("<hhh", head, 150, start.hour, start.minute, start.second)
    head[108:128] = patient_id.ljust(20, b"\x00")
    struct.pack_into("<H", head, 8, binascii.crc_hqx(bytes(head[10:522]), 0xFFFF))
    ramp = (first + np.arange(samples)) % 30000
    frames = np.column_stack([ramp, -ramp]).astype("<i2")
    return bytes(head) + frames.tobytes()


def chunk_names(count: int, prefix: str = PREFIX) -> list[str]:
    """The first chunk has the name of the chain, the next ones a number."""
    return [f"{prefix}.ecg"] + [f"{prefix}_{i}.ecg" for i in range(1, count)]


def write_recording(folder: Path, seconds=(900, 900, 151), names=None,
                    chain: bytes | None = None, chunk_starts: bool = False,
                    prefix: str = PREFIX) -> Path:
    """Chunk files of consecutive seconds and their chain.

    Each chunk header holds the start of the recording, as the recorder
    writes it, or the start of the chunk with *chunk_starts*.
    """
    names = names or chunk_names(len(seconds), prefix)
    folder.mkdir(parents=True, exist_ok=True)
    first = 0
    for name, length in zip(names, seconds):
        samples = round(length * RATE)
        start = START + timedelta(seconds=first / RATE) if chunk_starts else START
        (folder / name).write_bytes(ishne_chunk(first, samples, start))
        first += samples
    path = folder / f"{prefix}.chn"
    path.write_bytes(recorder_chain() if chain is None else chain)
    return path


def parse(path: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = CardionicsChainParser().parse(path)
    return record, [str(w.message) for w in caught]


def expected_ramp(seconds: float) -> np.ndarray:
    return (np.arange(round(seconds * RATE)) % 30000).astype(np.float64)


class TestRecorderChain:
    def test_record_bytes_match_the_recorder_layout(self):
        data = recorder_chain()
        assert len(data) == 3 * 276
        assert data[22:46] == f"{PREFIX}_1.ecg".encode()
        assert struct.unpack_from("<II", END_RECORD) == (900, 1951)

    def test_chunks_joined_in_chain_order(self, tmp_path: Path):
        record, other = parse(write_recording(tmp_path))
        assert other == []
        assert [lead.label for lead in record.leads] == ["I", "II"]
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(1951))
        np.testing.assert_array_equal(record.leads[1].samples, -expected_ramp(1951))

    def test_times_and_duration(self, tmp_path: Path):
        record, _ = parse(write_recording(tmp_path))
        assert record.recording.date == START
        assert record.recording.duration == timedelta(seconds=1951)
        assert record.leads[0].sampling_rate == RATE

    def test_chunk_metadata(self, tmp_path: Path):
        path = write_recording(tmp_path)
        record, _ = parse(path)
        raw = record.raw_metadata
        assert raw["filepath"] == str(path)
        assert raw["end_record"] is True
        assert raw["end_elapsed_s"] == 1951
        assert raw["end_written_at"] == datetime(2009, 9, 17, 8, 53, 49, 796000)
        chunks = raw["chunks"]
        assert [c["name"] for c in chunks] == chunk_names(3)
        assert [c["listed"] for c in chunks] == [False, True, True]
        assert [c["first_sample"] for c in chunks] == [0, 9000, 18000]
        assert [c["samples"] for c in chunks] == [9000, 9000, 1510]
        assert [c["start_s"] for c in chunks] == [0, 900, 1800]
        assert [c["end_s"] for c in chunks] == [900, 1800, 1951]
        assert [c["nominal_s"] for c in chunks] == [None, 900, 900]
        assert [c["start"] for c in chunks] == [START] * 3
        assert [c["written_at"] for c in chunks] == [
            None,
            datetime(2009, 9, 17, 8, 36, 18, 515000),
            datetime(2009, 9, 17, 8, 51, 18, 453000),
        ]
        assert chunks[0]["ishne"]["checksum_valid"] is True

    def test_ecg_files(self, tmp_path: Path):
        record, _ = parse(write_recording(tmp_path))
        assert [(f.name, f.start, f.end, f.samples) for f in record.ecg_files] == [
            (f"{PREFIX}.ecg", START, START + timedelta(seconds=900), 9000),
            (f"{PREFIX}_1.ecg", START + timedelta(seconds=900), START + timedelta(seconds=1800), 9000),
            (f"{PREFIX}_2.ecg", START + timedelta(seconds=1800), START + timedelta(seconds=1951), 1510),
        ]
        assert sum(f.samples for f in record.ecg_files) == len(record.leads[0].samples)
        assert record.to_dict(include_samples=False)["ecg_files"][1] == {
            "name": f"{PREFIX}_1.ecg",
            "start": "2009-09-17T08:36:06",
            "end": "2009-09-17T08:51:06",
            "samples": 9000,
        }

    def test_second_recorder_chain(self, tmp_path: Path):
        prefix = "111011131235ABC-DE"
        written = [(13, 27, 37, 687), (13, 42, 37, 812), (13, 57, 37, 546), (14, 12, 37, 437),
                   (14, 27, 37, 390), (14, 42, 37, 296), (14, 57, 37, 250)]
        chain = b"".join(chain_record(900, 900 * (i + 1), (11, 10, 2011, *w), f"{prefix}_{i + 1}.ecg")
                         for i, w in enumerate(written)) + END_RECORD_2
        assert len(chain) == 8 * 276
        record, other = parse(write_recording(tmp_path, seconds=(900,) * 7 + (13,),
                                              chain=chain, prefix=prefix))
        assert other == []
        chunks = record.raw_metadata["chunks"]
        assert [c["name"] for c in chunks] == chunk_names(8, prefix)
        assert [c["start_s"] for c in chunks] == [900 * i for i in range(8)]
        assert chunks[-1]["end_s"] == 6313
        assert chunks[-1]["samples"] == 130
        assert record.raw_metadata["end_written_at"] == datetime(2011, 10, 11, 14, 57, 50, 937000)
        assert record.recording.duration == timedelta(seconds=6313)
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(6313))

    def test_third_recorder_chain(self, tmp_path: Path):
        prefix = "110801130101ABC-DE"
        written = [(13, 16, 8, 843), (13, 31, 8, 750), (13, 46, 8, 671), (14, 1, 8, 625),
                   (14, 16, 8, 531), (14, 31, 8, 484), (14, 46, 8, 375)]
        chain = b"".join(chain_record(900, 900 * (i + 1), (1, 8, 2011, *w), f"{prefix}_{i + 1}.ecg")
                         for i, w in enumerate(written))
        chain += chain_record(900, 6652, (1, 8, 2011, 14, 52, 0, 640), "End")
        record, other = parse(write_recording(tmp_path, seconds=(900,) * 7 + (352.3,),
                                              chain=chain, prefix=prefix))
        assert other == []
        chunks = record.raw_metadata["chunks"]
        assert [c["samples"] for c in chunks] == [9000] * 7 + [3523]
        assert [c["end_s"] for c in chunks] == [900 * i for i in range(1, 8)] + [6652]
        assert record.recording.duration == timedelta(seconds=6652.3)
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(6652.3))

    def test_file_parser(self, tmp_path: Path):
        path = write_recording(tmp_path)
        assert FileParser.detect_format(path)["parser"] == "CardionicsChainParser"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            record = FileParser.parse(path)
        assert record.file_format.name == "Cardionics Holter chain"
        assert record.source_format == "cardionics_chain"
        assert record.leads[0].units == "mV"
        np.testing.assert_allclose(record.leads[0].samples, expected_ramp(1951) / 1000)

    def test_chunk_read_alone_is_unchanged(self, tmp_path: Path):
        write_recording(tmp_path)
        record = FileParser.parse(tmp_path / f"{PREFIX}_1.ecg", auto_scale=False)
        assert record.file_format.parser == "ISHNEHolterParser"
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(1951)[9000:18000])


class TestChunkNames:
    def test_order_is_the_chain_order_not_the_name_order(self, tmp_path: Path):
        names = [f"{PREFIX}.ecg", "rec_9.ecg", "rec_10.ecg"]
        chain = (chain_record(900, 900, (1, 1, 2020, 0, 15, 0, 0), names[1])
                 + chain_record(900, 1800, (1, 1, 2020, 0, 30, 0, 0), names[2])
                 + END_RECORD)
        record, other = parse(write_recording(tmp_path, names=names, chain=chain))
        assert other == []
        assert [c["name"] for c in record.raw_metadata["chunks"]] == names
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(1951))

    def test_stored_folder_is_ignored(self, tmp_path: Path):
        chain = (chain_record(900, 900, (1, 1, 2020, 0, 15, 0, 0), f"C:\\DATA\\{PREFIX}_1.ecg")
                 + chain_record(900, 1800, (1, 1, 2020, 0, 30, 0, 0), f"D:/X/{PREFIX}_2.ecg")
                 + END_RECORD)
        record, _ = parse(write_recording(tmp_path, chain=chain))
        assert record.raw_metadata["chunks"][1]["name"] == f"{PREFIX}_1.ecg"
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(1951))

    def test_file_name_case_differs_from_chain(self, tmp_path: Path):
        names = [name.replace(".ecg", ".ECG") for name in chunk_names(3)]
        record, _ = parse(write_recording(tmp_path, names=names))
        found = [Path(c["filepath"]).name.lower() for c in record.raw_metadata["chunks"]]
        assert found == [name.lower() for name in names]
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(1951))

    def test_chain_holding_only_end(self, tmp_path: Path):
        end = chain_record(900, 600, (17, 9, 2009, 8, 31, 6, 0), "End")
        record, other = parse(write_recording(tmp_path, seconds=(600,), chain=end))
        assert other == []
        assert [c["name"] for c in record.raw_metadata["chunks"]] == [f"{PREFIX}.ecg"]
        assert record.recording.duration == timedelta(seconds=600)


class TestTimeline:
    def test_chunk_headers_with_their_own_start(self, tmp_path: Path):
        record, other = parse(write_recording(tmp_path, chunk_starts=True))
        assert other == []
        assert [c["start"] for c in record.raw_metadata["chunks"]] == [
            START + timedelta(seconds=s) for s in (0, 900, 1800)]

    def test_start_time_mismatch_warns(self, tmp_path: Path):
        path = write_recording(tmp_path)
        late = START + timedelta(seconds=960)
        (tmp_path / f"{PREFIX}_1.ecg").write_bytes(ishne_chunk(9000, 9000, late))
        record, other = parse(path)
        assert len(other) == 1
        assert (f"{PREFIX}_1.ecg starts at 08:37:06, expected 08:21:06 or 08:36:06"
                in other[0])
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(1951))

    def test_chunk_lengths_within_one_second(self, tmp_path: Path):
        record, other = parse(write_recording(tmp_path, seconds=(900.5, 899.6, 151)))
        assert other == []
        assert record.recording.duration == timedelta(seconds=1951.1)

    def test_total_within_one_second(self, tmp_path: Path):
        record, other = parse(write_recording(tmp_path, seconds=(900, 900, 151.9)))
        assert other == []
        assert record.recording.duration == timedelta(seconds=1951.9)

    def test_no_end_record_warns(self, tmp_path: Path):
        chain = recorder_chain()[:2 * 276]
        record, other = parse(write_recording(tmp_path, seconds=(900, 900, 40), chain=chain))
        assert other == [(f"Cardionics chain: {PREFIX}.chn has no End record, "
                          "the recording may be incomplete")]
        assert record.raw_metadata["end_record"] is False
        assert record.raw_metadata["end_elapsed_s"] is None
        assert record.raw_metadata["chunks"][-1]["end_s"] is None
        assert record.ecg_files[-1].end == START + timedelta(seconds=1840)
        assert record.recording.duration == timedelta(seconds=1840)

    def test_trailing_bytes_warn(self, tmp_path: Path):
        _, other = parse(write_recording(tmp_path, chain=recorder_chain() + b"\x00" * 10))
        assert other == [(f"Cardionics chain: {PREFIX}.chn ends with 10 bytes that do not "
                          "form a whole record, ignored")]

    def test_records_after_end_ignored(self, tmp_path: Path):
        extra = chain_record(900, 2700, (17, 9, 2009, 9, 0, 0, 0), f"{PREFIX}_3.ecg")
        record, other = parse(write_recording(tmp_path, chain=recorder_chain() + extra))
        assert other == [(f"Cardionics chain: 1 record(s) after the End record of "
                          f"{PREFIX}.chn ignored")]
        assert len(record.raw_metadata["chunks"]) == 3

    def test_chunk_warnings_name_the_chunk(self, tmp_path: Path):
        path = write_recording(tmp_path)
        chunk = tmp_path / f"{PREFIX}_1.ecg"
        chunk.write_bytes(chunk.read_bytes() + b"\x00")
        _, other = parse(path)
        assert len(other) == 1
        assert other[0].startswith(f"Cardionics chain: chunk {PREFIX}_1.ecg of {PREFIX}.chn: "
                                   "ISHNE: odd number of data bytes")

    def test_chunk_comments_kept(self, tmp_path: Path):
        path = write_recording(tmp_path)
        chunk = tmp_path / f"{PREFIX}_1.ecg"
        data = bytearray(chunk.read_bytes())
        note = b"lead II noisy"
        data[522:522] = note
        struct.pack_into("<i", data, 10, len(note))
        struct.pack_into("<i", data, 22, 522 + len(note))
        struct.pack_into("<H", data, 8, binascii.crc_hqx(bytes(data[10:522 + len(note)]), 0xFFFF))
        chunk.write_bytes(bytes(data))
        record, other = parse(path)
        assert other == []
        assert record.raw_metadata["chunks"][1]["annotations"] == {"comments": "lead II noisy"}
        np.testing.assert_array_equal(record.leads[0].samples, expected_ramp(1951))


class TestErrors:
    def test_missing_first_chunk(self, tmp_path: Path):
        path = write_recording(tmp_path)
        (tmp_path / f"{PREFIX}.ecg").unlink()
        with pytest.raises(CorruptedFileError, match=f"chunk {PREFIX}.ecg of .* is missing"):
            parse(path)

    def test_missing_listed_chunk(self, tmp_path: Path):
        path = write_recording(tmp_path)
        (tmp_path / f"{PREFIX}_2.ecg").unlink()
        with pytest.raises(CorruptedFileError, match=f"chunk {PREFIX}_2.ecg of .* is missing"):
            parse(path)

    def test_short_chunk(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError,
                           match=f"chunk lengths of {PREFIX}.chn do not match its times: "
                                 f"{PREFIX}_1.ecg lasts 890000 ms, chain says 900000 ms$"):
            parse(write_recording(tmp_path, seconds=(900, 890, 151)))

    def test_short_first_chunk(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError,
                           match=f"{PREFIX}.ecg lasts 898000 ms, chain says 900000 ms"):
            parse(write_recording(tmp_path, seconds=(898, 900, 151)))

    def test_every_wrong_chunk_named(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError,
                           match=f"{PREFIX}_1.ecg lasts 890000 ms, chain says 900000 ms, "
                                 f"{PREFIX}_2.ecg lasts 161000 ms, chain says 151000 ms"):
            parse(write_recording(tmp_path, seconds=(900, 890, 161)))

    def test_short_last_chunk(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError,
                           match=f"{PREFIX}_2.ecg lasts 100000 ms, chain says 151000 ms"):
            parse(write_recording(tmp_path, seconds=(900, 900, 100)))

    def test_chunk_over_one_second(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError,
                           match=f"{PREFIX}_2.ecg lasts 152200 ms, chain says 151000 ms"):
            parse(write_recording(tmp_path, seconds=(900, 900, 152.2)))

    def test_total_over_one_second(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError,
                           match=f"samples of {PREFIX}.chn last 1953700 ms, "
                                 "its chunks last 1951000 ms"):
            parse(write_recording(tmp_path, seconds=(900.9, 900.9, 151.9)))

    def test_times_must_increase(self, tmp_path: Path):
        chain = (chain_record(900, 1800, (1, 1, 2020, 0, 15, 0, 0), f"{PREFIX}_1.ecg")
                 + chain_record(900, 900, (1, 1, 2020, 0, 30, 0, 0), f"{PREFIX}_2.ecg")
                 + END_RECORD)
        with pytest.raises(CorruptedFileError, match="do not increase"):
            parse(write_recording(tmp_path, chain=chain))

    def test_end_before_last_chunk_start(self, tmp_path: Path):
        end = bytearray(END_RECORD)
        struct.pack_into("<I", end, 4, 1700)
        chain = recorder_chain()[:2 * 276] + bytes(end)
        with pytest.raises(CorruptedFileError, match=r"do not increase: \[900, 1800, 1700\]"):
            parse(write_recording(tmp_path, chain=chain))

    def test_chunk_listed_twice(self, tmp_path: Path):
        chain = (chain_record(900, 900, (1, 1, 2020, 0, 15, 0, 0), f"{PREFIX}_1.ecg")
                 + chain_record(900, 1800, (1, 1, 2020, 0, 30, 0, 0), f"{PREFIX}_1.ecg")
                 + END_RECORD)
        with pytest.raises(CorruptedFileError, match="lists a chunk twice"):
            parse(write_recording(tmp_path, chain=chain))

    def test_first_chunk_listed(self, tmp_path: Path):
        chain = (chain_record(900, 900, (1, 1, 2020, 0, 15, 0, 0), f"{PREFIX}.ecg")
                 + END_RECORD)
        with pytest.raises(CorruptedFileError, match="lists a chunk twice"):
            parse(write_recording(tmp_path, chain=chain))

    def test_listed_file_is_not_ishne(self, tmp_path: Path):
        chain = chain_record(900, 900, (1, 1, 2020, 0, 15, 0, 0), "notes.txt") + END_RECORD
        with pytest.raises(CorruptedFileError, match="not an ISHNE .ecg file"):
            parse(write_recording(tmp_path, chain=chain))

    def test_sampling_rate_differs(self, tmp_path: Path):
        path = write_recording(tmp_path)
        (tmp_path / f"{PREFIX}_1.ecg").write_bytes(ishne_chunk(0, 18000, START, rate=20))
        with pytest.raises(CorruptedFileError, match="sampling_rate 20"):
            parse(path)

    def test_other_patient(self, tmp_path: Path):
        path = write_recording(tmp_path)
        (tmp_path / f"{PREFIX}_1.ecg").write_bytes(
            ishne_chunk(9000, 9000, START, patient_id=b"P0002"))
        with pytest.raises(CorruptedFileError, match="another patient"):
            parse(path)

    def test_chunk_error_names_the_chunk(self, tmp_path: Path):
        path = write_recording(tmp_path)
        (tmp_path / f"{PREFIX}_1.ecg").write_bytes(b"NOTISHNE" + bytes(600))
        with pytest.raises(CorruptedFileError, match=f"chunk {PREFIX}_1.ecg of {PREFIX}.chn: "
                                                     "Bad ISHNE magic"):
            parse(path)


class TestDetection:
    def test_detects_chain(self, tmp_path: Path):
        path = write_recording(tmp_path)
        assert CardionicsChainParser.can_parse(path, path.read_bytes()[:4096]) is True

    def test_detects_chain_holding_only_end(self, tmp_path: Path):
        path = tmp_path / "x.chn"
        path.write_bytes(END_RECORD)
        assert CardionicsChainParser.can_parse(path, END_RECORD) is True

    def test_rejects_other_extension(self, tmp_path: Path):
        path = tmp_path / "x.dat"
        path.write_bytes(recorder_chain())
        assert CardionicsChainParser.can_parse(path, recorder_chain()) is False

    def test_rejects_invalid_date(self, tmp_path: Path):
        data = chain_record(900, 900, (0, 13, 2009, 25, 0, 0, 0), "rec_1.ecg")
        path = tmp_path / "x.chn"
        path.write_bytes(data)
        assert CardionicsChainParser.can_parse(path, data) is False

    def test_rejects_short_file(self, tmp_path: Path):
        path = tmp_path / "x.chn"
        path.write_bytes(END_RECORD[:100])
        assert CardionicsChainParser.can_parse(path, END_RECORD[:100]) is False

    def test_rejects_ishne_file(self, tmp_path: Path):
        data = create_ishne_binary()
        path = tmp_path / "x.chn"
        path.write_bytes(data)
        assert CardionicsChainParser.can_parse(path, data[:4096]) is False
