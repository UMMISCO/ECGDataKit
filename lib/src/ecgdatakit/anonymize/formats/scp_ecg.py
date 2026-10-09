"""SCP-ECG anonymization.

Section 1 tags replaced (EN 1064 / ISO 11073-91064): 0 last name, 3 second
last name and 2 patient ID become the patient code, 1 first name is
emptied, 31 ECG sequence number becomes the ECG code. Text tags 13
(referral), 16 to 23 (institutions, departments, physicians), 30 (free
text) and 35 (medical history) are searched for names. Text is written in
the Section 1 language encoding (tag 14).

Section 1 is rebuilt. Sections stored after it move, so the Section 0
pointers and the record size are updated. Each CRC (record, Section 0,
Section 1) is recomputed only when the stored one was valid.
"""

from __future__ import annotations

import binascii
import struct
from pathlib import Path

from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite
from ecgdatakit.parsing.parsers.scp_ecg import _FIXED_TAG_SIZES, _LANGUAGE_CODECS, _WIDE_CODECS

_SCP_HEADER = 16
# Section 1 tags: 0 last name, 1 first name, 2 patient ID, 3 second last
# name, 31 ECG sequence number. Text tags searched for names: referral (13),
# institutions, departments and physicians (16-23), free text (30),
# medical history (35).
_SCP_IDENTITY = {0: "last_names", 1: "first_names", 2: "patient_ids", 3: "last_names",
                 31: "ecg_ids"}
_SCP_FREE_TEXT = {13, 16, 17, 18, 19, 20, 21, 22, 23, 30, 35}


def _crc(data: bytes) -> int:
    return binascii.crc_hqx(data, 0xFFFF)


class _SCP:
    """Section layout of an SCP-ECG record."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.size = struct.unpack_from("<I", data, 2)[0]
        sec0_len = struct.unpack_from("<I", data, 10)[0]
        self.pointers: list[tuple[int, int, int, int]] = []  # (pos, id, length, index)
        pos = 6 + _SCP_HEADER
        while pos + 10 <= 6 + sec0_len:
            sid, length, index = struct.unpack_from("<HII", data, pos)
            self.pointers.append((pos, sid, length, index))
            pos += 10
        self.sec0_len = sec0_len

    def section(self, sid: int) -> tuple[int, int] | None:
        for _, s, length, index in self.pointers:
            if s == sid and length and index:
                return index - 1, length
        return None

    def encoding(self, tags: list[tuple[int, bytes]]) -> str:
        for tag, value in tags:
            if tag == 14 and len(value) > 16 and value[16] & 0x01:
                return _LANGUAGE_CODECS.get(value[16], "latin-1")
        return "latin-1"


def _scp_tags(body: bytes, fixed: bool = False) -> list[tuple[int, bytes, bytes]] | None:
    """Split Section 1 into (tag, value, stored bytes); None if the walk fails.

    With *fixed*, tags of fixed size use the size of the standard, for files
    that store a wrong length (as the parser does).
    """
    tags = []
    pos = 0
    while pos + 3 <= len(body):
        start = pos
        tag, length = body[pos], struct.unpack_from("<H", body, pos + 1)[0]
        if fixed and tag in _FIXED_TAG_SIZES:
            length = _FIXED_TAG_SIZES[tag]
        pos += 3
        if tag == 255:
            tags.append((255, b"", body[start:pos]))
            return tags
        if pos + length > len(body):
            return None
        tags.append((tag, body[pos:pos + length], body[start:pos + length]))
        pos += length
    return tags


def _scp_text(value: bytes, encoding: str) -> str:
    if encoding in _WIDE_CODECS:
        end = next((i for i in range(0, len(value) - 1, 2) if value[i:i + 2] == b"\x00\x00"), None)
        value = value[:end] if end is not None else value[: len(value) // 2 * 2]
    else:
        value = value.split(b"\x00", 1)[0]
    try:
        return value.decode(encoding).strip()
    except (UnicodeDecodeError, LookupError):
        return value.decode("latin-1").strip()


def _scp_encode(text: str, encoding: str) -> bytes:
    terminator = b"\x00\x00" if encoding in _WIDE_CODECS else b"\x00"
    return text.encode(encoding, errors="replace") + terminator


class SCPHandler(Handler):
    """SCP-ECG Section 1 (patient data).

    Section 1 is rebuilt with the new values. Sections stored after it move,
    so the Section 0 pointers, the record size and the CRCs are updated. A
    CRC is recomputed only when the original one was valid.
    """

    parser = "SCPECGParser"

    def _read(self, data: bytes):
        scp = _SCP(data)
        sec1 = scp.section(1)
        if sec1 is None:
            return scp, None, None, "latin-1"
        offset, length = sec1
        body = data[offset + _SCP_HEADER: offset + length]
        tags = _scp_tags(body)
        if tags is None:
            tags = _scp_tags(body, fixed=True)
        return scp, sec1, tags, scp.encoding([(t, v) for t, v, _ in tags or []])

    def identity(self, path: Path) -> Identity:
        data = Path(path).read_bytes()
        _, _, tags, encoding = self._read(data)
        identity = Identity()
        for tag, value, _ in tags or []:
            kind = _SCP_IDENTITY.get(tag)
            if kind:
                identity.add(kind, _scp_text(value, encoding))
        return identity

    def rewrite(self, path, out_path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        data = Path(path).read_bytes()
        scp, sec1, tags, encoding = self._read(data)
        result = Rewrite()
        if sec1 is None:
            result.outputs[Path(out_path)] = data  # no patient section
            return result
        if tags is None:
            raise ValueError("SCP-ECG Section 1 cannot be split into tags")
        offset, length = sec1
        # Tags not changed keep their stored bytes, wrong lengths included
        body = b""
        for tag, value, stored in tags:
            kind = _SCP_IDENTITY.get(tag)
            text = _scp_text(value, encoding) if tag != 255 else ""
            new_value = None
            if kind and text:
                new = {"first_names": ""}.get(kind, codes.ecg if kind == "ecg_ids" else codes.patient)
                if kind == "ecg_ids":
                    result.ecg_written.append(new)
                new_value = _scp_encode(new, encoding)
            elif tag in _SCP_FREE_TEXT and text and replacer:
                new = replacer.replace(text)
                if new != text:
                    new_value = _scp_encode(new, encoding)
            body += stored if new_value is None else struct.pack("<BH", tag, len(new_value)) + new_value
        if len(body) % 2:
            body += b"\x00"  # sections have an even length
        header = bytearray(data[offset: offset + _SCP_HEADER])
        new_len = _SCP_HEADER + len(body)
        struct.pack_into("<I", header, 4, new_len)
        sec1_crc_ok = _crc(data[offset + 2: offset + length]) == struct.unpack_from("<H", data, offset)[0]
        section = bytes(header) + body
        if sec1_crc_ok:
            section = struct.pack("<H", _crc(section[2:])) + section[2:]
        delta = new_len - length

        out = bytearray(data[:offset]) + section + data[offset + length:]
        # Pointers of sections stored after Section 1 move by delta
        for pos, sid, plen, index in scp.pointers:
            if sid == 1:
                struct.pack_into("<I", out, pos + 2, new_len)
            elif plen and index - 1 > offset:
                struct.pack_into("<I", out, pos + 6, index + delta)
        sec0_crc_ok = _crc(data[8: 6 + scp.sec0_len]) == struct.unpack_from("<H", data, 6)[0]
        if sec0_crc_ok:
            struct.pack_into("<H", out, 6, _crc(bytes(out[8: 6 + scp.sec0_len])))
        record_crc_ok = _crc(data[2: scp.size]) == struct.unpack_from("<H", data, 0)[0]
        struct.pack_into("<I", out, 2, scp.size + delta)
        if record_crc_ok:
            struct.pack_into("<H", out, 0, _crc(bytes(out[2: scp.size + delta])))
        result.outputs[Path(out_path)] = bytes(out)
        return result

    def free_text(self, data: bytes) -> list[str]:
        _, _, tags, encoding = self._read(data)
        return [_scp_text(v, encoding) for t, v, _ in tags or []
                         if t in _SCP_FREE_TEXT or t in _SCP_IDENTITY]
