"""MFER anonymization.

Tags replaced (ISO 22077-1): MWF_PNM (0x81, patient name) and the first
component of MWF_PID (0x82, patient ID) become the patient code, MWF_UID
(0x87) the ECG code. MWF_NTE (0x16, comment) and MWF_MSS (0x86, message)
are searched for names. Text uses the MWF_TXC character set.

Only top-level tags with a definite length are rewritten, with their length
encoded again. Every other byte, including channel attribute blocks and
waveform data, is copied unchanged.
"""

from __future__ import annotations

from pathlib import Path

from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite
from ecgdatakit.parsing.parsers.mfer import (
    MWF_ATT,
    MWF_END,
    MWF_MSS,
    MWF_NTE,
    MWF_PID,
    MWF_PNM,
    MWF_SET,
    MWF_TXC,
    MWF_UID,
    MWF_ZRO,
    _read_length,
    _read_tag,
    _skip_indefinite,
    _text_codec,
)

_MFER_FREE_TEXT = {MWF_NTE, MWF_MSS}


def _mfer_length(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


class MFERHandler(Handler):
    """MFER patient tags: name (0x81), ID (0x82) and the ECG UID (0x87).

    Only top-level tags are rewritten, with their length re-encoded; every
    other byte is copied as is.
    """

    parser = "MFERParser"

    @staticmethod
    def _items(data: bytes):
        """Yield (tag, header_start, value_start, value_end, next_pos) at top level."""
        pos, end = 0, len(data)
        while pos < end:
            if data[pos] == MWF_ZRO:
                pos += 1
                continue
            start = pos
            tag, _, pos = _read_tag(data, pos, end)
            if tag == MWF_END:
                yield tag, start, pos, pos, pos
                return
            length, pos = _read_length(data, pos, end)
            if length is None:
                value_end, next_pos = _skip_indefinite(data, pos, end)
            else:
                value_end = next_pos = pos + length
            yield tag, start, pos, value_end, next_pos
            pos = next_pos

    @staticmethod
    def _encoding(data: bytes) -> str:
        for tag, _, vs, ve, _ in MFERHandler._items(data):
            if tag == MWF_TXC:
                name = data[vs:ve].decode("ascii", errors="replace").strip("\x00 ")
                return _text_codec(name) or "latin-1"
        return "latin-1"

    @staticmethod
    def _text(value: bytes, encoding: str) -> str:
        try:
            return value.decode(encoding).strip("\x00 ")
        except (UnicodeDecodeError, LookupError):
            return value.decode("latin-1").strip("\x00 ")

    def identity(self, path: Path) -> Identity:
        data = Path(path).read_bytes()
        encoding = self._encoding(data)
        identity = Identity()
        for tag, _, vs, ve, _ in self._items(data):
            text = self._text(data[vs:ve], encoding)
            if tag == MWF_PNM:
                parts = [p.strip() for p in text.split("^")]
                identity.add("last_names", parts[0])
                if len(parts) > 2:
                    identity.add("first_names", parts[2])
                if len(parts) > 4:
                    identity.add("first_names", parts[4])
            elif tag == MWF_PID:
                identity.add("patient_ids", text.split("^")[0])
            elif tag == MWF_UID:
                identity.add("ecg_ids", text)
        return identity

    def rewrite(self, path, out_path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        data = Path(path).read_bytes()
        encoding = self._encoding(data)
        result = Rewrite()
        out = bytearray()
        last = 0
        for tag, start, vs, ve, nxt in self._items(data):
            if tag in (MWF_ATT, MWF_SET):
                continue  # copied as is
            text = self._text(data[vs:ve], encoding)
            new = None
            if tag == MWF_PNM and text:
                new = codes.patient
            elif tag == MWF_PID and text:
                parts = text.split("^")
                parts[0] = codes.patient
                new = "^".join(parts)
            elif tag == MWF_UID and text:
                new = codes.ecg
                result.ecg_written.append(new)
            elif tag in _MFER_FREE_TEXT and text and replacer:
                replaced = replacer.replace(text)
                new = replaced if replaced != text else None
            if new is None or data[vs - 1:vs] == b"\x80":
                continue
            value = new.encode(encoding, errors="replace")
            out += data[last:start] + bytes([tag]) + _mfer_length(len(value)) + value
            last = nxt
        out += data[last:]
        result.outputs[Path(out_path)] = bytes(out)
        return result

    def free_text(self, data: bytes) -> list[str]:
        encoding = self._encoding(data)
        return [self._text(data[vs:ve], encoding) for tag, _, vs, ve, _ in self._items(data)
                         if tag in _MFER_FREE_TEXT or tag in (MWF_PNM, MWF_PID, MWF_UID)]
