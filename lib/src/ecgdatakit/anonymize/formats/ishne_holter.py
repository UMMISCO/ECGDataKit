"""ISHNE Holter anonymization.

Fields replaced in the fixed header (ISHNE 1.0): first name (offset 28,
40 bytes) is emptied, last name (68, 40 bytes) and patient ID (108, 20
bytes) become the patient code, NUL padded. The variable block is searched
for names with same-length replacement. ISHNE has no ECG ID.

The header CRC (CRC-CCITT over bytes 10 to the end of the variable block)
is recomputed only when the stored one was valid, in the same byte order.
Only the header is held in memory: the samples are copied as a stream.
"""

from __future__ import annotations

import binascii
import struct
from pathlib import Path

from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import (
    Codes,
    Handler,
    Rewrite,
    Spliced,
    fixed_width,
    replace_same_length,
)

_ISHNE_FIRST = (28, 40)
_ISHNE_LAST = (68, 40)
_ISHNE_ID = (108, 20)
_ISHNE_FIXED = 522


def _ishne_text(data: bytes, field: tuple[int, int]) -> str:
    start, size = field
    return data[start:start + size].split(b"\x00")[0].decode("latin-1").strip()


class ISHNEHandler(Handler):
    """ISHNE 1.0: names and ID are fixed-size header fields.

    The header CRC (CRC-CCITT over the header and variable block) is
    recomputed when the original one was valid, in the same byte order.
    """

    parser = "ISHNEHolterParser"

    def identity(self, path: Path) -> Identity:
        with open(path, "rb") as f:
            head = f.read(_ISHNE_FIXED)
        identity = Identity()
        identity.add("first_names", _ishne_text(head, _ISHNE_FIRST))
        identity.add("last_names", _ishne_text(head, _ISHNE_LAST))
        identity.add("patient_ids", _ishne_text(head, _ISHNE_ID))
        return identity

    @staticmethod
    def _header_end(data: bytes) -> int:
        var_size = struct.unpack_from("<i", data, 10)[0]
        return _ISHNE_FIXED + max(0, var_size)

    def rewrite(self, path, out_path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        with open(path, "rb") as f:
            data = bytearray(f.read(_ISHNE_FIXED))
            end = self._header_end(data)
            data += f.read(end - _ISHNE_FIXED)
        end = min(end, len(data))
        stored = struct.unpack_from("<H", data, 8)[0]
        old_crc = binascii.crc_hqx(bytes(data[10:end]), 0xFFFF)
        swapped = ((old_crc & 0xFF) << 8) | (old_crc >> 8)

        # The variable block is free text (binary fields are never searched)
        data[_ISHNE_FIXED:end] = replace_same_length(bytes(data[_ISHNE_FIXED:end]), replacer)
        for field, value in ((_ISHNE_FIRST, ""), (_ISHNE_LAST, codes.patient),
                             (_ISHNE_ID, codes.patient)):
            start, size = field
            original = _ishne_text(bytes(data), field)
            if field is _ISHNE_ID and original in codes.keep:
                continue
            if original or field is _ISHNE_LAST and identity.has_patient:
                data[start:start + size] = fixed_width(value, size)

        if stored and stored in (old_crc, swapped):
            crc = binascii.crc_hqx(bytes(data[10:end]), 0xFFFF)
            if stored == swapped and stored != old_crc:
                crc = ((crc & 0xFF) << 8) | (crc >> 8)
            struct.pack_into("<H", data, 8, crc)
        result = Rewrite()
        result.outputs[Path(out_path)] = Spliced(bytes(data), Path(path), len(data))
        return result

    def free_text(self, data: bytes) -> list[str]:
        end = min(self._header_end(data), len(data))
        fields = [data[s:s + n] for s, n in (_ISHNE_FIRST, _ISHNE_LAST, _ISHNE_ID)]
        return [f.split(b"\x00")[0].decode("latin-1").strip() for f in fields] + [
            data[_ISHNE_FIXED:end].decode("latin-1")]
