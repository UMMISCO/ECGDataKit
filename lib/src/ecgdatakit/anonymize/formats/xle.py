"""XLE anonymization (experimental format).

The patient fields are text in the header, with the two bytes of each
16-bit word swapped, each value after its label and ended by a NUL (see
:mod:`ecgdatakit.parsing.parsers.xle`). The text area is swapped back,
edited in place and swapped again: given name emptied, family name and
patient ID replaced by the patient code, names and IDs inside the history
and comment replaced. Birth date and sex are kept. XLE has no ECG ID.

Every value keeps to its slot: the bytes up to the alignment spaces of the
next label, and for the comment, its value and the NULs after it. Only the
header is held in memory: the samples are copied as a stream.
"""

from __future__ import annotations

from pathlib import Path

from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite, Spliced
from ecgdatakit.parsing.parsers.xle import _FIELDS, _TEXT_END, _fields, _swap16


def _slots(text: bytes) -> dict[str, tuple[int, int]]:
    """Start and end of the bytes each value may use, in the swapped text."""
    slots: dict[str, tuple[int, int]] = {}
    for k, (label, off) in enumerate(_FIELDS):
        start = off + len(label) + 1
        if text[start:start + 1] == b" ":
            start += 1
        if k + 1 < len(_FIELDS):
            end = _FIELDS[k + 1][1]
            end -= len(text[start:end]) - len(text[start:end].rstrip(b" "))
        else:
            value = text[start:_TEXT_END].split(b"\x00", 1)[0]
            end = start + len(value)
            while end < _TEXT_END and text[end] == 0:
                end += 1
        slots[label] = (start, max(start, end))
    return slots


class XLEHandler(Handler):
    """XLE ECG: patient fields in a byte-swapped header text area."""

    parser = "XLEParser"

    def identity(self, path: Path) -> Identity:
        with open(path, "rb") as f:
            fields = _fields(f.read(_TEXT_END))
        identity = Identity()
        identity.add("first_names", fields["Given name"])
        identity.add("last_names", fields["Family name"])
        identity.add("patient_ids", fields["ID"])
        return identity

    def rewrite(self, path, out_path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        with open(path, "rb") as f:
            head = bytearray(f.read(_TEXT_END))
        text = bytearray(_swap16(bytes(head)))
        values = _fields(bytes(head))
        result = Rewrite()

        def put(label: str, value: str) -> None:
            start, end = _slots(bytes(text))[label]
            raw = value.encode("latin-1", errors="replace")
            if len(raw) > end - start:
                result.notes.append(f"{label} cut to its {end - start}-byte field")
                raw = raw[:end - start]
            text[start:end] = raw + b"\x00" * (end - start - len(raw))

        if values["Given name"]:
            put("Given name", "")
        if values["Family name"] or identity.has_patient:
            put("Family name", codes.patient)
        if values["ID"] and values["ID"] not in codes.keep:
            put("ID", codes.patient)
        for label in ("History", "Comment"):
            if values[label]:
                put(label, replacer.replace(values[label]))

        head[:len(text)] = _swap16(bytes(text))
        result.outputs[Path(out_path)] = Spliced(bytes(head), Path(path), len(head))
        return result

    def free_text(self, data: bytes) -> list[str]:
        return [v for v in _fields(data[:_TEXT_END]).values() if v]
