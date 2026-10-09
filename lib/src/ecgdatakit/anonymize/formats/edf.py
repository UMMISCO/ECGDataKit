"""EDF and EDF+ anonymization.

Header fields are 80 ASCII characters, space padded.

- EDF+ patient field ``code sex birthdate name ...``: the code and the name
  become the patient code. Sex and birth date are kept, further subfields
  are searched for names.
- EDF+ recording field ``Startdate date admincode technician equipment``:
  the admin code becomes the ECG code.
- Plain EDF: the whole patient field is the patient ID (as the parser reads
  it) and becomes the patient code. The recording field is searched for
  names.

Only the 256-byte header is held in memory: the rest is copied as a stream.
"""

from __future__ import annotations

from pathlib import Path

from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite, Spliced

_EDF_PATIENT = (8, 80)
_EDF_RECORDING = (88, 80)


class EDFHandler(Handler):
    """EDF and EDF+ header fields (80 ASCII characters, space padded).

    EDF+: patient subfields ``code sex birthdate name``, recording subfields
    ``Startdate date admincode technician equipment``. The admin code is the
    ECG ID. Plain EDF: the whole patient field is the patient ID, as the
    parser reads it.
    """

    parser = "EDFParser"

    @staticmethod
    def _fields(head: bytes) -> tuple[bool, str, str]:
        plus = head[192:236].decode("latin-1").strip().upper().startswith("EDF+")
        patient = head[8:88].decode("latin-1")
        recording = head[88:168].decode("latin-1")
        return plus, patient, recording

    def identity(self, path: Path) -> Identity:
        with open(path, "rb") as f:
            head = f.read(256)
        plus, patient, recording = self._fields(head)
        identity = Identity()
        if plus:
            parts = patient.split()
            if parts and parts[0] != "X":
                identity.add("patient_ids", parts[0].replace("_", " "))
            if len(parts) >= 4 and parts[3] != "X":
                name = parts[3].replace("_", " ")
                last, _, first = name.partition(",")
                identity.add("last_names", last)
                identity.add("first_names", first)
            rec = recording.split()
            if len(rec) >= 3 and rec[0] == "Startdate" and rec[2] != "X":
                identity.add("ecg_ids", rec[2].replace("_", " "))
        else:
            identity.add("patient_ids", patient)
        return identity

    def rewrite(self, path, out_path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        with open(path, "rb") as f:
            data = bytearray(f.read(256))
        plus, patient, recording = self._fields(bytes(data))
        result = Rewrite()
        if plus:
            parts = patient.split()
            if parts and parts[0] != "X":
                parts[0] = codes.patient
            if len(parts) >= 4 and parts[3] != "X":
                parts[3] = codes.patient
            parts[4:] = [replacer.replace(p) for p in parts[4:]]
            new_patient = " ".join(parts)
            rec = recording.split()
            if len(rec) >= 3 and rec[0] == "Startdate" and rec[2] != "X":
                rec[2] = codes.ecg
                result.ecg_written.append(codes.ecg)
            rec[3:] = [replacer.replace(p).replace(" ", "_") for p in rec[3:]]
            new_recording = " ".join(rec)
        else:
            new_patient = codes.patient if patient.strip() else patient
            new_recording = replacer.replace(recording)
        for (start, size), value in ((_EDF_PATIENT, new_patient), (_EDF_RECORDING, new_recording)):
            raw = value.encode("latin-1", errors="replace")
            if len(raw) > size:
                raise ValueError(f"EDF header field longer than {size} characters after anonymization")
            data[start:start + size] = raw.ljust(size, b" ")
        result.outputs[Path(out_path)] = Spliced(bytes(data), Path(path), len(data))
        return result

    def free_text(self, data: bytes) -> list[str]:
        return data[8:168].decode("latin-1").split()
