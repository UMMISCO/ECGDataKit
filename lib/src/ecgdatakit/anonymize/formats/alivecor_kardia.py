"""AliveCor Kardia JSON anonymization.

Fields replaced (Kardia API recording object): ``patientID`` becomes the
patient code and the recording ``id`` the ECG code, wherever these values
appear as JSON strings. Other string values, such as ``note``, are
searched for names. Strings are edited in place, the JSON layout is kept.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite, is_data

_JSON_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"(\s*:)?')


class AliveCorHandler(Handler):
    """Kardia API recording: ``patientID`` and the recording ``id``.

    String values are edited in place, the JSON layout is kept.
    """

    parser = "AliveCorKardiaParser"

    def identity(self, path: Path) -> Identity:
        doc = json.loads(Path(path).read_bytes())
        identity = Identity()
        if isinstance(doc, dict):
            identity.add("patient_ids", str(doc.get("patientID") or ""))
            identity.add("ecg_ids", str(doc.get("id") or ""))
        return identity

    def rewrite(self, path, out_path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8", errors="surrogateescape")
        exact = {v: codes.patient for v in identity.patient_ids}
        exact.update({v: codes.ecg for v in identity.ecg_ids})
        result = Rewrite()

        def edit(match: re.Match) -> str:
            if match.group(2):  # an object key
                return match.group(0)
            value = match.group(1)
            if value in exact:
                return f'"{exact[value]}"'
            return f'"{replacer.replace_value(value, is_data(value))}"'

        new = _JSON_STRING_RE.sub(edit, text)
        result.outputs[Path(out_path)] = new.encode("utf-8", errors="surrogateescape")
        return result

    def free_text(self, data: bytes) -> list[str]:
        text = data.decode("utf-8", errors="replace")
        return [m.group(1) for m in _JSON_STRING_RE.finditer(text)
                         if not m.group(2)]
