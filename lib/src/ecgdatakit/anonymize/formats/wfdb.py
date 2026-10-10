"""WFDB anonymization.

A record is its ``.hea`` header and the files it names (signals,
annotations). Header comments ``name:`` and ``id:`` (also ``patient name:``
and ``patient id:``) become the patient code. Other comments are searched
for names. Signal files are copied unchanged.

When the anonymized file name changes the record name, the record line, the
signal file names in the header and every file of the record are renamed
together. Multi-segment records keep their names.
"""

from __future__ import annotations

import re
from pathlib import Path

from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite, Spliced

_NAME_KEYS = {"name", "patient name"}
_ID_KEYS = {"id", "patient id"}
# "# key: value" comment line: (prefix "# ", key, separator ": ", value, trailing spaces)
_COMMENT_RE = re.compile(r"^(\s*#\s*)(<?[^:<>]+>?)(\s*:\s*)(.*?)(\s*)$")


def _read_lines(path: Path) -> list[tuple[str, str]]:
    """Lines of a header as (text, line ending).

    Bytes that are not valid UTF-8 are kept as they are (surrogateescape),
    so writing the lines back gives the same bytes.
    """
    text = Path(path).read_bytes().decode("utf-8", errors="surrogateescape")
    lines = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        lines.append((body, line[len(body):]))
    return lines


def _comment(body: str) -> tuple[str, str] | None:
    """(key, value) of a ``# key: value`` comment, key in lower case."""
    match = _COMMENT_RE.match(body)
    if match is None:
        return None
    return match.group(2).strip().strip("<>").strip().lower(), match.group(4).strip()


class WFDBHandler(Handler):
    """WFDB record: the ``.hea`` header and the files it names."""

    parser = "WFDBParser"

    def identity(self, path: Path) -> Identity:
        identity = Identity()
        for body, _ in _read_lines(path):
            comment = _comment(body) if body.lstrip().startswith("#") else None
            if comment is None:
                continue
            key, value = comment
            if key in _NAME_KEYS and value:
                # Same split as the parser: first word, then the rest
                first, _, last = value.partition(" ")
                identity.add("first_names", first)
                identity.add("last_names", last)
            elif key in _ID_KEYS:
                identity.add("patient_ids", value)
        return identity

    def companions(self, path: Path) -> list[Path]:
        """Files of the record next to the header (``100.dat``, ``100.atr``...)."""
        path = Path(path)
        prefix = path.stem + "."
        return sorted(p for p in path.parent.iterdir()
                      if p.is_file() and p != path and p.name.startswith(prefix))

    @staticmethod
    def is_multi_segment(path: Path) -> bool:
        """True when the record line is ``name/segments ...``."""
        for body, _ in _read_lines(path):
            if body.strip() and not body.lstrip().startswith("#"):
                return "/" in body.split()[0]
        return False

    def rewrite(self, path, out_path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        path, out_path = Path(path), Path(out_path)
        old_name, new_name = path.stem, out_path.stem
        out_lines = []
        record_line_done = False
        for body, ending in _read_lines(path):
            if body.lstrip().startswith("#"):
                body = self._comment_line(body, codes, replacer)
            elif body.strip():
                # The first non-comment line is the record line, the next
                # ones are signal lines starting with their file name
                body = self._rename(body, old_name, new_name, first=not record_line_done)
                record_line_done = True
            out_lines.append(body + ending)

        result = Rewrite()
        result.outputs[out_path] = "".join(out_lines).encode("utf-8", errors="surrogateescape")
        for companion in self.companions(path):
            target = out_path.parent / (new_name + companion.name[len(old_name):])
            result.outputs[target] = Spliced(b"", companion, 0)  # copied unchanged
        return result

    @staticmethod
    def _comment_line(body: str, codes: Codes, replacer: Replacer) -> str:
        """Patient name or ID comment: value replaced by the code. Other
        comments: names and IDs replaced where they appear."""
        match = _COMMENT_RE.match(body)
        comment = _comment(body)
        if (match and comment and comment[0] in _ID_KEYS and comment[1] in codes.keep):
            return body
        if match and comment and comment[0] in _NAME_KEYS | _ID_KEYS and comment[1]:
            prefix, key, separator, _, trailing = match.groups()
            return prefix + key + separator + codes.patient + trailing
        return replacer.replace(body)

    @staticmethod
    def _rename(body: str, old_name: str, new_name: str, first: bool) -> str:
        """Record line: record name. Signal line: ``<record>.dat`` file name."""
        if old_name == new_name:
            return body
        tokens = body.split(" ")
        if first and tokens[0] == old_name:
            tokens[0] = new_name
        elif not first and tokens[0].startswith(old_name + "."):
            tokens[0] = new_name + tokens[0][len(old_name):]
        return " ".join(tokens)

    def free_text(self, data: bytes) -> list[str]:
        return [line for line in data.decode("utf-8", errors="replace").splitlines()
                if line.lstrip().startswith("#")]
