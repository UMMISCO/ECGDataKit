"""Shared rewriter of the XML formats.

Values are edited in the file text. The document is never parsed into a
tree and saved again, so the encoding, the XML declaration, indentation,
attribute order and line endings stay byte for byte as written.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, unescape

from ecgdatakit.anonymize._codes import code_uuid
from ecgdatakit.anonymize._identity import Identity, Replacer
from ecgdatakit.anonymize.formats._base import Codes, Handler, Rewrite, is_data

_TOKEN_RE = re.compile(
    r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<![^>]*>|<[^>]*>", re.DOTALL)
_NAME_RE = re.compile(r"<\s*/?\s*([^\s/>]+)")
_ATTR_RE = re.compile(r"([^\s=/<>]+)(\s*=\s*)([\"'])(.*?)\3", re.DOTALL)
_DECL_RE = re.compile(rb"^<\?xml[^>]*encoding\s*=\s*[\"']([A-Za-z0-9_.\-]+)[\"']")
_ENTITIES = {"&quot;": '"', "&apos;": "'"}


@dataclass(frozen=True)
class Rule:
    """An identity value at an element path.

    *path* is matched against the end of the element path (local names,
    case-insensitive), or the whole path when *anchored*. *attr* is the
    attribute holding the value, ``None`` for the element text. *when*
    requires an attribute value on the same element.
    """

    path: tuple[str, ...]
    kind: str
    attr: str | None = None
    anchored: bool = False
    when: tuple[str, str] | None = None
    uuid: bool = False

    def matches(self, path: list[str], attrs: dict[str, str]) -> bool:
        n = len(self.path)
        if self.anchored and len(path) != n:
            return False
        if len(path) < n or [p.lower() for p in path[-n:]] != [p.lower() for p in self.path]:
            return False
        if self.when is not None:
            key, value = self.when
            return attrs.get(key.lower(), "").strip() == value
        return True


@dataclass
class _Value:
    start: int
    end: int
    raw: str
    rule: Rule | None


def _local(name: str) -> str:
    return name.rsplit(":", 1)[-1].split("}")[-1]


def _decode(data: bytes) -> tuple[str, str, bytes]:
    """Return (text, codec, BOM) so that encoding back gives the same bytes."""
    for bom, codec in ((codecs.BOM_UTF8, "utf-8"), (codecs.BOM_UTF16_LE, "utf-16-le"),
                       (codecs.BOM_UTF16_BE, "utf-16-be")):
        if data.startswith(bom):
            return data[len(bom):].decode(codec, errors="surrogateescape"), codec, bom
    match = _DECL_RE.match(data.lstrip()[:200])
    codec = match.group(1).decode().lower() if match else "utf-8"
    try:
        codecs.lookup(codec)
    except LookupError:
        codec = "utf-8"
    if codecs.lookup(codec).name.startswith("utf-16"):
        codec = "utf-8"  # declaration without a BOM, the bytes are ASCII-compatible
    return data.decode(codec, errors="surrogateescape"), codec, b""


class XMLHandler(Handler):
    """Rules for one XML format."""

    rules: tuple[Rule, ...] = ()
    skip_elements: frozenset[str] = frozenset()
    """Elements whose text is sample data (never searched for names)."""
    skip_attrs: frozenset[str] = frozenset()

    def _values(self, text: str) -> tuple[list[_Value], list[_Value]]:
        """Identity values and free-text values of the document."""
        fields: list[_Value] = []
        free: list[_Value] = []
        # Open elements: (local name, attributes, inside sample data)
        stack: list[tuple[str, dict[str, str], bool]] = []
        pos = 0
        for match in _TOKEN_RE.finditer(text):
            if stack and match.start() > pos:
                self._text(text, pos, match.start(), stack, fields, free)
            pos = match.end()
            token = match.group(0)
            if token.startswith(("<!--", "<![CDATA[")):
                # Comments and CDATA are free text
                inner = (4, 3) if token.startswith("<!--") else (9, 3)
                if not (stack and stack[-1][2]):
                    free.append(_Value(match.start() + inner[0], match.end() - inner[1],
                                       token[inner[0]:-inner[1]], None))
                continue
            if token.startswith(("<!", "<?")):
                continue  # declaration, doctype, processing instruction
            name_match = _NAME_RE.match(token)
            if not name_match:
                continue
            if re.match(r"<\s*/", token):
                if stack:
                    stack.pop()
                continue
            name = _local(name_match.group(1))
            skip = (stack[-1][2] if stack else False) or name.lower() in self.skip_elements
            attrs: dict[str, str] = {}
            spans = []
            for attr in _ATTR_RE.finditer(token, name_match.end()):
                key = _local(attr.group(1))
                attrs[key.lower()] = unescape(attr.group(4), _ENTITIES)
                spans.append((key, match.start() + attr.start(4), match.start() + attr.end(4),
                              attr.group(4)))
            path = [s[0] for s in stack] + [name]
            for key, start, end, raw in spans:
                rule = next((r for r in self.rules if r.attr and r.attr.lower() == key.lower()
                             and r.matches(path, attrs)), None)
                if rule is not None:
                    fields.append(_Value(start, end, raw, rule))
                elif not skip and key.lower() not in self.skip_attrs:
                    free.append(_Value(start, end, raw, None))
            if not token.rstrip().endswith("/>"):
                stack.append((name, attrs, skip))
        return fields, free

    def _text(self, text, start, end, stack, fields, free) -> None:
        raw = text[start:end]
        if not raw.strip():
            return
        path = [s[0] for s in stack]
        attrs = stack[-1][1]
        rule = next((r for r in self.rules if r.attr is None and r.matches(path, attrs)), None)
        # Only the stripped value is replaced, surrounding whitespace is kept
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw.rstrip())
        value = _Value(start + lead, start + trail, raw.strip(), rule)
        if rule is not None:
            fields.append(value)
        elif not stack[-1][2]:
            free.append(value)

    def identity(self, path: Path) -> Identity:
        return self.identity_from(Path(path).read_bytes())

    def identity_from(self, data: bytes) -> Identity:
        text, _, _ = _decode(data)
        fields, _ = self._values(text)
        identity = Identity()
        for value in fields:
            if value.rule.kind != "skip":
                identity.add(value.rule.kind, unescape(value.raw, _ENTITIES))
        return identity

    def rewrite(self, path: Path, out_path: Path, identity: Identity, codes: Codes,
                replacer: Replacer) -> Rewrite:
        data = Path(path).read_bytes()
        text, codec, bom = _decode(data)
        fields, free = self._values(text)
        result = Rewrite()
        edits: list[tuple[int, int, str]] = []
        for value in fields:
            kind = value.rule.kind
            if kind == "skip" or not unescape(value.raw, _ENTITIES).strip():
                continue
            if kind in ("patient_ids", "last_names"):
                new = codes.patient
            elif kind == "first_names":
                new = ""
            else:  # ecg_ids
                new = code_uuid(codes.ecg) if value.rule.uuid else codes.ecg
                if value.raw.isupper() and value.rule.uuid:
                    new = new.upper()
                result.ecg_written.append(new)
            edits.append((value.start, value.end, escape(new, {'"': "&quot;"})))
        for value in free:
            new = replacer.replace_value(value.raw, is_data(value.raw))
            if new != value.raw:
                edits.append((value.start, value.end, new))
        for start, end, new in sorted(edits, reverse=True):
            text = text[:start] + new + text[end:]
        result.outputs[Path(out_path)] = bom + text.encode(codec, errors="surrogateescape")
        return result

    def free_text(self, data: bytes) -> list[str]:
        text, _, _ = _decode(data)
        fields, free = self._values(text)
        return [v.raw for v in free + fields]
