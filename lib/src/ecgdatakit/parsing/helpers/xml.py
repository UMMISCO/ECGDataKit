"""XML navigation utilities for xmltodict-parsed documents.

These helpers operate on the nested dict/list structures produced by
``xmltodict.parse()``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from defusedxml import ElementTree as SafeET

from ecgdatakit.parsing.helpers.labels import decode_text

_DECL_RE = re.compile(rb"^\s*<\?xml[^>]*?encoding\s*=\s*[\"']([A-Za-z0-9._-]+)[\"'][^>]*\?>")


def parse_xml_root(raw: bytes) -> ET.Element:
    """Parse XML bytes into an element tree with namespaces removed.

    The declared encoding (or a BOM) is honoured. When the bytes do not match
    the declared encoding, the text is decoded with :func:`decode_text` so no
    character is lost. Namespace URIs are stripped from tags and attribute
    names (``{urn:hl7-org:v3}series`` becomes ``series``, ``xsi:type``
    becomes ``type``). Raises :class:`xml.etree.ElementTree.ParseError` when
    the document is not well-formed.
    """
    try:
        root = SafeET.fromstring(raw)
    except ET.ParseError:
        body = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
        match = _DECL_RE.match(body)
        text = decode_text(body, match.group(1).decode("ascii") if match else None)
        text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text, count=1)
        root = SafeET.fromstring(text)
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
        if any("}" in k for k in el.attrib):
            el.attrib = {k.split("}", 1)[-1]: v for k, v in el.attrib.items()}
    return root


def find_tag(doc: dict | list | None, tag: str) -> list | dict | str | None:
    """Recursively find all occurrences of *tag* in an xmltodict structure.

    Returns a single value when exactly one match is found, a list for
    multiple matches, or ``None`` when nothing matches.

    Parameters
    ----------
    doc : dict | list | None
        The xmltodict-parsed document (or a sub-tree thereof).
    tag : str
        The tag name to search for (case-insensitive).
    """
    if doc is None:
        return None

    results: list = []
    _collect(doc, tag.lower(), results)

    if len(results) == 0:
        return None
    if len(results) == 1:
        return results[0]
    return results


def _collect(doc: dict | list, tag_lower: str, results: list) -> None:
    """Internal recursive collector."""
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k.lower() == tag_lower:
                results.append(v)
            else:
                _collect(v, tag_lower, results)
    elif isinstance(doc, list):
        for item in doc:
            _collect(item, tag_lower, results)


def read_path(doc: dict | None, path: str) -> object | None:
    """Navigate a nested dict by a slash-delimited path.

    Example::

        read_path(doc, "AnnotatedECG/effectiveTime/low/@value")

    Parameters
    ----------
    doc : dict | None
        The root dict to navigate.
    path : str
        Slash-separated key path (e.g. ``"root/child/@attr"``).
    """
    if doc is None:
        return None

    current: object = doc
    for part in path.split("/"):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
