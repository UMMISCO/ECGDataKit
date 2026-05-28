"""XML navigation utilities for xmltodict-parsed documents.

These helpers operate on the nested dict/list structures produced by
``xmltodict.parse()``.
"""

from __future__ import annotations


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
