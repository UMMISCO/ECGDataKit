"""Tests for XML navigation helpers."""

from __future__ import annotations

import pytest

from ecgdatakit.parsing.helpers.xml import find_tag, read_path


class TestReadPath:
    def test_simple_path(self):
        doc = {"a": {"b": {"c": "value"}}}
        assert read_path(doc, "a/b/c") == "value"

    def test_single_key(self):
        doc = {"key": 42}
        assert read_path(doc, "key") == 42

    def test_missing_key(self):
        doc = {"a": {"b": 1}}
        assert read_path(doc, "a/x") is None

    def test_none_doc(self):
        assert read_path(None, "a/b") is None

    def test_attribute_path(self):
        doc = {"node": {"@attr": "val"}}
        assert read_path(doc, "node/@attr") == "val"

    def test_deeply_nested(self):
        doc = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        assert read_path(doc, "a/b/c/d/e") == "deep"


class TestFindTag:
    def test_find_single(self):
        doc = {"root": {"child": {"name": "found"}}}
        assert find_tag(doc, "name") == "found"

    def test_find_nested(self):
        doc = {"a": {"b": {"target": "hit"}}}
        assert find_tag(doc, "target") == "hit"

    def test_find_multiple(self):
        doc = {"a": {"x": 1}, "b": {"x": 2}}
        result = find_tag(doc, "x")
        assert isinstance(result, list)
        assert set(result) == {1, 2}

    def test_find_none(self):
        doc = {"a": {"b": 1}}
        assert find_tag(doc, "missing") is None

    def test_find_in_list(self):
        doc = {"items": [{"tag": "a"}, {"tag": "b"}]}
        result = find_tag(doc, "tag")
        assert isinstance(result, list)
        assert result == ["a", "b"]

    def test_case_insensitive(self):
        doc = {"Root": {"Child": "value"}}
        assert find_tag(doc, "child") == "value"

    def test_none_input(self):
        assert find_tag(None, "tag") is None
