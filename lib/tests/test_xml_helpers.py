"""Tests for the XML parsing, navigation and waveform text helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.parsing.helpers.waveform_text import decode_int16_text
from ecgdatakit.parsing.helpers.xml import find_tag, parse_xml_root, read_path


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


class TestParseXmlRoot:
    def test_namespaces_stripped(self):
        root = parse_xml_root(b'<a xmlns="urn:x" xmlns:xsi="urn:y"><b xsi:type="T"/></a>')
        assert root.tag == "a"
        assert root.find("b").get("type") == "T"

    def test_undeclared_latin1_bytes_decoded(self):
        root = parse_xml_root('<?xml version="1.0" encoding="UTF-8"?><a>Hôpital</a>'.encode("latin-1"))
        assert root.text == "Hôpital"

    @pytest.mark.parametrize("dtd", [
        '<!ENTITY e "hello">',
        '<!ENTITY e SYSTEM "file:///etc/passwd">',
        '<!ENTITY l0 "LOL"><!ENTITY l1 "&l0;&l0;&l0;&l0;&l0;&l0;&l0;&l0;">',
    ])
    def test_entities_refused(self, dtd):
        with pytest.raises(CorruptedFileError, match="entity"):
            parse_xml_root(f'<?xml version="1.0"?><!DOCTYPE a [{dtd}]><a>x</a>'.encode())


class TestDecodeInt16Text:
    def test_int_list_not_read_as_base64(self):
        samples, enc = decode_int16_text("1000,2000")
        assert enc == "int_list"
        np.testing.assert_array_equal(samples, [1000, 2000])

    def test_whitespace_separated(self):
        np.testing.assert_array_equal(decode_int16_text("1000 2000")[0], [1000, 2000])

    @pytest.mark.parametrize("text, value", [("1234", 1234), ("7", 7), ("-5", -5), ("32767", 32767)])
    def test_single_integer(self, text, value):
        samples, enc = decode_int16_text(text)
        assert enc == "int_list"
        np.testing.assert_array_equal(samples, [value])

    def test_base64(self):
        samples, enc = decode_int16_text("ZAAy AA==")
        assert enc == "base64_int16le"
        np.testing.assert_array_equal(samples, [100, 50])

    def test_long_digit_run_is_base64(self):
        samples, enc = decode_int16_text("12345678")
        assert enc == "base64_int16le"
        assert samples.size == 3

    @pytest.mark.parametrize("bad", ["ZAAy", "Z!!A", "abc"])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            decode_int16_text(bad)
