"""Tests for the Mindray BeneHeart R12 parser."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import BENEHEART_R12_XML
from ecgdatakit.exceptions import CorruptedFileError, MissingElementError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.beneheart_r12 import BeneHeartR12Parser


def _parse(tmp_path: Path, xml: str) -> ECGRecord:
    p = tmp_path / "bh.xml"
    p.write_text(xml, encoding="utf-8")
    return BeneHeartR12Parser().parse(p)


class TestBeneHeartR12Parser:
    def test_parse_returns_ecg_record(self, beneheart_r12_file: Path):
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert isinstance(record, ECGRecord)
        assert record.source_format == "beneheart_r12"

    def test_patient(self, beneheart_r12_file: Path):
        p = BeneHeartR12Parser().parse(beneheart_r12_file).patient
        assert (p.patient_id, p.first_name, p.last_name, p.sex) == ("BH001", "Alice", "Wonder", "F")
        assert p.birth_date == datetime(1992, 3, 20)
        assert p.age == 31  # computed from DOB at acquisition
        assert (p.height, p.weight) == (165.0, 60.0)

    def test_recording(self, beneheart_r12_file: Path):
        rec = BeneHeartR12Parser().parse(beneheart_r12_file).recording
        assert rec.date == datetime(2023, 12, 1, 9, 15, 0)
        assert rec.duration == timedelta(seconds=2 / 500)
        assert rec.device.model == "BeneHeart R12"
        assert rec.device.manufacturer == "Mindray"
        assert (rec.device.serial_number, rec.device.software_version) == ("R12-0001", "01.02")

    def test_filters_zero_notch_is_none(self, beneheart_r12_file: Path):
        f = BeneHeartR12Parser().parse(beneheart_r12_file).recording.acquisition.filters
        assert (f.highpass, f.lowpass, f.notch, f.notch_active) == (0.05, 150.0, None, None)

    def test_leads_raw_without_file_resolution(self, beneheart_r12_file: Path):
        """No resolution in the file: counts stay raw, no unit is claimed."""
        record = BeneHeartR12Parser().parse(beneheart_r12_file)
        assert [l.label for l in record.leads] == ["I", "II"]
        for lead in record.leads:
            np.testing.assert_array_equal(lead.samples, [100, 50])
            assert lead.sampling_rate == 500
            assert (lead.resolution, lead.resolution_unit, lead.units) == (1.0, "", "")
            assert lead.is_raw is True

    def test_auto_scale_warns_for_unscaled_leads(self, beneheart_r12_file: Path):
        with pytest.warns(UserWarning, match="raw ADC"):
            record = FileParser().parse(beneheart_r12_file)
        np.testing.assert_array_equal(record.leads[0].samples, [100, 50])

    @pytest.mark.parametrize("lead_xml", [
        '<Lead Name="I" Data="ZAAyAA==" Resolution="2.5" Unit="uV"/>',
        '<Lead Name="I" Data="ZAAyAA=="><Resolution Unit="uV">2.5</Resolution></Lead>',
    ])
    def test_resolution_from_file(self, tmp_path: Path, lead_xml: str):
        xml = BENEHEART_R12_XML.replace('<Lead Name="I" Data="ZAAyAA=="/>', lead_xml)
        lead = _parse(tmp_path, xml).leads[0]
        assert (lead.resolution, lead.resolution_unit, lead.is_raw) == (2.5, "uV", True)
        assert lead.adc_resolution == 2.5
        np.testing.assert_allclose(lead.to_physical().samples, [250.0, 125.0])

    def test_resolution_without_unit_stays_unitless(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace('<Lead Name="I" Data="ZAAyAA=="/>',
                                        '<Lead Name="I" Data="ZAAyAA==" Resolution="2.5"/>')
        lead = _parse(tmp_path, xml).leads[0]
        assert (lead.resolution, lead.resolution_unit, lead.is_raw) == (2.5, "", True)

    def test_measurements_rounded(self, beneheart_r12_file: Path):
        m = BeneHeartR12Parser().parse(beneheart_r12_file).measurements
        assert (m.heart_rate, m.pr_interval, m.qtc_bazett) == (73, 150, 410)

    def test_machine_statements(self, beneheart_r12_file: Path):
        interp = BeneHeartR12Parser().parse(beneheart_r12_file).interpretation
        assert interp.source == "machine"
        assert interp.statements == [("Sinus rhythm", ""), ("Normal ECG", "")]

    def test_plain_text_diagnosis(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace(
            "<Diagnosis>\n    <Statement>Sinus rhythm</Statement>\n"
            "    <Statement>Normal ECG</Statement>\n  </Diagnosis>",
            "<Diagnosis>Sinus rhythm</Diagnosis>")
        assert _parse(tmp_path, xml).interpretation.statements == [("Sinus rhythm", "")]

    def test_attribute_statements_each_kept(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace(
            "<Diagnosis>\n    <Statement>Sinus rhythm</Statement>\n"
            "    <Statement>Normal ECG</Statement>\n  </Diagnosis>",
            '<Diagnosis Text="A"/><Diagnosis Text="B"/>')
        assert _parse(tmp_path, xml).interpretation.statements == [("A", ""), ("B", "")]

    def test_no_statements_leaves_interpretation_empty(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace(
            "<Diagnosis>\n    <Statement>Sinus rhythm</Statement>\n"
            "    <Statement>Normal ECG</Statement>\n  </Diagnosis>", "")
        interp = _parse(tmp_path, xml).interpretation
        assert interp.statements == [] and interp.source == ""

    def test_per_label_children(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace(
            '<Lead Name="I" Data="ZAAyAA=="/>\n    <Lead Name="II" Data="ZAAyAA=="/>',
            '<I unit="x">ZAAyAA==</I><V1>1,2,3</V1>')
        leads = _parse(tmp_path, xml).leads
        assert [l.label for l in leads] == ["I", "V1"]
        np.testing.assert_array_equal(leads[0].samples, [100, 50])
        np.testing.assert_array_equal(leads[1].samples, [1, 2, 3])

    def test_float_sample_rate_consistent(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace("<SampleRate>500</SampleRate>",
                                        "<SampleRate>250.0</SampleRate>")
        record = _parse(tmp_path, xml)
        assert record.leads[0].sampling_rate == 250
        assert record.recording.acquisition.signal.sampling_rate == 250
        assert record.recording.duration == timedelta(seconds=2 / 250)

    def test_ambiguous_dob_not_guessed(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace("1992-03-20", "03/04/1990")
        record = _parse(tmp_path, xml)
        assert record.patient.birth_date is None
        assert record.raw_metadata["birth_date"] == "03/04/1990"
        assert record.patient.age == 31  # falls back to the Age field

    def test_lowercase_label_normalized_and_unique(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace('Name="II"', 'Name="i"')
        assert [l.label for l in _parse(tmp_path, xml).leads] == ["I", "I_2"]

    def test_corrupt_lead_raises(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace('Name="I" Data="ZAAyAA=="', 'Name="I" Data="ZAAy"')
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, xml)

    def test_missing_rate_raises(self, tmp_path: Path):
        xml = BENEHEART_R12_XML.replace("<SampleRate>500</SampleRate>", "")
        with pytest.raises(MissingElementError):
            _parse(tmp_path, xml)

    def test_malformed_xml_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, BENEHEART_R12_XML[:250])

    def test_to_json_roundtrip(self, beneheart_r12_file: Path):
        parsed = json.loads(BeneHeartR12Parser().parse(beneheart_r12_file).to_json())
        assert parsed["source_format"] == "beneheart_r12"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, beneheart_r12_file: Path):
        with pytest.warns(UserWarning):
            assert FileParser().parse(beneheart_r12_file).source_format == "beneheart_r12"
