"""Tests for the GE MAC 2000 parser."""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import GE_MAC2000_MEDIAN, GE_MAC2000_RHYTHM, GE_MAC2000_XML
from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.helpers.waveform_text import decode_int16_text
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.ge_mac2000 import GEMAC2000Parser


def _parse(tmp_path: Path, xml: str) -> ECGRecord:
    p = tmp_path / "mac.xml"
    p.write_text(xml, encoding="utf-8")
    return GEMAC2000Parser().parse(p)


def _strip_waveforms(xml: str, body: str) -> str:
    return re.sub(r"<Waveform>.*</Waveform>", body, xml, flags=re.S)


class TestGEMAC2000Parser:
    def test_parse_returns_ecg_record(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert isinstance(record, ECGRecord)
        assert record.source_format == "ge_mac2000"

    def test_patient(self, ge_mac2000_file: Path):
        p = GEMAC2000Parser().parse(ge_mac2000_file).patient
        assert (p.patient_id, p.first_name, p.last_name) == ("MAC001", "Bob", "Builder")
        assert p.sex == "M"
        assert p.birth_date == datetime(1978, 7, 22)  # MUSE MM-DD-YYYY
        assert p.age == 99  # the stored PatientAge wins over the DOB
        assert (p.height, p.weight) == (180.0, 82.0)

    def test_recording(self, ge_mac2000_file: Path):
        rec = GEMAC2000Parser().parse(ge_mac2000_file).recording
        assert rec.date == datetime(2023, 12, 1, 11, 0, 0)
        assert rec.duration == timedelta(seconds=10)
        assert rec.end_date is None  # not stated in the file, not computed
        assert (rec.location, rec.room, rec.technician) == ("Cardiology", "4B", "TK")
        assert rec.referring_physician == "House"

    def test_device_and_filters(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        dev = record.recording.device
        assert (dev.model, dev.software_version, dev.institution) == (
            "MAC2000", "010A", "General Hospital")
        f = record.recording.acquisition.filters
        assert (f.highpass, f.lowpass, f.notch, f.notch_active) == (0.16, 150.0, 60.0, True)

    def test_rhythm_and_median_by_waveform_type(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        assert [l.label for l in record.leads] == ["I", "II"]
        for lead, expected in zip(record.leads, GE_MAC2000_RHYTHM):
            np.testing.assert_array_equal(lead.samples, expected)
            assert lead.sampling_rate == 500
            assert lead.resolution == pytest.approx(4.88)
            assert lead.resolution_unit == "uV"
            assert lead.adc_resolution == pytest.approx(4.88)
            assert lead.is_raw is True
        assert len(record.median_beats) == 1
        np.testing.assert_array_equal(record.median_beats[0].samples, GE_MAC2000_MEDIAN)

    def test_auto_scale_gives_mv(self, ge_mac2000_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            record = FileParser().parse(ge_mac2000_file)
        assert record.leads[0].units == "mV"
        np.testing.assert_allclose(record.leads[0].samples, GE_MAC2000_RHYTHM[0] * 0.00488)

    def test_measurements(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        m = record.measurements
        assert (m.heart_rate, m.pr_interval, m.qrs_duration, m.qt_interval) == (68, 160, 92, 380)
        assert m.qtc_bazett is None  # QTCorrected does not state its formula
        assert record.annotations["qtcorrected"] == "404.6"
        assert (m.p_axis, m.qrs_axis, m.t_axis) == (50, -10, 30)
        assert record.annotations["ventricularrate"] == "68"

    def test_confirmed_interpretation_with_machine_in_annotations(self, ge_mac2000_file: Path):
        record = GEMAC2000Parser().parse(ge_mac2000_file)
        interp = record.interpretation
        assert interp.source == "confirmed"
        assert interp.statements == [("Sinus rhythm", ""), ("Normal ECG", "")]
        assert interp.interpreter == "Ann Smith"
        assert interp.interpretation_date == datetime(2023, 12, 2, 8, 30)
        assert record.annotations["machine_interpretation"] == "Sinus rhythm\nBorderline ECG"

    def test_machine_only_interpretation(self, tmp_path: Path):
        xml = re.sub(r"<OriginalDiagnosis>.*</OriginalDiagnosis>", "", GE_MAC2000_XML, flags=re.S)
        interp = _parse(tmp_path, xml).interpretation
        assert interp.source == "machine"
        assert interp.statements[1] == ("Normal ECG", "")

    def test_lead_without_scale_stays_raw(self, tmp_path: Path):
        xml = GE_MAC2000_XML.replace(
            "<LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>\n"
            "      <LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>\n"
            "      <LeadID>II</LeadID>", "<LeadID>II</LeadID>")
        leads = _parse(tmp_path, xml).leads
        assert leads[0].resolution == pytest.approx(4.88)
        assert (leads[1].resolution, leads[1].resolution_unit, leads[1].is_raw) == (1.0, "", True)

    def test_comma_separated_samples(self, tmp_path: Path):
        xml = _strip_waveforms(GE_MAC2000_XML, (
            "<Waveform><SampleBase>250.0</SampleBase><LeadData><LeadID>V1</LeadID>"
            "<WaveFormData>1000,2000,-5\n7</WaveFormData></LeadData></Waveform>"))
        record = _parse(tmp_path, xml)
        np.testing.assert_array_equal(record.leads[0].samples, [1000, 2000, -5, 7])
        assert record.leads[0].sampling_rate == 250
        assert record.recording.acquisition.signal.sampling_rate == 250
        assert record.recording.acquisition.signal.data_encoding == "int_list"

    def test_sample_rate_from_test_demographics(self, tmp_path: Path):
        xml = GE_MAC2000_XML.replace("<SampleBase>500</SampleBase>", "").replace(
            "<AcquisitionDate>", "<SampleRate>250</SampleRate><AcquisitionDate>")
        assert _parse(tmp_path, xml).leads[0].sampling_rate == 250

    def test_missing_time_keeps_date_only(self, tmp_path: Path):
        xml = GE_MAC2000_XML.replace("<AcquisitionTime>11:00:00</AcquisitionTime>", "")
        record = _parse(tmp_path, xml)
        assert record.recording.date is None
        assert record.raw_metadata["acquisition_date"] == "2023-12-01"

    def test_corrupt_waveform_raises(self, tmp_path: Path):
        xml = re.sub(r"(<LeadID>II</LeadID>\s*<WaveFormData>)[^<]*", r"\1ZAAy", GE_MAC2000_XML)
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, xml)

    def test_malformed_xml_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, GE_MAC2000_XML[:300])

    def test_no_waveform_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, _strip_waveforms(GE_MAC2000_XML, ""))

    def test_does_not_match_muse(self, tmp_path: Path):
        f = tmp_path / "muse.xml"
        f.write_text('<?xml version="1.0"?><RestingECG><MuseInfo/></RestingECG>')
        assert GEMAC2000Parser.can_parse(f, f.read_bytes()[:4096]) is False

    def test_to_json_roundtrip(self, ge_mac2000_file: Path):
        parsed = json.loads(GEMAC2000Parser().parse(ge_mac2000_file).to_json())
        assert parsed["source_format"] == "ge_mac2000"
        assert len(parsed["leads"]) == 2

    def test_auto_detection_via_file_parser(self, ge_mac2000_file: Path):
        assert FileParser().parse(ge_mac2000_file).source_format == "ge_mac2000"


class TestDecodeInt16Text:
    def test_int_list_not_read_as_base64(self):
        samples, enc = decode_int16_text("1000,2000")
        assert enc == "int_list"
        np.testing.assert_array_equal(samples, [1000, 2000])

    def test_whitespace_separated(self):
        np.testing.assert_array_equal(decode_int16_text("1000 2000")[0], [1000, 2000])

    def test_base64(self):
        samples, enc = decode_int16_text("ZAAy AA==")
        assert enc == "base64_int16le"
        np.testing.assert_array_equal(samples, [100, 50])

    @pytest.mark.parametrize("bad", ["ZAAy", "Z!!A", "abc"])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            decode_int16_text(bad)
