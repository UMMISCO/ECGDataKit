"""Tests for the Mortara ELI XML parser."""

from __future__ import annotations

import json
import re
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MORTARA_MEDIAN_N, MORTARA_RHYTHM_N, MORTARA_XML, mortara_counts
from ecgdatakit.exceptions import CorruptedFileError, UnsupportedFormatError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.mortara_el250 import MortaraEL250Parser


def _write(tmp_path: Path, xml: str) -> Path:
    p = tmp_path / "m.xml"
    p.write_text(xml, encoding="utf-8")
    return p


def _parse(tmp_path: Path, xml: str) -> ECGRecord:
    return MortaraEL250Parser().parse(_write(tmp_path, xml))


class TestMortaraEL250Parser:
    def test_parse_returns_ecg_record(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert isinstance(record, ECGRecord)
        assert record.source_format == "mortara_el250"

    def test_patient_from_subject(self, mortara_file: Path):
        p = MortaraEL250Parser().parse(mortara_file).patient
        assert p.first_name == "José O'Neil"  # no characters dropped
        assert p.last_name == "Doe"
        assert p.patient_id == "PAT001"
        assert p.sex == "M"
        assert p.birth_date == datetime(1990, 5, 15)

    def test_stored_age_wins_over_dob(self, mortara_file: Path):
        # AGE attribute says 99, DOB and acquisition date would give 33
        assert MortaraEL250Parser().parse(mortara_file).patient.age == 99

    def test_height_weight_converted(self, mortara_file: Path):
        p = MortaraEL250Parser().parse(mortara_file).patient
        assert p.height == pytest.approx(177.8)
        assert p.weight == pytest.approx(80.0)

    def test_medications(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.patient.medications == ["Bisoprolol"]
        assert record.raw_metadata["medications"] == [
            {"CLASS_CODE": "2", "DRUG_CODE": "7", "NAME": "Bisoprolol"}
        ]

    def test_recording(self, mortara_file: Path):
        rec = MortaraEL250Parser().parse(mortara_file).recording
        assert rec.date == datetime(2023, 12, 1, 12, 0, 0)
        assert rec.duration == timedelta(seconds=1)
        assert rec.end_date is None  # not stated in the file, not computed
        assert (rec.room, rec.location, rec.technician) == ("R12", "Cardio", "TS")

    def test_device(self, mortara_file: Path):
        dev = MortaraEL250Parser().parse(mortara_file).recording.device
        assert dev.manufacturer == "Mortara Instrument, Inc."
        assert dev.model == "ELI250c"
        assert dev.serial_number == "SN123"
        assert dev.software_version == "V2.0.4.0"
        assert dev.institution == "Main Hospital"
        assert dev.acquisition_type == "RESTING"

    def test_filters(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        f = record.recording.acquisition.filters
        assert f.lowpass == 150.0
        assert f.highpass is None and f.notch is None
        assert record.raw_metadata["filters"] == {
            "BASELINE_ROLL_FILTER": "5", "PRINT_FILTER": "150", "FILTER_BITMAP": "2",
        }

    def test_file_format_and_raw_metadata(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert record.file_format.creation_date == date(2023, 12, 2)
        raw = record.raw_metadata
        assert raw["generator"] == "ELI Link 4.5.0.2"
        assert raw["source"]["ACQUIRING_DEVICE_INTERPRETATION_SW_VERSION"] == "7.2.3"
        assert raw["systolic_bp"] == "120" and raw["diastolic_bp"] == "80"
        assert raw["site"] == {"ID": "3"}

    def test_rhythm_leads(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        assert [lead.label for lead in record.leads] == ["I", "II", "V1"]
        for i, lead in enumerate(record.leads):
            np.testing.assert_array_equal(lead.samples, mortara_counts(i, MORTARA_RHYTHM_N))
            assert lead.samples.dtype == np.float64
            assert lead.sampling_rate == 500
            assert lead.resolution == pytest.approx(2.5)  # 1000 / 400 uV per count
            assert lead.resolution_unit == "uV"
            assert lead.adc_resolution == 400.0
            assert lead.adc_resolution_unit == "counts/mV"
            assert lead.is_raw is True and lead.units == ""

    def test_median_beats_use_typical_cycle_scale(self, mortara_file: Path):
        beats = MortaraEL250Parser().parse(mortara_file).median_beats
        assert [b.label for b in beats] == ["I", "II"]
        assert beats[0].sampling_rate == 250
        assert beats[0].resolution == pytest.approx(5.0)  # UNITS_PER_MV=200
        np.testing.assert_array_equal(beats[1].samples, mortara_counts(1, MORTARA_MEDIAN_N))

    def test_auto_scale_gives_mv(self, mortara_file: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            record = FileParser().parse(mortara_file)
        lead = record.leads[1]
        assert lead.units == "mV"
        np.testing.assert_allclose(lead.samples, mortara_counts(1, MORTARA_RHYTHM_N) / 400.0)
        np.testing.assert_allclose(
            record.median_beats[0].samples, mortara_counts(0, MORTARA_MEDIAN_N) / 200.0)

    def test_raw_counts_without_auto_scale(self, mortara_file: Path):
        with pytest.warns(UserWarning, match="raw ADC"):
            record = FileParser().parse(mortara_file, auto_scale=False)
        np.testing.assert_array_equal(record.leads[0].samples, mortara_counts(0, MORTARA_RHYTHM_N))

    def test_measurements(self, mortara_file: Path):
        record = MortaraEL250Parser().parse(mortara_file)
        m = record.measurements
        assert (m.heart_rate, m.rr_interval, m.qrs_count) == (75, 800, 12)
        assert (m.pr_interval, m.qrs_duration, m.qt_interval) == (163, 93, 456)
        assert m.qtc_bazett == 469 and m.qtc_fridericia == 464
        assert (m.p_axis, m.qrs_axis, m.t_axis) == (57, -35, 40)
        assert record.raw_metadata["qtc"] == 463
        assert record.annotations["q_onset"] == "-45"
        assert record.annotations["comment"] == "Chest pain"

    def test_machine_statements(self, mortara_file: Path):
        interp = MortaraEL250Parser().parse(mortara_file).interpretation
        assert interp.source == "machine"
        assert interp.statements == [
            ("SINUS RHYTHM", ""),
            ("LEFT AXIS DEVIATION", "QRS AXIS < -30"),
            ("UNCONFIRMED REPORT", ""),
        ]
        assert interp.severity == ""

    def test_signal_characteristics(self, mortara_file: Path):
        sig = MortaraEL250Parser().parse(mortara_file).recording.acquisition.signal
        assert sig.sampling_rate == 500
        assert sig.resolution == pytest.approx(2.5)
        assert sig.bits_per_sample == 16
        assert sig.data_encoding == "base64_int16le"

    def test_to_json_roundtrip(self, mortara_file: Path):
        parsed = json.loads(MortaraEL250Parser().parse(mortara_file).to_json())
        assert parsed["source_format"] == "mortara_el250"
        assert parsed["patient"]["last_name"] == "Doe"

    def test_auto_detection_via_file_parser(self, mortara_file: Path):
        assert FileParser().parse(mortara_file).source_format == "mortara_el250"

    def test_detection_without_channel_or_brand_in_header(self, tmp_path: Path):
        xml = MORTARA_XML.replace("Mortara Instrument, Inc.", "Welch Allyn, Inc.")
        xml = xml.replace("<!-- Generated by ELI Link 4.5.0.2 -->\n", "")
        # Push CHANNEL far beyond the sniffed header
        xml = xml.replace("<SITE ID=\"3\" />", "<SITE ID=\"3\" />" + " " * 8000)
        p = _write(tmp_path, xml)
        assert MortaraEL250Parser.can_parse(p, p.read_bytes()[:4096])

    def test_placeholder_dob_gives_none(self, tmp_path: Path):
        xml = MORTARA_XML.replace('DOB="19900515" DOB_XML="1990-05-15"',
                                  'DOB="00000000" DOB_XML="0000-00-00"')
        p = _parse(tmp_path, xml).patient
        assert p.birth_date is None
        assert p.age == 99  # falls back to AGE in years

    def test_demographic_fields_fallback(self, tmp_path: Path):
        xml = re.sub(r"<SUBJECT [^>]*/>", "", MORTARA_XML)
        p = _parse(tmp_path, xml).patient
        assert (p.last_name, p.first_name, p.patient_id) == ("Doe", "John", "PAT001")

    def test_rhythm_scale_read_per_channel(self, tmp_path: Path):
        xml = re.sub(r"<TYPICAL_CYCLE .*?</TYPICAL_CYCLE>", "", MORTARA_XML, flags=re.S)
        record = _parse(tmp_path, xml)
        assert record.median_beats == []
        assert record.leads[0].resolution == pytest.approx(2.5)

    def test_empty_typical_cycle(self, tmp_path: Path):
        xml = re.sub(r"<TYPICAL_CYCLE .*?</TYPICAL_CYCLE>", "<TYPICAL_CYCLE />",
                     MORTARA_XML, flags=re.S)
        assert _parse(tmp_path, xml).median_beats == []

    @pytest.mark.parametrize("mutation", [
        ('DURATION="500"', 'DURATION="400"'),
        ('SAMPLE_FREQ="500"', 'SAMPLE_FREQ="abc"'),
        ('DATA="', 'DATA="!!'),
    ])
    def test_corrupt_channel_raises(self, tmp_path: Path, mutation):
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, MORTARA_XML.replace(*mutation, 1))

    def test_odd_length_data_raises(self, tmp_path: Path):
        xml = re.sub(r'(<CHANNEL [^>]*DATA=")[^"]*"', r'\1ZAAy"', MORTARA_XML, count=1)
        with pytest.raises(CorruptedFileError, match="int16"):
            _parse(tmp_path, xml)

    def test_unsupported_bits_raises(self, tmp_path: Path):
        xml = MORTARA_XML.replace('BITS="16" FORMAT="SIGNED" UNITS_PER_MV="400"',
                                  'BITS="8" FORMAT="SIGNED" UNITS_PER_MV="400"', 1)
        with pytest.raises(UnsupportedFormatError):
            _parse(tmp_path, xml)

    def test_malformed_xml_raises(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, MORTARA_XML[:400])

    def test_no_channels_raises(self, tmp_path: Path):
        xml = re.sub(r"<CHANNEL [^>]*/>\n", "", MORTARA_XML)
        with pytest.raises(CorruptedFileError):
            _parse(tmp_path, xml)

    def test_nonzero_offset_warns_and_is_kept(self, tmp_path: Path):
        xml = MORTARA_XML.replace('<CHANNEL OFFSET="0"', '<CHANNEL OFFSET="5"', 1)
        with pytest.warns(UserWarning, match="OFFSET") as caught:
            record = _parse(tmp_path, xml)
        assert record.raw_metadata["channel_offsets"] == {"I": "5"}
        assert caught[0].filename == __file__

    def test_duplicate_labels_made_unique(self, tmp_path: Path):
        xml = MORTARA_XML.replace('NAME="V1"', 'NAME="II"')
        assert [lead.label for lead in _parse(tmp_path, xml).leads] == ["I", "II", "II_2"]


def test_median_rate_not_stated_uses_rhythm_rate(tmp_path: Path):
    xml = MORTARA_XML.replace(f'DURATION="{MORTARA_MEDIAN_N}" SAMPLE_FREQ="250" ',
                              f'DURATION="{MORTARA_MEDIAN_N}" ')
    with pytest.warns(UserWarning, match="no SAMPLE_FREQ"):
        record = _parse(tmp_path, xml)
    assert record.median_beats[0].sampling_rate == record.leads[0].sampling_rate
    assert record.raw_metadata["median_sampling_rate_stated"] is False


def test_median_scale_not_stated_stays_raw(tmp_path: Path):
    xml = MORTARA_XML.replace('UNITS_PER_MV="200" DURATION', 'DURATION')
    with pytest.warns(UserWarning, match="no UNITS_PER_MV"):
        record = _parse(tmp_path, xml)
    beat = record.median_beats[0]
    assert beat.is_raw and beat.resolution_unit == ""
    assert record.raw_metadata["median_scale_stated"] is False


def test_undeclared_latin1_text_decoded(tmp_path: Path):
    xml = MORTARA_XML.replace('COMMENT="Chest pain"', 'COMMENT="Fréquence élevée"')
    p = tmp_path / "latin1.xml"
    p.write_bytes(xml.encode("latin-1"))  # declared UTF-8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = MortaraEL250Parser().parse(p)
    assert record.annotations["comment"] == "Fréquence élevée"


def test_entity_declarations_refused(tmp_path: Path):
    xml = MORTARA_XML.replace("<ECG ", '<!DOCTYPE ECG [<!ENTITY e "x">]>\n<ECG ', 1)
    with pytest.raises(CorruptedFileError, match="entity"):
        _parse(tmp_path, xml)


def test_utf16_file_detected_and_parsed(tmp_path: Path):
    p = tmp_path / "utf16.xml"
    p.write_bytes(MORTARA_XML.replace('encoding="UTF-8"', 'encoding="UTF-16"').encode("utf-16"))
    assert MortaraEL250Parser.can_parse(p, p.read_bytes()[:4096]) is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert FileParser().parse(p).source_format == "mortara_el250"
