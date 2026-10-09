"""Tests for the GE MUSE XML parser."""

from __future__ import annotations

import base64
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import ChecksumWarning, CorruptedFileError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.ge_muse_xml import GEMuseXMLParser

from tests.conftest import GE_MUSE_XML, MUSE_RHYTHM_I, MUSE_RHYTHM_II, _muse_lead


def _parse(tmp_path: Path, xml: str, encoding: str = "iso-8859-1") -> ECGRecord:
    p = tmp_path / "mut.xml"
    p.write_bytes(xml.encode(encoding))
    return GEMuseXMLParser().parse(p)


RHYTHM_I_DATA = base64.b64encode(np.asarray(MUSE_RHYTHM_I, dtype="<i2").tobytes()).decode()
RHYTHM_START = GE_MUSE_XML.index("<WaveformType>Rhythm</WaveformType>")


def _mutate_rhythm(old: str, new: str) -> str:
    """Replace the first occurrence of *old* inside the rhythm waveform."""
    head, tail = GE_MUSE_XML[:RHYTHM_START], GE_MUSE_XML[RHYTHM_START:]
    assert old in tail
    return head + tail.replace(old, new, 1)


class TestGEMuseXMLParser:
    def test_parse_returns_ecg_record(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert isinstance(record, ECGRecord)
        assert record.source_format == "ge_muse_xml"

    def test_patient(self, ge_muse_xml_file: Path):
        p = GEMuseXMLParser().parse(ge_muse_xml_file).patient
        assert p.patient_id == "MUSE001"
        assert (p.first_name, p.last_name) == ("Jane", "Smith")
        assert p.sex == "F"
        assert p.birth_date == datetime(1985, 3, 15)
        assert p.age == 38
        assert p.race == "CAUCASIAN"
        assert (p.height, p.weight) == (165.0, 60.0)

    def test_recording(self, ge_muse_xml_file: Path):
        rec = GEMuseXMLParser().parse(ge_muse_xml_file).recording
        assert rec.date == datetime(2023, 12, 1, 14, 30)
        assert rec.location == "Cardiology"
        assert rec.room == "4B"
        assert rec.technician == "Alex Tech"
        assert rec.referring_physician == "Greg House"
        assert rec.device.model == "MAC5500"
        assert rec.device.software_version == "010A"
        assert rec.device.institution == "Main Hospital"
        assert rec.device.name == "7"

    def test_filters(self, ge_muse_xml_file: Path):
        fs = GEMuseXMLParser().parse(ge_muse_xml_file).recording.acquisition.filters
        assert fs.highpass == 0.16  # stored in 0.01 Hz
        assert fs.lowpass == 150.0
        assert fs.notch == 60.0
        assert fs.notch_active is True

    def test_signal(self, ge_muse_xml_file: Path):
        signal = GEMuseXMLParser().parse(ge_muse_xml_file).recording.acquisition.signal
        assert signal.sampling_rate == 500
        assert signal.resolution == 4.88
        assert signal.number_channels_allocated == 2
        assert signal.number_channels_valid == 2

    def test_only_stored_leads(self, ge_muse_xml_file: Path):
        leads = {lead.label: lead for lead in GEMuseXMLParser().parse(ge_muse_xml_file).leads}
        assert list(leads) == ["I", "II"]  # III, aVR, aVL, aVF are not computed
        np.testing.assert_array_equal(leads["I"].samples, MUSE_RHYTHM_I)
        np.testing.assert_array_equal(leads["II"].samples, MUSE_RHYTHM_II)

    def test_lead_scaling_fields(self, ge_muse_xml_file: Path):
        for lead in GEMuseXMLParser().parse(ge_muse_xml_file).leads:
            assert lead.resolution == 4.88
            assert lead.resolution_unit == "uV"
            assert lead.adc_resolution == 4.88
            assert lead.adc_resolution_unit == "MICROVOLTS"
            assert lead.is_raw is True
            assert lead.samples.dtype == np.float64

    def test_auto_scale(self, ge_muse_xml_file: Path):
        record = FileParser().parse(ge_muse_xml_file)
        assert record.leads[0].units == "mV"
        np.testing.assert_allclose(record.leads[0].samples, np.array(MUSE_RHYTHM_I) * 4.88e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = FileParser().parse(ge_muse_xml_file, auto_scale=False)
        np.testing.assert_array_equal(raw.leads[0].samples, MUSE_RHYTHM_I)

    def test_median_beats(self, ge_muse_xml_file: Path):
        beats = GEMuseXMLParser().parse(ge_muse_xml_file).median_beats
        assert [b.label for b in beats] == ["I", "II"]
        np.testing.assert_array_equal(beats[0].samples, [1, 2, 3])

    def test_measurements(self, ge_muse_xml_file: Path):
        m = GEMuseXMLParser().parse(ge_muse_xml_file).measurements
        assert m.heart_rate == 72
        assert m.pr_interval == 160
        assert m.qrs_duration == 88
        assert m.qt_interval == 380
        assert m.qtc_bazett is None  # QTCorrected does not state its formula
        assert m.qtc_fridericia == 404
        assert (m.p_axis, m.qrs_axis, m.t_axis) == (51, 20, 40)
        assert m.qrs_count == 12
        assert m.rr_interval == 833

    def test_unmapped_measurements_in_annotations(self, ge_muse_xml_file: Path):
        ann = GEMuseXMLParser().parse(ge_muse_xml_file).annotations
        assert ann["AtrialRate"] == "72"
        assert "VentricularRate" not in ann

    def test_interpretation_machine(self, ge_muse_xml_file: Path):
        interp = GEMuseXMLParser().parse(ge_muse_xml_file).interpretation
        assert interp.source == "machine"
        assert interp.statements == [("Normal sinus rhythm", ""), ("Normal ECG", "")]
        assert interp.severity == ""  # not inferred from statement text

    def test_file_format_and_checksum(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert record.file_format.version == "9.0.7.17363"
        assert record.raw_metadata["checksum_valid"] is True

    def test_to_dict_and_json(self, ge_muse_xml_file: Path):
        record = GEMuseXMLParser().parse(ge_muse_xml_file)
        assert set(record.to_dict()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats", "leads_enhanced",
            "annotations",
        }
        assert len(json.loads(record.to_json())["leads"]) == 2

    def test_does_not_match_sierra(self, tmp_path: Path):
        f = tmp_path / "sierra.xml"
        f.write_text('<?xml version="1.0"?><restingecgdata></restingecgdata>')
        assert GEMuseXMLParser.can_parse(f, f.read_bytes()) is False

    def test_auto_detection_via_file_parser(self, ge_muse_xml_file: Path):
        assert FileParser().parse(ge_muse_xml_file).source_format == "ge_muse_xml"


class TestMuseVariants:
    def test_confirmed_report(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("<Status>UNCONFIRMED</Status>", "<Status>CONFIRMED</Status>").replace(
            "</TestDemographics>",
            "<EditTime>09:15:00</EditTime><EditDate>12-02-2023</EditDate>"
            "<OverreaderLastName>Wilson</OverreaderLastName><OverreaderFirstName>James</OverreaderFirstName>"
            "</TestDemographics>",
        ).replace(
            "  <QRSTimesTypes>",
            "  <OriginalDiagnosis><Modality>RESTING</Modality><DiagnosisStatement><StmtFlag>ENDSLINE</StmtFlag>"
            "<StmtText>Sinus bradycardia</StmtText></DiagnosisStatement></OriginalDiagnosis>\n  <QRSTimesTypes>",
        )
        record = _parse(tmp_path, xml)
        interp = record.interpretation
        assert interp.source == "confirmed"
        assert interp.interpreter == "James Wilson"
        assert interp.interpretation_date == datetime(2023, 12, 2, 9, 15)
        assert interp.statements[0] == ("Normal sinus rhythm", "")
        assert record.annotations["machine_interpretation"] == "Sinus bradycardia"
        # EditDate is the last edit, not the file creation date
        assert record.file_format.creation_date is None
        assert record.raw_metadata["EditDate"] == "12-02-2023"

    def test_statement_fragments_joined_until_endsline(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace(
            "<StmtFlag>ENDSLINE</StmtFlag>\n      <StmtText>Normal sinus rhythm</StmtText>",
            "<StmtText>Sinus rhythm with</StmtText>\n    </DiagnosisStatement>\n    <DiagnosisStatement>\n"
            "      <StmtFlag>ENDSLINE</StmtFlag>\n      <StmtText>1st degree AV block</StmtText>",
        ).replace("<StmtText>Normal ECG</StmtText>", "<StmtText>Abnormal ECG</StmtText>")
        interp = _parse(tmp_path, xml).interpretation
        assert interp.statements == [("Sinus rhythm with 1st degree AV block", ""), ("Abnormal ECG", "")]

    def test_severity_only_from_summary_statement(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("<StmtText>Normal ECG</StmtText>",
                                  "<StmtText>When compared with ECG of 01-JAN-2020, normal sinus rhythm has replaced atrial fibrillation</StmtText>")
        assert _parse(tmp_path, xml).interpretation.severity == ""

    def test_current_measurements_preferred(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace(
            "  <Diagnosis>",
            "  <OriginalRestingECGMeasurements><VentricularRate>60</VentricularRate>"
            "<ECGSampleBase>500</ECGSampleBase><ECGSampleExponent>0</ECGSampleExponent>"
            "</OriginalRestingECGMeasurements>\n  <Diagnosis>",
        )
        record = _parse(tmp_path, xml)
        assert record.measurements.heart_rate == 72
        assert record.raw_metadata["original_measurements"]["VentricularRate"] == "60"

    def test_age_units_without_birth_date(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("<DateofBirth>03-15-1985</DateofBirth>", "").replace(
            "<PatientAge>38</PatientAge>\n    <AgeUnits>YEARS</AgeUnits>",
            "<PatientAge>30</PatientAge>\n    <AgeUnits>MONTHS</AgeUnits>")
        assert _parse(tmp_path, xml).patient.age == 2

    def test_imperial_height_weight(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("<HeightCM>165</HeightCM>", "<HeightIN>65</HeightIN>").replace(
            "<WeightKG>60</WeightKG>", "<WeightLBS>132</WeightLBS>")
        p = _parse(tmp_path, xml).patient
        assert p.height == 165.1
        assert p.weight == 59.9

    def test_date_without_time_is_not_invented(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("<AcquisitionTime>14:30:00</AcquisitionTime>", "")
        record = _parse(tmp_path, xml)
        assert record.recording.date is None
        assert record.raw_metadata["acquisition_date"] == "12-01-2023"

    def test_time_without_seconds(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("14:30:00", "14:30")
        assert _parse(tmp_path, xml).recording.date == datetime(2023, 12, 1, 14, 30)

    def test_millivolt_amplitude_units(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("<LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>",
                                  "<LeadAmplitudeUnitsPerBit>0.005</LeadAmplitudeUnitsPerBit>").replace(
            "<LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>", "<LeadAmplitudeUnits>MILLIVOLTS</LeadAmplitudeUnits>")
        lead = _parse(tmp_path, xml).leads[0]
        assert (lead.resolution, lead.resolution_unit, lead.adc_resolution_unit) == (0.005, "mV", "MILLIVOLTS")
        np.testing.assert_allclose(lead.to_physical().samples, np.array(MUSE_RHYTHM_I) * 0.005)

    def test_sample_exponent(self, tmp_path: Path):
        xml = _mutate_rhythm("<SampleBase>500</SampleBase>\n    <SampleExponent>0</SampleExponent>",
                             "<SampleBase>25</SampleBase>\n    <SampleExponent>1</SampleExponent>")
        assert _parse(tmp_path, xml).leads[0].sampling_rate == 250

    def test_missing_sample_base(self, tmp_path: Path):
        xml = _mutate_rhythm("<SampleBase>500</SampleBase>", "")
        with pytest.raises(CorruptedFileError, match="SampleBase"):
            _parse(tmp_path, xml)

    def test_ac_filter_none(self, tmp_path: Path):
        xml = _mutate_rhythm("<ACFilter>60</ACFilter>", "<ACFilter>NONE</ACFilter>")
        fs = _parse(tmp_path, xml).recording.acquisition.filters
        assert fs.notch is None
        assert fs.notch_active is False

    def test_pace_spikes(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace(
            "    </LeadData>\n  </Waveform>\n</RestingECG>",
            "    </LeadData>\n    <PaceSpikes><PaceType>VENTRICULAR</PaceType><IndexOffsets>10 510</IndexOffsets>"
            "</PaceSpikes>\n  </Waveform>\n</RestingECG>",
        )
        record = _parse(tmp_path, xml)
        # Detected spikes are reported, not turned into a patient attribute
        assert record.patient.has_pacemaker is None
        assert record.raw_metadata["pace_spikes"]["PaceType"] == "VENTRICULAR"

    def test_latin1_text(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("Smith", "Müller")
        assert _parse(tmp_path, xml).patient.last_name == "Müller"

    def test_duplicate_rhythm_waveform_warns(self, tmp_path: Path):
        rhythm = GE_MUSE_XML[GE_MUSE_XML.index("  <Waveform>\n    <WaveformType>Rhythm"):GE_MUSE_XML.index("</RestingECG>")]
        xml = GE_MUSE_XML.replace("</RestingECG>", rhythm.replace("<SampleBase>500", "<SampleBase>250") + "</RestingECG>")
        with pytest.warns(UserWarning, match="Additional rhythm waveform") as rec:
            record = _parse(tmp_path, xml)
        assert record.leads[0].sampling_rate == 500
        assert Path(rec[0].filename).name == "test_ge_muse_xml.py"

    def test_duplicate_lead_ids_made_unique(self, tmp_path: Path):
        xml = _mutate_rhythm("<LeadID>II</LeadID>", "<LeadID>V1</LeadID>").replace(
            "</RestingECG>", "</RestingECG>")
        xml = xml.replace(_muse_lead("I", MUSE_RHYTHM_I), _muse_lead("I", MUSE_RHYTHM_I) + _muse_lead("V1", [1, 2, 3, 4]))
        labels = [lead.label for lead in _parse(tmp_path, xml).leads]
        assert labels == ["I", "V1", "V1_2"]

    def test_edited_unconfirmed_statements_are_overread(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace(
            "  <QRSTimesTypes>",
            "  <OriginalDiagnosis><DiagnosisStatement><StmtText>Atrial fibrillation</StmtText>"
            "</DiagnosisStatement></OriginalDiagnosis>\n  <QRSTimesTypes>",
        )
        record = _parse(tmp_path, xml)
        assert record.interpretation.source == "overread"
        assert record.interpretation.statements[0] == ("Normal sinus rhythm", "")
        assert record.annotations["machine_interpretation"] == "Atrial fibrillation"
        assert "original_diagnosis" not in record.annotations

    def test_missing_amplitude_unit_is_not_assumed(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace("<LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>", "")
        with pytest.warns(UserWarning, match="LeadAmplitudeUnits missing"):
            record = _parse(tmp_path, xml)
        lead = record.leads[0]
        assert lead.resolution == 4.88
        assert lead.resolution_unit == ""
        assert lead.is_raw is True
        assert record.raw_metadata["amplitude_unit_stated"] is False

    def test_first_sample_baseline_reported_not_applied(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace(
            "<FirstSampleBaseline>0</FirstSampleBaseline>", "<FirstSampleBaseline>100</FirstSampleBaseline>")
        with pytest.warns(UserWarning, match="FirstSampleBaseline is non-zero"):
            record = _parse(tmp_path, xml)
        np.testing.assert_array_equal(record.leads[0].samples, MUSE_RHYTHM_I)
        assert record.leads[0].offset == 0.0
        assert record.raw_metadata["FirstSampleBaseline"]["I (rhythm)"] == "100"

    def test_lead_offset_first_sample_warns(self, tmp_path: Path):
        xml = GE_MUSE_XML.replace(
            "<LeadOffsetFirstSample>0</LeadOffsetFirstSample>", "<LeadOffsetFirstSample>250</LeadOffsetFirstSample>")
        with pytest.warns(UserWarning, match="LeadOffsetFirstSample is non-zero"):
            record = _parse(tmp_path, xml)
        assert record.raw_metadata["LeadOffsetFirstSample"]["I (rhythm)"] == "250"

    @pytest.mark.parametrize("units", ["<AgeUnits>DECADES</AgeUnits>", ""])
    def test_age_in_unknown_unit_kept_and_dob_not_used(self, tmp_path: Path, units: str):
        xml = GE_MUSE_XML.replace(
            "<PatientAge>38</PatientAge>\n    <AgeUnits>YEARS</AgeUnits>",
            f"<PatientAge>7</PatientAge>\n    {units}")
        record = _parse(tmp_path, xml)
        assert record.patient.age is None
        assert record.raw_metadata["age"] == {"value": "7", "units": units[10:-11]}


class TestMuseCorruption:
    def test_invalid_base64(self, tmp_path: Path):
        xml = _mutate_rhythm(RHYTHM_I_DATA, "Z@@@")
        with pytest.raises(CorruptedFileError, match="Base64"):
            _parse(tmp_path, xml)

    def test_odd_byte_count(self, tmp_path: Path):
        odd = base64.b64encode(b"\x01\x02\x03").decode()
        xml = _mutate_rhythm(RHYTHM_I_DATA, odd)
        with pytest.raises(CorruptedFileError, match="whole number"):
            _parse(tmp_path, xml)

    def test_byte_count_mismatch(self, tmp_path: Path):
        xml = _mutate_rhythm("<LeadByteCountTotal>8</LeadByteCountTotal>", "<LeadByteCountTotal>10000</LeadByteCountTotal>")
        with pytest.raises(CorruptedFileError, match="LeadByteCountTotal"):
            _parse(tmp_path, xml)

    def test_sample_count_mismatch(self, tmp_path: Path):
        xml = _mutate_rhythm("<LeadSampleCountTotal>4</LeadSampleCountTotal>", "<LeadSampleCountTotal>5</LeadSampleCountTotal>")
        with pytest.raises(CorruptedFileError, match="LeadSampleCountTotal"):
            _parse(tmp_path, xml)

    def test_crc_mismatch_warns(self, tmp_path: Path):
        head, tail = GE_MUSE_XML[:RHYTHM_START], GE_MUSE_XML[RHYTHM_START:]
        start = tail.index("<LeadDataCRC32>") + len("<LeadDataCRC32>")
        end = tail.index("</LeadDataCRC32>")
        xml = head + tail[:start] + "12345" + tail[end:]
        with pytest.warns(ChecksumWarning, match="CRC32"):
            record = _parse(tmp_path, xml)
        assert record.raw_metadata["checksum_valid"] is False

    def test_malformed_xml(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError, match="Malformed XML"):
            _parse(tmp_path, GE_MUSE_XML[:-30])

    def test_no_rhythm(self, tmp_path: Path):
        xml = GE_MUSE_XML[:GE_MUSE_XML.index("  <Waveform>\n    <WaveformType>Rhythm")] + "</RestingECG>\n"
        with pytest.raises(CorruptedFileError, match="No rhythm"):
            _parse(tmp_path, xml)


def test_entity_declarations_refused(tmp_path: Path):
    xml = GE_MUSE_XML.replace("<RestingECG>", '<!DOCTYPE RestingECG [<!ENTITY e "x">]>\n<RestingECG>', 1)
    with pytest.raises(CorruptedFileError, match="entity"):
        _parse(tmp_path, xml)


def test_utf16_file_detected_and_parsed(tmp_path: Path):
    p = tmp_path / "utf16.xml"
    p.write_bytes(GE_MUSE_XML.replace('encoding="ISO-8859-1"', 'encoding="UTF-16"').encode("utf-16"))
    assert GEMuseXMLParser.can_parse(p, p.read_bytes()[:4096]) is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = FileParser().parse(p)
    assert record.source_format == "ge_muse_xml"
    assert record.patient.last_name == "Smith"
