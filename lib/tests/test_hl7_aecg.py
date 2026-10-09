"""Tests for the HL7 aECG parser."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import ECGRecord
from ecgdatakit.parsing.parser import FileParser
from ecgdatakit.parsing.parsers.hl7_aecg import HL7aECGParser, _parse_ts

from tests.conftest import HL7_AECG_XML


def _write(tmp_path: Path, xml: str, name: str = "mut.xml", encoding: str = "utf-8") -> Path:
    p = tmp_path / name
    p.write_bytes(xml.encode(encoding))
    return p


def _parse(tmp_path: Path, xml: str, **kw) -> ECGRecord:
    return HL7aECGParser().parse(_write(tmp_path, xml, **kw))


LEAD_II_BLOCK = HL7_AECG_XML[
    HL7_AECG_XML.index("          <component>\n            <sequence>\n              <code code=\"MDC_ECG_LEAD_II\""):
    HL7_AECG_XML.index("        </sequenceSet>\n      </component>\n      <derivation>")
]


def _rhythm_lead(code: str, digits: str = "1 2 3 4 5", scale: str = '<scale value="5" unit="uV"/>') -> str:
    return (
        "          <component><sequence>"
        f'<code code="{code}"/><value><origin value="0" unit="uV"/>{scale}'
        f"<digits>{digits}</digits></value></sequence></component>\n"
    )


def _with_rhythm_leads(*blocks: str) -> str:
    return HL7_AECG_XML.replace(LEAD_II_BLOCK, "".join(blocks))


class TestHL7aECGParser:
    def test_parse_returns_ecg_record(self, hl7_aecg_file: Path):
        assert isinstance(HL7aECGParser().parse(hl7_aecg_file), ECGRecord)

    def test_source_format(self, hl7_aecg_file: Path):
        assert HL7aECGParser().parse(hl7_aecg_file).source_format == "hl7_aecg"

    def test_patient(self, hl7_aecg_file: Path):
        p = HL7aECGParser().parse(hl7_aecg_file).patient
        assert p.patient_id == "SUBJ-001"
        assert p.first_name == "Ann Marie"
        assert p.last_name == "Doe"
        assert p.sex == "M"
        assert p.birth_date == datetime(1980, 1, 1)
        assert p.race == "White"
        assert p.age == 43

    def test_recording_dates(self, hl7_aecg_file: Path):
        rec = HL7aECGParser().parse(hl7_aecg_file).recording
        assert rec.date == datetime(2023, 6, 15, 10, 30, 0)
        assert rec.end_date == datetime(2023, 6, 15, 10, 30, 10)
        assert rec.duration.total_seconds() == 10.0

    def test_device_and_people(self, hl7_aecg_file: Path):
        rec = HL7aECGParser().parse(hl7_aecg_file).recording
        assert rec.device.model == "ELI250"
        assert rec.device.software_version == "5.3"
        assert rec.device.manufacturer == "Acme Medical"
        assert rec.device.serial_number == "SN-42"
        assert rec.device.institution == "Test Clinic"
        assert rec.technician == "KAB"

    def test_filters(self, hl7_aecg_file: Path):
        fs = HL7aECGParser().parse(hl7_aecg_file).recording.acquisition.filters
        assert fs.lowpass == 150.0
        assert fs.highpass == 0.05
        assert fs.notch == 50.0
        assert fs.notch_active is True

    def test_leads(self, hl7_aecg_file: Path):
        record = HL7aECGParser().parse(hl7_aecg_file)
        assert [lead.label for lead in record.leads] == ["I", "II"]
        np.testing.assert_array_equal(record.leads[0].samples, [100, 200, 300, 400, 500])
        # digits split over a newline and a tab
        np.testing.assert_array_equal(record.leads[1].samples, [110, 210, 310, 410, 510])

    def test_lead_scaling_fields(self, hl7_aecg_file: Path):
        for lead in HL7aECGParser().parse(hl7_aecg_file).leads:
            assert lead.sampling_rate == 500
            assert lead.resolution == 5.0
            assert lead.resolution_unit == "uV"
            assert lead.offset == 0.0
            assert lead.adc_resolution == 5.0
            assert lead.adc_resolution_unit == "uV"
            assert lead.is_raw is True
            assert lead.units == ""

    def test_signal_summary(self, hl7_aecg_file: Path):
        signal = HL7aECGParser().parse(hl7_aecg_file).recording.acquisition.signal
        assert signal.sampling_rate == 500
        assert signal.resolution == 5.0

    def test_median_beats(self, hl7_aecg_file: Path):
        beats = HL7aECGParser().parse(hl7_aecg_file).median_beats
        assert [b.label for b in beats] == ["I", "II"]
        assert beats[0].sampling_rate == 500  # TIME_RELATIVE increment 2 ms
        np.testing.assert_array_equal(beats[1].samples, [4, 5, 6])

    def test_measurements_unit_conversion(self, hl7_aecg_file: Path):
        m = HL7aECGParser().parse(hl7_aecg_file).measurements
        assert m.heart_rate == 72
        assert m.pr_interval == 160
        assert m.qt_interval == 412  # 0.412 s

    def test_interpretation(self, hl7_aecg_file: Path):
        interp = HL7aECGParser().parse(hl7_aecg_file).interpretation
        assert interp.statements == [("Sinus rhythm", "")]
        assert interp.severity == "NORMAL"
        assert interp.source == "machine"
        assert interp.interpretation_date == datetime(2023, 6, 15, 10, 30, 10)

    def test_auto_scale(self, hl7_aecg_file: Path):
        record = FileParser().parse(hl7_aecg_file)
        assert record.source_format == "hl7_aecg"
        assert record.leads[0].units == "mV"
        np.testing.assert_allclose(record.leads[0].samples, [0.5, 1.0, 1.5, 2.0, 2.5])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = FileParser().parse(hl7_aecg_file, auto_scale=False)
        np.testing.assert_array_equal(raw.leads[0].samples, [100, 200, 300, 400, 500])

    def test_to_dict_unified_schema(self, hl7_aecg_file: Path):
        d = HL7aECGParser().parse(hl7_aecg_file).to_dict()
        assert set(d.keys()) == {
            "source_format", "file_format", "patient", "recording",
            "leads", "interpretation", "measurements", "median_beats",
            "annotations",
        }

    def test_to_json_roundtrip(self, hl7_aecg_file: Path):
        parsed = json.loads(HL7aECGParser().parse(hl7_aecg_file).to_json())
        assert parsed["source_format"] == "hl7_aecg"
        assert len(parsed["leads"]) == 2

    def test_can_parse_prefixed_root(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace(
            '<AnnotatedECG xmlns="urn:hl7-org:v3"', '<hl7:AnnotatedECG xmlns:hl7="urn:hl7-org:v3"'
        ).replace("</AnnotatedECG>", "</hl7:AnnotatedECG>")
        # Prefix only the root; children stay in no namespace, which is fine after stripping
        p = _write(tmp_path, xml)
        assert HL7aECGParser.can_parse(p, p.read_bytes()[:4096])
        assert len(HL7aECGParser().parse(p).leads) == 2


class TestTimeBase:
    def test_time_relative_ms(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace('code="TIME_ABSOLUTE"', 'code="TIME_RELATIVE"').replace(
            '<head value="20230615103000.000"/>\n                <increment value="0.002" unit="s"/>',
            '<head value="0" unit="ms"/>\n                <increment value="4" unit="ms"/>',
        )
        record = _parse(tmp_path, xml)
        assert record.recording.acquisition.signal.sampling_rate == 250
        assert all(lead.sampling_rate == 250 for lead in record.leads)

    def test_non_integer_rate_warns(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace('value="0.002" unit="s"', 'value="0.0039" unit="s"')
        with pytest.warns(UserWarning, match="not an integer"):
            record = _parse(tmp_path, xml)
        assert record.leads[0].sampling_rate == 256

    def test_missing_increment_is_corrupt(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace('<increment value="0.002" unit="s"/>', "")
        with pytest.raises(CorruptedFileError, match="time increment"):
            _parse(tmp_path, xml)

    def test_absolute_head_used_when_no_effective_time(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace(
            '<effectiveTime>\n    <low value="20230615103000"/>\n    <high value="20230615103010"/>\n  </effectiveTime>', ""
        ).replace(
            '<effectiveTime>\n        <low value="20230615103000"/>\n        <high value="20230615103010"/>\n      </effectiveTime>', ""
        )
        rec = _parse(tmp_path, xml).recording
        assert rec.date == datetime(2023, 6, 15, 10, 30, 0)
        assert rec.duration == timedelta(seconds=5 * 0.002)


class TestDates:
    def test_ts_formats(self):
        assert _parse_ts("20230615103000.250") == datetime(2023, 6, 15, 10, 30, 0, 250000)
        assert _parse_ts("202306151030") == datetime(2023, 6, 15, 10, 30)
        assert _parse_ts("20230615103000-0500") == datetime(
            2023, 6, 15, 10, 30, tzinfo=timezone(timedelta(hours=-5)))
        assert _parse_ts("20230615", need_time=True) is None
        # Hour-only precision: minutes and seconds are not filled in
        assert _parse_ts("2023061510", need_time=True) is None
        assert _parse_ts("2023061510") == datetime(2023, 6, 15, 10)
        assert _parse_ts("2023") is None
        assert _parse_ts("garbage") is None

    def test_center_effective_time(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace(
            '<low value="20230615103000"/>\n    <high value="20230615103010"/>',
            '<center value="20230615111500"/>', 1,
        )
        assert _parse(tmp_path, xml).recording.date == datetime(2023, 6, 15, 11, 15)

    def test_date_only_effective_time_invents_no_time(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace('value="20230615103000"', 'value="20230615"').replace(
            'value="20230615103010"', 'value="20230615"').replace(
            '<head value="20230615103000.000"/>', '<head value="20230615"/>')
        record = _parse(tmp_path, xml)
        assert record.recording.date is None
        assert record.raw_metadata["effective_time"] == "20230615"


class TestLeads:
    def test_lead_code_variants(self, tmp_path: Path):
        xml = _with_rhythm_leads(
            _rhythm_lead("MDC_ECG_LEAD_aVR"),
            _rhythm_lead("MDC_ECG_LEAD_AVL"),
            _rhythm_lead("MDC_ECG_LEAD_AVRneg"),
            _rhythm_lead("MDC_ECG_LEAD_V4R"),
            _rhythm_lead("MORTARA_ECG_LEAD_TEA"),
        )
        labels = [lead.label for lead in _parse(tmp_path, xml).leads]
        assert labels == ["I", "aVR", "aVL", "-aVR", "V4R", "TEA"]

    def test_duplicate_labels_made_unique(self, tmp_path: Path):
        xml = _with_rhythm_leads(_rhythm_lead("MDC_ECG_LEAD_I"))
        assert [lead.label for lead in _parse(tmp_path, xml).leads] == ["I", "I_2"]

    def test_unknown_sequence_warns(self, tmp_path: Path):
        xml = _with_rhythm_leads(_rhythm_lead("SOMETHING_ELSE"))
        with pytest.warns(UserWarning, match="SOMETHING_ELSE") as rec:
            record = _parse(tmp_path, xml)
        assert [lead.label for lead in record.leads] == ["I"]
        assert Path(rec[0].filename).name == "test_hl7_aecg.py"

    def test_non_numeric_digits(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace("100 200 300 400 500", "100 abc")
        with pytest.raises(CorruptedFileError, match="Invalid digits"):
            _parse(tmp_path, xml)

    def test_empty_digits(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace("100 200 300 400 500", " ")
        with pytest.raises(CorruptedFileError, match="no digits"):
            _parse(tmp_path, xml)

    def test_decimal_digits_kept(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace("100 200 300 400 500", "1.5 -2.25")
        np.testing.assert_array_equal(_parse(tmp_path, xml).leads[0].samples, [1.5, -2.25])

    def test_origin_unit_converted_to_scale_unit(self, tmp_path: Path):
        xml = _with_rhythm_leads(_rhythm_lead("MDC_ECG_LEAD_II", "0 2", '<scale value="5" unit="uV"/>').replace(
            '<origin value="0" unit="uV"/>', '<origin value="1" unit="mV"/>'))
        lead = _parse(tmp_path, xml).leads[1]
        assert lead.offset == 1000.0
        np.testing.assert_allclose(lead.to_physical().samples, [1000.0, 1010.0])

    def test_scale_without_unit_stays_raw(self, tmp_path: Path):
        xml = _with_rhythm_leads(_rhythm_lead("MDC_ECG_LEAD_II", scale='<scale value="5"/>'))
        lead = _parse(tmp_path, xml).leads[1]
        assert lead.resolution == 5.0
        assert lead.resolution_unit == ""
        assert lead.is_raw is True

    def test_scale_in_mv(self, tmp_path: Path):
        xml = _with_rhythm_leads(_rhythm_lead("MDC_ECG_LEAD_II", "2", '<scale value="0.005" unit="mV"/>'))
        lead = FileParser().parse(_write(tmp_path, xml)).leads[1]
        np.testing.assert_allclose(lead.samples, [0.01])

    def test_multiple_rhythm_series_warn(self, tmp_path: Path):
        extra = (
            "  <component><series><code code=\"RHYTHM\"/><component><sequenceSet>"
            "<component><sequence><code code=\"TIME_RELATIVE\"/><value><increment value=\"0.001\" unit=\"s\"/></value></sequence></component>"
            "<component><sequence><code code=\"MDC_ECG_LEAD_V1\"/><value><digits>1 2</digits></value></sequence></component>"
            "</sequenceSet></component></series></component>\n</AnnotatedECG>"
        )
        xml = HL7_AECG_XML.replace("</AnnotatedECG>", extra)
        with pytest.warns(UserWarning, match="2 rhythm series"):
            record = _parse(tmp_path, xml)
        assert [lead.label for lead in record.leads] == ["I", "II"]

    def test_multiple_sequence_sets_warn(self, tmp_path: Path):
        extra = (
            "      <component><sequenceSet><component><sequence><code code=\"MDC_ECG_LEAD_V1\"/>"
            "<value><digits>1</digits></value></sequence></component></sequenceSet></component>\n"
            "      <derivation>"
        )
        xml = HL7_AECG_XML.replace("      <derivation>", extra, 1)
        with pytest.warns(UserWarning, match="2 sequence sets"):
            record = _parse(tmp_path, xml)
        assert [lead.label for lead in record.leads] == ["I", "II"]


class TestErrors:
    def test_malformed_xml(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError, match="Malformed XML"):
            _parse(tmp_path, HL7_AECG_XML[:-40])

    def test_wrong_root(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError, match="AnnotatedECG"):
            _parse(tmp_path, "<?xml version='1.0'?><Other/>")

    def test_no_series(self, tmp_path: Path):
        xml = HL7_AECG_XML[:HL7_AECG_XML.index("  <component>\n    <series>")] + "</AnnotatedECG>\n"
        with pytest.raises(CorruptedFileError, match="rhythm series"):
            _parse(tmp_path, xml)

    def test_file_parser_wraps(self, tmp_path: Path):
        with pytest.raises(CorruptedFileError):
            FileParser().parse(_write(tmp_path, HL7_AECG_XML[:-40]))


class TestEncoding:
    def test_declared_latin1(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace('encoding="UTF-8"', 'encoding="ISO-8859-1"').replace(
            "<family>Doe</family>", "<family>Müller</family>")
        assert _parse(tmp_path, xml, encoding="latin-1").patient.last_name == "Müller"

    def test_undeclared_latin1_keeps_characters(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace("<family>Doe</family>", "<family>Müller</family>")
        assert _parse(tmp_path, xml, encoding="latin-1").patient.last_name == "Müller"

    def test_name_parts_with_attributes(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace(
            "<name><given>Ann</given><given>Marie</given><family>Doe</family></name>",
            '<name><prefix>Dr</prefix><given partType="GIV">Ann</given><family partType="FAM">Doe</family>'
            '<family>Smith</family></name>')
        p = _parse(tmp_path, xml).patient
        assert p.first_name == "Ann"
        assert p.last_name == "Doe Smith"

    def test_plain_text_name(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace(
            "<name><given>Ann</given><given>Marie</given><family>Doe</family></name>", "<name>JD</name>")
        p = _parse(tmp_path, xml).patient
        assert (p.first_name, p.last_name) == ("", "JD")


ANN_SET = HL7_AECG_XML[HL7_AECG_XML.index("      <subjectOf>"):HL7_AECG_XML.index("    </series>")]


def _annotation(code: str, value: str = "", children: str = "", support: str = "") -> str:
    return (f"<component><annotation><code code=\"{code}\"/>{value}{support}{children}"
            "</annotation></component>")


def _with_annotation_sets(*sets: str) -> str:
    return HL7_AECG_XML.replace(ANN_SET, "".join(f"      <subjectOf>{s}</subjectOf>\n" for s in sets))


PERSON_AUTHOR = (
    "<author><assignedEntity><id extension=\"DR-1\"/><assignedAuthorType><assignedPerson>"
    "<name><given>Greg</given><family>House</family></name></assignedPerson></assignedAuthorType>"
    "</assignedEntity></author>"
)


class TestAnnotations:
    def test_expert_overread_and_machine(self, tmp_path: Path):
        machine = "<annotationSet>" + _annotation(
            "MDC_ECG_INTERPRETATION", children=_annotation(
                "MDC_ECG_INTERPRETATION_STATEMENT", "<value>Sinus rhythm</value>")) + "</annotationSet>"
        expert = (
            "<annotationSet><activityTime value=\"20230616090000\"/>" + PERSON_AUTHOR
            + _annotation("MDC_ECG_INTERPRETATION", children=_annotation(
                "MDC_ECG_INTERPRETATION_STATEMENT", "<value>Sinus tachycardia</value>")
                + _annotation("MDC_ECG_INTERPRETATION_SEVERITY", '<value code="ABNORMAL"/>'))
            + _annotation("MDC_ECG_TIME_PD_QT", '<value value="400" unit="ms"/>')
            + "</annotationSet>"
        )
        record = _parse(tmp_path, _with_annotation_sets(machine, expert))
        interp = record.interpretation
        assert interp.source == "overread"
        assert interp.statements == [("Sinus tachycardia", "")]
        assert interp.severity == "ABNORMAL"
        assert interp.interpreter == "Greg House"
        assert interp.interpretation_date == datetime(2023, 6, 16, 9, 0)
        assert record.annotations["machine_interpretation"] == "Sinus rhythm"
        assert record.measurements.qt_interval == 400

    def test_no_duplicate_statements_or_severity_text(self, hl7_aecg_file: Path):
        interp = HL7aECGParser().parse(hl7_aecg_file).interpretation
        assert len(interp.statements) == 1

    def test_summary_and_comment(self, tmp_path: Path):
        aset = "<annotationSet>" + _annotation("MDC_ECG_INTERPRETATION", children=(
            _annotation("MDC_ECG_INTERPRETATION_STATEMENT", "<value>Rythme sinusal</value>")
            + _annotation("MDC_ECG_INTERPRETATION_SUMMARY", "<value>ECG normal</value>")
            + _annotation("MDC_ECG_INTERPRETATION_COMMENT", "<value>Diagnostic non confirmé.</value>")
        )) + "</annotationSet>"
        record = _parse(tmp_path, _with_annotation_sets(aset))
        assert record.interpretation.statements == [("Rythme sinusal", ""), ("ECG normal", "")]
        assert record.interpretation.severity == ""
        assert record.annotations["interpretation_comment"] == "Diagnostic non confirmé."

    def test_lead_specific_values_do_not_override_globals(self, tmp_path: Path):
        lead_roi = ("<support><supportingROI><component><boundary><code code=\"MDC_ECG_LEAD_V1\"/>"
                    "</boundary></component></supportingROI></support>")
        aset = ("<annotationSet>"
                + _annotation("MDC_ECG_TIME_PD_QRS", '<value value="90" unit="ms"/>')
                + _annotation("VENDOR_MATRIX", support=lead_roi, children=_annotation(
                    "MDC_ECG_TIME_PD_QRS", '<value value="120" unit="ms"/>'))
                + "</annotationSet>")
        record = _parse(tmp_path, _with_annotation_sets(aset))
        assert record.measurements.qrs_duration == 90
        assert record.raw_metadata["lead_measurements"]["V1"]["MDC_ECG_TIME_PD_QRS"] == "120 ms"

    def test_beat_values_and_pacing(self, tmp_path: Path):
        beats = "".join(
            _annotation("MDC_ECG_BEAT", '<value code="MDC_ECG_BEAT_PACED_V"/>',
                        children=_annotation("MDC_ECG_TIME_PD_QT", f'<value value="{qt}" unit="ms"/>'))
            for qt in (400, 410, 420)
        )
        record = _parse(tmp_path, _with_annotation_sets(f"<annotationSet>{beats}</annotationSet>"))
        # Per-beat values are kept as stored, no median and no pacemaker inference
        assert record.measurements.qt_interval is None
        assert record.raw_metadata["beat_measurements"]["qt_interval"] == [400, 410, 420]
        assert record.patient.has_pacemaker is None

    def test_intervals_from_representative_beat_boundaries(self, tmp_path: Path):
        def wave(code, low=None, high=None):
            edges = "".join(f'<{k} value="{v}" unit="ms"/>' for k, v in (("low", low), ("high", high)) if v)
            return _annotation("MDC_ECG_WAVC", f'<value code="{code}"/>', support=(
                "<support><supportingROI><component><boundary><code code=\"TIME_RELATIVE\"/>"
                f"<value>{edges}</value></boundary></component></supportingROI></support>"))
        aset = ("<annotationSet>" + wave("MDC_ECG_WAVC_PWAVE", 300)
                + wave("MDC_ECG_WAVC_QRSWAVE", 460, 545) + wave("MDC_ECG_WAVC_TWAVE", high=890)
                + "</annotationSet>")
        xml = HL7_AECG_XML.replace(ANN_SET, "").replace(
            "        </derivedSeries>", f"          <subjectOf>{aset}</subjectOf>\n        </derivedSeries>")
        record = _parse(tmp_path, xml)
        m = record.measurements
        # Wave boundaries are reported, intervals are not computed from them
        assert (m.pr_interval, m.qrs_duration, m.qt_interval) == (None, None, None)
        assert record.annotations["wave_boundaries_ms"] == (
            "p_onset=300, qrs_onset=460, qrs_offset=545, t_offset=890")

    def test_unknown_measurement_unit_warns(self, tmp_path: Path):
        aset = "<annotationSet>" + _annotation("MDC_ECG_TIME_PD_QT", '<value value="4" unit="furlong"/>') + "</annotationSet>"
        with pytest.warns(UserWarning, match="furlong"):
            record = _parse(tmp_path, _with_annotation_sets(aset))
        assert record.measurements.qt_interval is None


class TestContext:
    def test_reported_age_without_birth_date(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace('<birthTime value="19800101"/>', "").replace(
            "  <component>\n    <series>",
            "  <controlVariable><relatedObservation><code code=\"21612-7\" codeSystem=\"2.16.840.1.113883.6.1\"/>"
            "<value value=\"18\" unit=\"mo\"/></relatedObservation></controlVariable>\n  <component>\n    <series>", 1)
        assert _parse(tmp_path, xml).patient.age == 1

    def test_creation_date_from_writer_comment(self, tmp_path: Path):
        xml = HL7_AECG_XML.replace(
            '<?xml version="1.0" encoding="UTF-8"?>\n',
            '<?xml version="1.0" encoding="UTF-8"?>\n<!-- File created on 06/16/2023 07:58:38 -->\n')
        assert _parse(tmp_path, xml).file_format.creation_date.isoformat() == "2023-06-16"

    def test_clinical_trial_metadata(self, hl7_aecg_file: Path):
        meta = HL7aECGParser().parse(hl7_aecg_file).raw_metadata
        assert meta["trial_id"] == "TRIAL-1"
        assert meta["trial_site_id"] == "SITE-9"
        assert meta["visit"] == "VISIT_1"


def test_official_hl7_example_structure(tmp_path: Path):
    """Elements laid out as in the HL7 2003-12 example (center time, ITSVersion)."""
    xml = HL7_AECG_XML.replace(
        '<AnnotatedECG xmlns="urn:hl7-org:v3"', '<AnnotatedECG ITSVersion="XML_1.0" xmlns="urn:hl7-org:v3"')
    assert _parse(tmp_path, xml).file_format.version == "XML_1.0"


def test_unknown_time_increment_unit_raises(tmp_path: Path):
    # The sampling rate is never guessed from an unknown unit
    xml = HL7_AECG_XML.replace('<increment value="0.002" unit="s"/>', '<increment value="0.002" unit="tick"/>')
    with pytest.raises(CorruptedFileError, match="time increment unit"):
        _parse(tmp_path, xml)


@pytest.mark.parametrize("value, expected", [
    ('<value xsi:type="CE" code="NORMAL" displayName="Normal ECG"/>', "NORMAL"),
    ("<value>Otherwise normal ECG</value>", "OTHERWISE NORMAL ECG"),
    ("<value>Not normal</value>", "NOT NORMAL"),
])
def test_severity_reported_as_stored(tmp_path: Path, value: str, expected: str):
    xml = HL7_AECG_XML.replace(
        '<value xsi:type="CE" code="NORMAL" displayName="Normal ECG"/>', value)
    assert _parse(tmp_path, xml).interpretation.severity == expected


def test_entity_declarations_refused(tmp_path: Path):
    xml = HL7_AECG_XML.replace(
        '<AnnotatedECG', '<!DOCTYPE x [<!ENTITY e "hello">]>\n<AnnotatedECG', 1)
    with pytest.raises(CorruptedFileError, match="entity"):
        _parse(tmp_path, xml)


def test_effective_time_with_mixed_timezones(tmp_path: Path):
    # low has an offset, high has none: they cannot be compared, the duration
    # comes from the samples instead of failing
    xml = HL7_AECG_XML.replace('<low value="20230615103000"/>', '<low value="20230615103000+0200"/>', 1)
    record = _parse(tmp_path, xml)
    assert record.recording.date.utcoffset() == timedelta(hours=2)
    assert record.recording.duration is not None
