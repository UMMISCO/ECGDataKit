"""Tests for the Philips Sierra ECG XML parser.

Files are built here from the format description: XLI chunks are encoded
with an LZW (10-bit codes) and second-order delta encoder that mirrors the
reference decoder (sierra-ecg-tools), and leads III, aVR, aVL, aVF are
stored as residuals the way PageWriter devices write them.
"""

from __future__ import annotations

import base64
import struct
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ecgdatakit import FileParser
from ecgdatakit.exceptions import (
    CorruptedFileError,
    MissingElementError,
    UnsupportedFormatError,
)
from ecgdatakit.parsing.codecs.xli import xli_decode
from ecgdatakit.parsing.parsers.sierra_xml import SierraXMLParser

LABELS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
DURATION_MS = 2000
N = FS * DURATION_MS // 1000


# ── encoders ─────────────────────────────────────────────────────────

def lzw_encode(data: bytes, bits: int = 10) -> bytes:
    max_code = (1 << bits) - 2
    table = {bytes([i]): i for i in range(256)}
    next_code = 256
    codes: list[int] = []
    w = b""
    for byte in data:
        wc = w + bytes([byte])
        if wc in table:
            w = wc
            continue
        codes.append(table[w])
        if next_code <= max_code:
            table[wc] = next_code
            next_code += 1
        w = bytes([byte])
    if w:
        codes.append(table[w])
    acc = 0
    nbits = 0
    out = bytearray()
    for code in codes:
        acc = (acc << bits) | code
        nbits += bits
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def xli_chunk(samples: np.ndarray) -> bytes:
    s = [int(v) for v in samples]
    n = len(s)
    deltas = [0] * n
    deltas[0], deltas[1] = s[0], s[1]
    start = 2 * s[1] - s[0] - s[2]
    for i in range(3, n):
        deltas[i - 1] = 2 * s[i - 1] - s[i - 2] - s[i] + 64
    deltas[n - 1] = 64
    words = [d & 0xFFFF for d in deltas]
    body = bytes(w >> 8 for w in words) + bytes(w & 0xFF for w in words)
    payload = lzw_encode(body)
    return struct.pack("<ihh", len(payload), 1, start) + payload


def xli_encode(leads: list[np.ndarray]) -> bytes:
    return b"".join(xli_chunk(lead) for lead in leads)


def true_leads(n: int = N) -> list[np.ndarray]:
    t = np.arange(n)
    rng = np.random.default_rng(7)
    out = []
    for k in range(12):
        wave = 300 * np.sin(2 * np.pi * (k + 1) * t / n) + rng.integers(-20, 21, n)
        out.append(np.round(wave).astype(np.int64))
    return out


def stored_residuals(leads: list[np.ndarray]) -> list[np.ndarray]:
    """Limb leads as stored in XLI data (inverse of the device reconstruction)."""
    i, ii, iii, avr, avl, avf = leads[:6]
    stored = [i, ii, ii - i - iii, -avr - (i + ii) // 2, (i - iii) // 2 - avl, (ii + iii) // 2 - avf]
    return stored + leads[6:]


# ── document builders ───────────────────────────────────────────────

PATIENT_104 = """
  <patient>
    <generalpatientdata>
      <patientid>P-001</patientid>
      <MRN>M-77</MRN>
      <name><lastname>Doe</lastname><firstname>Jérôme</firstname><middlename>---</middlename></name>
      <age defaultage="50"><dateofbirth>1980-06-15</dateofbirth></age>
      <pacestatus>Unknown</pacestatus>
      <sex>Male</sex>
      <race id="8" code="White">White</race>
      <height><inch>70</inch></height>
      <weight><lb>154</lb></weight>
    </generalpatientdata>
  </patient>"""

INTERP_CONFIRMED = """
  <interpretations>
    <interpretation date="2020-03-01" time="10:15:30" criteriaversion="0A">
      <interpretationdatastructure>
        <statementcomponents>
          <codedstatement source="Analysis program" deleted="False">
            <modifiers/>
            <variables><numericvalue ndigits="3">50</numericvalue><numericvalue ndigits="3">99</numericvalue></variables>
            <unparsedstatement code="SR"><lhsstatement>SINUS RHYTHM</lhsstatement><rhsstatement>V-rate ***-***</rhsstatement></unparsedstatement>
          </codedstatement>
          <codedstatement source="Analysis program" deleted="True">
            <modifiers><modifier>Possible</modifier></modifiers>
            <variables/>
            <unparsedstatement code="LAE"><lhsstatement>LEFT ATRIAL ENLARGEMENT</lhsstatement><rhsstatement/></unparsedstatement>
          </codedstatement>
          <uncodedstatement source="Editor">Compared with prior</uncodedstatement>
        </statementcomponents>
      </interpretationdatastructure>
      <globalmeasurements editedflag="False">
        <heartrate editedflag="False">72</heartrate>
        <rrint editedflag="False">833</rrint>
        <print editedflag="False">160</print>
        <qrsdur editedflag="False">90</qrsdur>
        <qtint editedflag="False">380</qtint>
        <qtcb editedflag="False">416</qtcb>
        <qtcf editedflag="False">404</qtcf>
        <pfrontaxis editedflag="False">45</pfrontaxis>
        <qrsfrontaxis editedflag="False">Indeterminate</qrsfrontaxis>
        <tfrontaxis editedflag="False">30</tfrontaxis>
        <phorizaxis editedflag="False">12</phorizaxis>
      </globalmeasurements>
      <mdsignatureline>Confirmed by: Smith, Ann 02-Mar-2020 08:00:00</mdsignatureline>
      <confirmingclinician date="2020-03-02" time="08:00:00" id="7">Smith, Ann</confirmingclinician>
      <severity code="AB" id="4">- ABNORMAL ECG -</severity>
      <statement editedflag="False"><statementcode>SR</statementcode><leftstatement>SINUS RHYTHM</leftstatement><rightstatement>V-rate 50-99</rightstatement></statement>
      <statement editedflag="True"><statementcode/><leftstatement>Compared with prior</leftstatement><rightstatement>
      </rightstatement></statement>
    </interpretation>
  </interpretations>"""

INTERP_MACHINE = """
  <interpretations>
    <interpretation date="2020-03-01" time="10:15:30">
      <mdsignatureline>Unconfirmed Diagnosis</mdsignatureline>
      <severity code="BO" id="3">- BORDERLINE ECG -</severity>
      <statement><statementcode>SR</statementcode><leftstatement>SINUS RHYTHM</leftstatement><rightstatement/></statement>
    </interpretation>
  </interpretations>"""


def repbeats_104(beats: dict[str, np.ndarray], duration_ms: int, rate: int = 500) -> str:
    items = []
    for name, data in beats.items():
        b64 = base64.b64encode(data.astype("<i2").tobytes()).decode()
        items.append(
            f'<repbeat leadname="{name}"><qrsdur>92</qrsdur><qtint>380</qtint>'
            f'<waveform duration="{duration_ms}">{b64}</waveform></repbeat>'
        )
    return (f'<repbeats dataencoding="Base64" samplespersec="{rate}" resolution="2.5" '
            f'repbeatmethod="mean">{"".join(items)}</repbeats>')


def sierra_104(
    waveform: str,
    *,
    pw_attrs: str = 'dataencoding="Base64" compression="XLI"',
    labels: list[str] = LABELS,
    version: str = "1.04",
    status: str = "Confirmed",
    interp: str = INTERP_CONFIRMED,
    repbeats: str = "",
    patient: str = PATIENT_104,
    duration_ms: int = DURATION_MS,
    samplingrate: str = "500",
    encoding: str = "utf-8",
) -> str:
    return f"""<?xml version="1.0" encoding="{encoding}"?>
<restingecgdata xmlns="http://www3.medical.philips.com" status="{status}">
  <documentinfo>
    <documenttype>PhilipsECG</documenttype>
    <documentversion>
      {version}
    </documentversion>
    <editor date="2020-03-02" time="08:00:00" id="9">Editor Name</editor>
  </documentinfo>
  <reportinfo date="2020-03-01" time="10:15:30">
    <reportbandwidth><highpassfiltersetting>0.5</highpassfiltersetting><notchfiltersetting>60</notchfiltersetting></reportbandwidth>
  </reportinfo>
  <dataacquisition date="2020-03-01" time="10:15:30" statflag="False">
    <machine machineid="CART-12" detaildescription="Philips Medical Products:860306:B.01.02">PageWriter TC70</machine>
    <acquirer>
      <operator id="TECH1"/>
      <room>R12</room>
      <departmentname>Cardiology</departmentname>
      <institutionname>General Hospital</institutionname>
      <facilityname>Main Campus</facilityname>
      <referringclinician id="">,</referringclinician>
      <orderingclinician id="3">Brown, Bob</orderingclinician>
    </acquirer>
    <signalcharacteristics>
      <samplingrate>{samplingrate}</samplingrate>
      <resolution>5</resolution>
      <hipass>0.05</hipass>
      <lowpass>150</lowpass>
      <acsetting>60</acsetting>
      <notchfiltered>False</notchfiltered>
      <notchfilterfreqs/>
      <acquisitiontype>10-WIRE</acquisitiontype>
      <bitspersample>16</bitspersample>
      <signaloffset>0</signaloffset>
      <signalsigned>True</signalsigned>
      <numberchannelsallocated>12</numberchannelsallocated>
      <numberchannelsvalid>12</numberchannelsvalid>
      <electrodeplacement>STD</electrodeplacement>
    </signalcharacteristics>
  </dataacquisition>{patient}
  <internalmeasurements>
    <crossleadmeasurements>
      <numberofcomplexes>9</numberofcomplexes>
      <meanqtc>415</meanqtc>
    </crossleadmeasurements>
    <leadmeasurements>
      <leadmeasurement leadname="I" pmeasflag="True"><pamp>55</pamp><leadqualitystates/></leadmeasurement>
    </leadmeasurements>
  </internalmeasurements>{interp}
  <waveforms>
    <parsedwaveforms {pw_attrs} numberofleads="{len(labels)}" leadlabels="{' '.join(labels)}"
        durationperchannel="{duration_ms}" samplespersecond="500" resolution="5" signaloffset="0"
        signalsigned="True" bitspersample="16" hipass="0.05" lowpass="150" notchfiltered="False"
        notchfilterfreqs="0">{waveform}</parsedwaveforms>
    {repbeats}
  </waveforms>
</restingecgdata>
"""


def xli_b64(leads: list[np.ndarray]) -> str:
    text = base64.b64encode(xli_encode(leads)).decode()
    # Real files wrap the Base64 text over several lines
    return "\n".join(text[i:i + 76] for i in range(0, len(text), 76))


def write(tmp_path: Path, text: str, name: str = "ecg.xml", encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


@pytest.fixture
def leads() -> list[np.ndarray]:
    return true_leads()


@pytest.fixture
def xli_file(tmp_path: Path, leads) -> Path:
    beats = {name: (leads[k][:600] // 2) for k, name in enumerate(LABELS)}
    return write(tmp_path, sierra_104(xli_b64(stored_residuals(leads)), repbeats=repbeats_104(beats, 1200)))


# ── codec ───────────────────────────────────────────────────────────

class TestXliRoundTrip:
    def test_chunks_decode_to_original(self, leads):
        decoded = xli_decode(xli_encode(leads[:3]))
        assert len(decoded) == 3
        for got, want in zip(decoded, leads[:3]):
            np.testing.assert_array_equal(got, want)

    def test_long_constant_chunk_fills_lzw_table(self):
        # Long repetitive data exhausts the 10-bit code table
        data = np.tile(np.array([0, 5, 10, 5], dtype=np.int64), 3000)
        np.testing.assert_array_equal(xli_decode(xli_chunk(data))[0], data)

    @pytest.mark.parametrize("cut", [3, 8, 20])
    def test_truncated_stream(self, leads, cut):
        blob = xli_encode(leads[:1])
        with pytest.raises(CorruptedFileError):
            xli_decode(blob[:cut])

    def test_negative_size(self):
        with pytest.raises(CorruptedFileError):
            xli_decode(struct.pack("<ihh", -4, 1, 0) + b"\x00" * 16)


# ── detection ───────────────────────────────────────────────────────

class TestDetection:
    @pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-16-le", "utf-16-be"])
    def test_can_parse_encodings(self, tmp_path, leads, encoding):
        declared = "utf-16" if encoding.startswith("utf-16") else "utf-8"
        text = sierra_104(xli_b64(stored_residuals(leads)), encoding=declared)
        raw = text.encode(encoding)
        if encoding == "utf-16-le":
            raw = b"\xff\xfe" + raw
        if encoding == "utf-16-be":
            raw = b"\xfe\xff" + raw
        path = tmp_path / "ecg.xml"
        path.write_bytes(raw)
        assert SierraXMLParser.can_parse(path, raw[:4096])
        record = FileParser().parse(path)
        assert record.source_format == "sierra_xml"
        assert record.patient.first_name == "Jérôme"

    def test_rejects_other_xml(self, tmp_path):
        assert not SierraXMLParser.can_parse(tmp_path / "x.xml", b"<?xml version='1.0'?><AnnotatedECG/>")


# ── rhythm leads ────────────────────────────────────────────────────

class TestRhythmLeads:
    def test_xli_limb_leads_rebuilt(self, xli_file, leads):
        record = SierraXMLParser().parse(xli_file)
        assert [lead.label for lead in record.leads] == LABELS
        for lead, want in zip(record.leads, leads):
            assert lead.samples.dtype == np.float64
            np.testing.assert_array_equal(lead.samples, want)
            assert lead.sampling_rate == 500

    def test_scaling_metadata(self, xli_file):
        lead = SierraXMLParser().parse(xli_file).leads[0]
        assert lead.resolution == 5.0
        assert lead.resolution_unit == "uV"
        assert lead.adc_resolution == 5.0
        assert lead.adc_resolution_unit == "uV"
        assert lead.offset == 0.0
        assert lead.is_raw is True
        assert lead.units == ""

    def test_file_parser_scaling(self, xli_file, leads):
        scaled = FileParser().parse(xli_file)
        assert scaled.leads[0].units == "mV"
        np.testing.assert_allclose(scaled.leads[0].samples, leads[0] * 0.005)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = FileParser().parse(xli_file, auto_scale=False)
        np.testing.assert_array_equal(raw.leads[1].samples, leads[1])

    def test_uncompressed_base64_kept_as_stored(self, tmp_path, leads):
        blob = np.stack(leads).astype("<i2").tobytes()
        text = sierra_104(base64.b64encode(blob).decode(), pw_attrs='dataencoding="Base64"')
        record = SierraXMLParser().parse(write(tmp_path, text))
        for lead, want in zip(record.leads, leads):
            np.testing.assert_array_equal(lead.samples, want)

    def test_plain_encoding(self, tmp_path, leads):
        text = " ".join(str(v) for lead in leads for v in lead)
        record = SierraXMLParser().parse(write(tmp_path, sierra_104(text, pw_attrs='dataencoding="Plain"')))
        np.testing.assert_array_equal(record.leads[5].samples, leads[5])

    def test_nonstandard_order_not_rebuilt(self, tmp_path, leads):
        order = LABELS[6:] + LABELS[:6]
        text = sierra_104(xli_b64(leads), labels=order)
        record = SierraXMLParser().parse(write(tmp_path, text))
        assert [lead.label for lead in record.leads] == order
        for lead, want in zip(record.leads, leads):
            np.testing.assert_array_equal(lead.samples, want)

    def test_extra_chunks_ignored(self, tmp_path, leads):
        extra = stored_residuals(leads) + [np.zeros(N, dtype=np.int64)] * 4
        record = SierraXMLParser().parse(write(tmp_path, sierra_104(xli_b64(extra))))
        assert len(record.leads) == 12

    def test_signal_summary(self, xli_file):
        record = SierraXMLParser().parse(xli_file)
        signal = record.recording.acquisition.signal
        assert signal.sampling_rate == 500
        assert signal.resolution == 5.0
        assert signal.compression == "XLI"
        assert signal.number_channels_allocated == 12


# ── 1.03 documents ──────────────────────────────────────────────────

def sierra_103(leads: list[np.ndarray], beats: dict[str, np.ndarray]) -> str:
    reps = "".join(
        f'<repbeat leadname="{name}" duration="600" qrsdur="98">'
        f'{base64.b64encode(b.astype("<i2").tobytes()).decode()}</repbeat>'
        for name, b in beats.items()
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<restingecgdata xmlns="http://www3.medical.philips.com" status="Not yet determined">
  <documentinfo><documenttype>SierraECG</documenttype><documentversion>1.03</documentversion></documentinfo>
  <dataacquisition date="2011-12-01" time="07:27:34" statflag="False">
    <machine machineid="0001" detaildescription="">HeartstartMRx</machine>
    <acquirer><operatorid>OP5</operatorid><institutionname>EMS</institutionname></acquirer>
    <signalcharacteristics>
      <samplingrate>500</samplingrate>
      <signalresolution>5</signalresolution>
      <signalbandwidth>0.05-150</signalbandwidth>
      <acquisitiontype>STD-12</acquisitiontype>
      <bitspersample>16</bitspersample>
      <signaloffset>0</signaloffset>
      <signalsigned>True</signalsigned>
      <numberchannelsallocated>12</numberchannelsallocated>
      <numberchannelsvalid>12</numberchannelsvalid>
      <leadset>STD-12</leadset>
    </signalcharacteristics>
  </dataacquisition>
  <patient><generalpatientdata>
    <patientid>X1</patientid><name><lastname></lastname><firstname></firstname></name>
    <age><years>61</years></age><pacestatus>Unknown</pacestatus><sex>Female</sex>
  </generalpatientdata></patient>
  <measurements>
    <globalmeasurements>
      <numberofcomplexes>6</numberofcomplexes>
      <pfrontaxis>66</pfrontaxis><qrsfrontaxis>-21</qrsfrontaxis><tfrontaxis>59</tfrontaxis>
      <meanventrate>41</meanventrate><meanprint>151</meanprint><meanqrsdur>135</meanqrsdur>
      <meanqtint>476</meanqtint><meanqtc>393</meanqtc>
    </globalmeasurements>
  </measurements>
  <interpretations>
    <interpretation date="2011-12-01" time="07:27:34">
      <interpretationmeasurements><heartrate>42</heartrate></interpretationmeasurements>
      <mdsignatureline>Unconfirmed diagnosis</mdsignatureline>
      <severity code="AB">- ABNORMAL ECG -</severity>
      <statement><statementcode/><leftstatement>Sinus bradycardia</leftstatement><rightstatement>rate&lt; 50</rightstatement></statement>
      <statement><statementcode/><leftstatement>&gt;&gt;&gt; Acute MI &lt;&lt;&lt;</leftstatement><rightstatement></rightstatement></statement>
    </interpretation>
  </interpretations>
  <waveforms>
    <parsedwaveforms compressflag="True" compressmethod="XLI" filterflag="True" dataencoding="Base64"
        durationperchannel="{DURATION_MS}" nbitspersample="16">{xli_b64(stored_residuals(leads))}</parsedwaveforms>
    <repbeats dataencoding="Base64" samplespersec="1000" resolution="1.0" repbeatmethod="mean">{reps}</repbeats>
  </waveforms>
</restingecgdata>
"""


class TestVersion103:
    @pytest.fixture
    def record(self, tmp_path, leads):
        beats = {name: leads[k][:600] for k, name in enumerate(LABELS)}
        return SierraXMLParser().parse(write(tmp_path, sierra_103(leads, beats)))

    def test_leads(self, record, leads):
        assert [lead.label for lead in record.leads] == LABELS
        for lead, want in zip(record.leads, leads):
            np.testing.assert_array_equal(lead.samples, want)
        assert record.leads[0].resolution == 5.0
        assert record.file_format.version == "1.03"

    def test_repbeats_in_repbeat_text(self, record, leads):
        assert len(record.median_beats) == 12
        beat = record.median_beats[2]
        assert beat.label == "III"
        assert beat.sampling_rate == 1000
        assert beat.resolution == 1.0
        assert beat.resolution_unit == "uV"
        assert beat.is_raw is False
        np.testing.assert_array_equal(beat.samples, leads[2][:600])
        assert beat.annotations["qrsdur"] == "98"

    def test_measurements(self, record):
        m = record.measurements
        assert m.heart_rate == 42
        assert m.pr_interval == 151
        assert m.qrs_duration == 135
        assert m.qt_interval == 476
        assert m.qtc_bazett is None  # 1.03 meanqtc does not state its formula
        assert record.annotations["qtc"] == "393 ms (formula not stated)"
        assert (m.p_axis, m.qrs_axis, m.t_axis) == (66, -21, 59)
        assert m.qrs_count == 6

    def test_interpretation(self, record):
        interp = record.interpretation
        assert interp.source == "machine"
        assert interp.interpreter == ""
        assert interp.statements == [("Sinus bradycardia", "rate< 50"), (">>> Acute MI <<<", "")]
        assert interp.severity == "ABNORMAL"
        assert interp.interpretation_date == datetime(2011, 12, 1, 7, 27, 34)
        assert "machine_interpretation" not in record.annotations

    def test_patient_and_filters(self, record):
        assert record.patient.age == 61
        assert record.patient.sex == "F"
        assert record.patient.first_name == ""
        assert record.recording.technician == "OP5"
        filters = record.recording.acquisition.filters
        assert (filters.highpass, filters.lowpass) == (0.05, 150.0)
        assert record.recording.acquisition.signal.filtered is True


# ── metadata (1.04) ─────────────────────────────────────────────────

class TestMetadata:
    def test_patient(self, xli_file):
        record = SierraXMLParser().parse(xli_file)
        p = record.patient
        assert p.patient_id == "P-001"
        assert (p.first_name, p.last_name) == ("Jérôme", "Doe")
        assert p.birth_date == datetime(1980, 6, 15)
        assert p.age == 39  # at acquisition 2020-03-01, not the default age 50
        assert record.raw_metadata["default_age"] == "50"
        assert p.sex == "M"
        assert p.race == "White"
        assert p.height == 177.8
        assert p.weight == 69.9
        assert p.has_pacemaker is None
        assert record.raw_metadata["pace_status"] == "Unknown"
        assert record.raw_metadata["mrn"] == "M-77"

    def test_default_age_without_birth_date(self, tmp_path, leads):
        patient = PATIENT_104.replace("<dateofbirth>1980-06-15</dateofbirth>", "<dateofbirth/>")
        record = SierraXMLParser().parse(write(tmp_path, sierra_104(xli_b64(stored_residuals(leads)), patient=patient)))
        assert record.patient.age is None
        assert record.patient.birth_date is None

    def test_recording_and_device(self, xli_file):
        record = SierraXMLParser().parse(xli_file)
        rec = record.recording
        assert rec.date == datetime(2020, 3, 1, 10, 15, 30)
        assert rec.duration.total_seconds() == 2.0
        assert rec.end_date is None  # not stated in the file, not computed
        assert rec.technician == "TECH1"
        assert rec.room == "R12"
        assert rec.location == "Main Campus"
        assert rec.referring_physician == "Brown, Bob"
        dev = rec.device
        assert dev.manufacturer == "Philips Medical Products"
        assert dev.model == "PageWriter TC70"
        assert dev.name == "CART-12"
        assert dev.software_version == "B.01.02"
        assert dev.serial_number == ""
        assert dev.institution == "General Hospital"
        assert dev.department == "Cardiology"
        assert dev.acquisition_type == "10-WIRE"
        assert record.raw_metadata["device_product_number"] == "860306"
        assert record.file_format.version == "1.04"

    def test_filters_notch_zero_is_none(self, xli_file):
        filters = SierraXMLParser().parse(xli_file).recording.acquisition.filters
        assert filters.highpass == 0.05
        assert filters.lowpass == 150.0
        assert filters.notch is None
        assert filters.notch_active is False

    def test_measurements(self, xli_file):
        m = SierraXMLParser().parse(xli_file).measurements
        assert (m.heart_rate, m.rr_interval, m.pr_interval, m.qrs_duration) == (72, 833, 160, 90)
        assert (m.qt_interval, m.qtc_bazett, m.qtc_fridericia) == (380, 416, 404)
        assert (m.p_axis, m.qrs_axis, m.t_axis) == (45, None, 30)
        assert m.qrs_count == 9

    def test_lead_measurements(self, xli_file):
        record = SierraXMLParser().parse(xli_file)
        assert record.leads[0].annotations == {"pmeasflag": "True", "pamp": "55"}

    def test_repbeats(self, xli_file, leads):
        beats = SierraXMLParser().parse(xli_file).median_beats
        assert [b.label for b in beats] == LABELS
        beat = beats[1]
        assert beat.sampling_rate == 500
        assert beat.resolution == 2.5
        assert beat.adc_resolution == 2.5
        assert len(beat.samples) == 600
        np.testing.assert_array_equal(beat.samples, leads[1][:600] // 2)
        assert beat.annotations == {"qrsdur": "92", "qtint": "380", "repbeatmethod": "mean"}
        scaled = FileParser().parse(xli_file).median_beats[1]
        assert scaled.units == "mV"
        np.testing.assert_allclose(scaled.samples, (leads[1][:600] // 2) * 0.0025)


class TestInterpretation:
    def test_confirmed(self, xli_file):
        record = SierraXMLParser().parse(xli_file)
        interp = record.interpretation
        assert interp.source == "confirmed"
        assert interp.interpreter == "Smith, Ann"
        assert interp.interpretation_date == datetime(2020, 3, 2, 8, 0, 0)
        assert interp.statements == [("SINUS RHYTHM", "V-rate 50-99"), ("Compared with prior", "")]
        assert interp.severity == "ABNORMAL"
        assert record.annotations["machine_interpretation"] == (
            "SINUS RHYTHM | V-rate 50-99\nPossible LEFT ATRIAL ENLARGEMENT"
        )

    def test_signature_only(self, tmp_path, leads):
        interp_xml = INTERP_CONFIRMED.replace(
            '<confirmingclinician date="2020-03-02" time="08:00:00" id="7">Smith, Ann</confirmingclinician>', ""
        )
        record = SierraXMLParser().parse(write(tmp_path, sierra_104(xli_b64(stored_residuals(leads)), interp=interp_xml)))
        assert record.interpretation.source == "confirmed"
        assert record.interpretation.interpreter == "Smith, Ann"
        assert record.interpretation.interpretation_date == datetime(2020, 3, 2, 8, 0, 0)

    def test_machine(self, tmp_path, leads):
        text = sierra_104(xli_b64(stored_residuals(leads)), status="Unconfirmed", interp=INTERP_MACHINE)
        record = SierraXMLParser().parse(write(tmp_path, text))
        interp = record.interpretation
        assert interp.source == "machine"
        assert interp.interpreter == ""
        assert interp.statements == [("SINUS RHYTHM", "")]
        assert interp.severity == "BORDERLINE"
        assert interp.interpretation_date == datetime(2020, 3, 1, 10, 15, 30)
        assert "machine_interpretation" not in record.annotations


# ── errors ──────────────────────────────────────────────────────────

class TestErrors:
    def _parse(self, tmp_path, text):
        return SierraXMLParser().parse(write(tmp_path, text))

    def test_truncated_xli(self, tmp_path, leads):
        blob = xli_encode(stored_residuals(leads))
        text = sierra_104(base64.b64encode(blob[: len(blob) // 2]).decode())
        with pytest.raises(CorruptedFileError):
            self._parse(tmp_path, text)

    def test_too_few_chunks(self, tmp_path, leads):
        with pytest.raises(CorruptedFileError, match="leads"):
            self._parse(tmp_path, sierra_104(xli_b64(leads[:8])))

    def test_short_chunks(self, tmp_path, leads):
        short = [lead[: N // 2] for lead in stored_residuals(leads)]
        with pytest.raises(CorruptedFileError, match="samples"):
            self._parse(tmp_path, sierra_104(xli_b64(short)))

    def test_invalid_base64(self, tmp_path):
        with pytest.raises(CorruptedFileError, match="Base64"):
            self._parse(tmp_path, sierra_104("@@@@"))

    def test_uncompressed_too_short(self, tmp_path, leads):
        blob = np.stack(leads).astype("<i2").tobytes()[:1000]
        with pytest.raises(CorruptedFileError):
            self._parse(tmp_path, sierra_104(base64.b64encode(blob).decode(), pw_attrs='dataencoding="Base64"'))

    def test_bad_sampling_rate(self, tmp_path, leads):
        text = sierra_104(xli_b64(stored_residuals(leads))).replace('samplespersecond="500"', 'samplespersecond="abc"')
        text = text.replace("<samplingrate>500</samplingrate>", "<samplingrate>abc</samplingrate>")
        with pytest.raises(CorruptedFileError, match="sampling rate"):
            self._parse(tmp_path, text)

    def test_malformed_xml(self, tmp_path, leads):
        text = sierra_104(xli_b64(stored_residuals(leads)))
        with pytest.raises(CorruptedFileError, match="Malformed XML"):
            self._parse(tmp_path, text[: len(text) // 2])

    def test_truncated_repbeat(self, tmp_path, leads):
        beats = {"I": leads[0][:300]}
        text = sierra_104(xli_b64(stored_residuals(leads)), repbeats=repbeats_104(beats, 1200))
        with pytest.raises(CorruptedFileError, match="repbeat I"):
            self._parse(tmp_path, text)

    def test_unsupported_version(self, tmp_path, leads):
        with pytest.raises(UnsupportedFormatError):
            self._parse(tmp_path, sierra_104(xli_b64(stored_residuals(leads)), version="1.05"))

    def test_missing_waveforms(self, tmp_path, leads):
        text = sierra_104("x")
        start = text.index("<parsedwaveforms")
        end = text.index("</parsedwaveforms>") + len("</parsedwaveforms>")
        with pytest.raises(MissingElementError):
            self._parse(tmp_path, text[:start] + text[end:])

    def test_file_parser_wraps_errors(self, tmp_path):
        path = write(tmp_path, sierra_104("@@@@"))
        with pytest.raises(CorruptedFileError):
            FileParser().parse(path)


class TestWarnings:
    def test_missing_resolution_warns_at_caller(self, tmp_path, leads):
        text = sierra_104(xli_b64(stored_residuals(leads)))
        text = text.replace("<resolution>5</resolution>", "").replace(' resolution="5"', "")
        path = write(tmp_path, text)
        with pytest.warns(UserWarning, match="no signal resolution") as caught:
            record = SierraXMLParser().parse(path)
        assert caught[0].filename == __file__
        lead = record.leads[0]
        assert lead.is_raw is True
        assert lead.resolution_unit == ""
        np.testing.assert_array_equal(lead.samples, leads[0])

    def test_compressed_repbeats_skipped(self, tmp_path, leads):
        reps = '<repbeats dataencoding="Base64" compression="Huffman" samplespersec="500"><repbeat leadname="I"/></repbeats>'
        path = write(tmp_path, sierra_104(xli_b64(stored_residuals(leads)), repbeats=reps))
        with pytest.warns(UserWarning, match="Huffman") as caught:
            record = SierraXMLParser().parse(path)
        assert caught[0].filename == __file__
        assert record.median_beats == []
        assert len(record.leads) == 12

    def test_repbeats_without_rate_or_resolution_flagged(self, tmp_path, leads):
        beats = {"I": leads[0][:600]}
        reps = repbeats_104(beats, 1200).replace(' samplespersec="500"', "").replace(' resolution="2.5"', "")
        path = write(tmp_path, sierra_104(xli_b64(stored_residuals(leads)), repbeats=reps))
        with pytest.warns(UserWarning) as caught:
            record = SierraXMLParser().parse(path)
        messages = [str(w.message) for w in caught]
        assert any("no sampling rate" in m for m in messages)
        assert any("no resolution" in m for m in messages)
        assert all(w.filename == __file__ for w in caught)
        beat = record.median_beats[0]
        assert (beat.sampling_rate, beat.resolution) == (500, 5.0)
        assert record.raw_metadata["median_sampling_rate_stated"] is False
        assert record.raw_metadata["median_resolution_stated"] is False

    def test_stated_repbeat_values_not_flagged(self, xli_file):
        record = SierraXMLParser().parse(xli_file)
        assert "median_sampling_rate_stated" not in record.raw_metadata
        assert "median_resolution_stated" not in record.raw_metadata


class TestSecurity:
    def test_entity_declarations_refused(self, tmp_path):
        text = ('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e "hello">]>'
                "<restingecgdata><n>&e;</n></restingecgdata>")
        with pytest.raises(CorruptedFileError, match="entity"):
            SierraXMLParser().parse(write(tmp_path, text))
