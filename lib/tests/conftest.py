"""Shared test fixtures for ECGDataKit."""

from __future__ import annotations

import base64
import binascii
import os
import struct
import textwrap
from pathlib import Path

# Prevent plots from opening GUI windows during tests
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _no_plot_display(monkeypatch):
    """Suppress all plot display during tests."""
    # Matplotlib: use non-interactive Agg backend
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda *a, **kw: None)
    except ImportError:
        pass
    # Plotly: prevent browser opening
    try:
        import plotly.io as pio
        pio.renderers.default = "json"
        monkeypatch.setattr(pio, "show", lambda *a, **kw: None)
    except ImportError:
        pass


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# HL7 aECG XML fixture (structure per ANSI/HL7 V3 ECG R1-2004, PORT_MT020001)
# ---------------------------------------------------------------------------

HL7_AECG_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<AnnotatedECG xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <id root="test-uuid-1234"/>
  <code code="93000" codeSystem="2.16.840.1.113883.6.12"/>
  <effectiveTime>
    <low value="20230615103000"/>
    <high value="20230615103010"/>
  </effectiveTime>
  <componentOf>
    <timepointEvent>
      <code code="VISIT_1" codeSystem="1.2.3"/>
      <componentOf>
        <subjectAssignment>
          <subject>
            <trialSubject>
              <id root="1.2.3" extension="SUBJ-001"/>
              <subjectDemographicPerson>
                <name><given>Ann</given><given>Marie</given><family>Doe</family></name>
                <administrativeGenderCode code="M" codeSystem="2.16.840.1.113883.5.1"/>
                <birthTime value="19800101"/>
                <raceCode code="2106-3" codeSystem="2.16.840.1.113883.5.104" displayName="White"/>
              </subjectDemographicPerson>
            </trialSubject>
          </subject>
          <componentOf>
            <clinicalTrial>
              <id root="1.2.3" extension="TRIAL-1"/>
              <location>
                <trialSite>
                  <id root="1.2.3" extension="SITE-9"/>
                  <location><name>Test Clinic</name></location>
                </trialSite>
              </location>
            </clinicalTrial>
          </componentOf>
        </subjectAssignment>
      </componentOf>
    </timepointEvent>
  </componentOf>
  <component>
    <series>
      <code code="RHYTHM" codeSystem="2.16.840.1.113883.5.4"/>
      <effectiveTime>
        <low value="20230615103000"/>
        <high value="20230615103010"/>
      </effectiveTime>
      <author>
        <seriesAuthor>
          <manufacturedSeriesDevice>
            <id extension="SN-42"/>
            <manufacturerModelName>ELI250</manufacturerModelName>
            <softwareName>5.3</softwareName>
          </manufacturedSeriesDevice>
          <manufacturerOrganization><name>Acme Medical</name></manufacturerOrganization>
        </seriesAuthor>
      </author>
      <secondaryPerformer>
        <seriesPerformer><assignedPerson><name>KAB</name></assignedPerson></seriesPerformer>
      </secondaryPerformer>
      <controlVariable>
        <controlVariable>
          <code code="MDC_ECG_CTL_VBL_ATTR_FILTER_LOW_PASS" codeSystem="2.16.840.1.113883.6.24"/>
          <component>
            <controlVariable>
              <code code="MDC_ECG_CTL_VBL_ATTR_FILTER_CUTOFF_FREQ" codeSystem="2.16.840.1.113883.6.24"/>
              <value xsi:type="PQ" value="150" unit="Hz"/>
            </controlVariable>
          </component>
        </controlVariable>
      </controlVariable>
      <controlVariable>
        <controlVariable>
          <code code="MDC_ECG_CTL_VBL_ATTR_FILTER_HIGH_PASS" codeSystem="2.16.840.1.113883.6.24"/>
          <component>
            <controlVariable>
              <code code="MDC_ECG_CTL_VBL_ATTR_FILTER_CUTOFF_FREQ" codeSystem="2.16.840.1.113883.6.24"/>
              <value xsi:type="PQ" value="0.05" unit="Hz"/>
            </controlVariable>
          </component>
        </controlVariable>
      </controlVariable>
      <controlVariable>
        <controlVariable>
          <code code="MDC_ECG_CTL_VBL_ATTR_FILTER_NOTCH" codeSystem="2.16.840.1.113883.6.24"/>
          <component>
            <controlVariable>
              <code code="MDC_ECG_CTL_VBL_ATTR_FILTER_NOTCH_FREQ" codeSystem="2.16.840.1.113883.6.24"/>
              <value xsi:type="PQ" value="50" unit="Hz"/>
            </controlVariable>
          </component>
        </controlVariable>
      </controlVariable>
      <component>
        <sequenceSet>
          <component>
            <sequence>
              <code code="TIME_ABSOLUTE" codeSystem="2.16.840.1.113883.5.4"/>
              <value xsi:type="GLIST_TS">
                <head value="20230615103000.000"/>
                <increment value="0.002" unit="s"/>
              </value>
            </sequence>
          </component>
          <component>
            <sequence>
              <code code="MDC_ECG_LEAD_I" codeSystem="2.16.840.1.113883.6.24"/>
              <value xsi:type="SLIST_PQ">
                <origin value="0" unit="uV"/>
                <scale value="5" unit="uV"/>
                <digits>100 200 300 400 500</digits>
              </value>
            </sequence>
          </component>
          <component>
            <sequence>
              <code code="MDC_ECG_LEAD_II" codeSystem="2.16.840.1.113883.6.24"/>
              <value xsi:type="SLIST_PQ">
                <origin value="0" unit="uV"/>
                <scale value="5" unit="uV"/>
                <digits>110 210
                  310	410 510</digits>
              </value>
            </sequence>
          </component>
        </sequenceSet>
      </component>
      <derivation>
        <derivedSeries>
          <code code="REPRESENTATIVE_BEAT" codeSystem="2.16.840.1.113883.5.4"/>
          <component>
            <sequenceSet>
              <component>
                <sequence>
                  <code code="TIME_RELATIVE" codeSystem="2.16.840.1.113883.5.4"/>
                  <value xsi:type="GLIST_PQ">
                    <head value="0" unit="ms"/>
                    <increment value="2" unit="ms"/>
                  </value>
                </sequence>
              </component>
              <component>
                <sequence>
                  <code code="MDC_ECG_LEAD_I" codeSystem="2.16.840.1.113883.6.24"/>
                  <value xsi:type="SLIST_PQ">
                    <origin value="0" unit="uV"/>
                    <scale value="5" unit="uV"/>
                    <digits>1 2 3</digits>
                  </value>
                </sequence>
              </component>
              <component>
                <sequence>
                  <code code="MDC_ECG_LEAD_II" codeSystem="2.16.840.1.113883.6.24"/>
                  <value xsi:type="SLIST_PQ">
                    <origin value="0" unit="uV"/>
                    <scale value="5" unit="uV"/>
                    <digits>4 5 6</digits>
                  </value>
                </sequence>
              </component>
            </sequenceSet>
          </component>
        </derivedSeries>
      </derivation>
      <subjectOf>
        <annotationSet>
          <activityTime value="20230615103010"/>
          <component>
            <annotation>
              <code code="MDC_ECG_HEART_RATE" codeSystem="2.16.840.1.113883.6.24"/>
              <value xsi:type="PQ" value="72" unit="bpm"/>
            </annotation>
          </component>
          <component>
            <annotation>
              <code code="MDC_ECG_TIME_PD_PR" codeSystem="2.16.840.1.113883.6.24"/>
              <value xsi:type="PQ" value="160" unit="ms"/>
            </annotation>
          </component>
          <component>
            <annotation>
              <code code="MDC_ECG_TIME_PD_QT" codeSystem="2.16.840.1.113883.6.24"/>
              <value xsi:type="PQ" value="0.412" unit="s"/>
            </annotation>
          </component>
          <component>
            <annotation>
              <code code="MDC_ECG_INTERPRETATION" codeSystem="2.16.840.1.113883.6.24"/>
              <component>
                <annotation>
                  <code code="MDC_ECG_INTERPRETATION_STATEMENT" codeSystem="2.16.840.1.113883.6.24"/>
                  <value xsi:type="ST">Sinus rhythm</value>
                </annotation>
              </component>
              <component>
                <annotation>
                  <code code="MDC_ECG_INTERPRETATION_SEVERITY" codeSystem="2.16.840.1.113883.6.24"/>
                  <value xsi:type="CE" code="NORMAL" displayName="Normal ECG"/>
                </annotation>
              </component>
            </annotation>
          </component>
        </annotationSet>
      </subjectOf>
    </series>
  </component>
</AnnotatedECG>
""")


# ---------------------------------------------------------------------------
# Mortara ELI XML fixture (layout of real ELI 250/280 exports, synthetic data)
# ---------------------------------------------------------------------------

def mortara_counts(lead_index: int, n: int) -> np.ndarray:
    """Synthetic int16 counts for Mortara lead *lead_index*."""
    return ((np.arange(n) % 50) - 25 + 10 * lead_index).astype("<i2")


def _mortara_b64(lead_index: int, n: int) -> str:
    return base64.b64encode(mortara_counts(lead_index, n).tobytes()).decode()


MORTARA_RHYTHM_N = 500     # 1 s at 500 Hz
MORTARA_MEDIAN_N = 300

MORTARA_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!-- Generated by ELI Link 4.5.0.2 -->\n'
    '<ECG ACQUISITION_TIME="20231201120000" ACQUISITION_TIME_XML="2023-12-01T12:00:00" '
    'ROOM="R12" LOCATION="Cardio" COMMENT="Chest pain" AGE="99" AGE_UNITS="Y" '
    'HEIGHT="70" HEIGHT_UNITS="IN" WEIGHT="80" WEIGHT_UNITS="KG" NUM_QRS="12" '
    'AVERAGE_RR="800" VENT_RATE="75" TECHNICIAN="TS" SYSTOLIC_BP="120" '
    'DIASTOLIC_BP="80" SEQUENCE_NUMBER="42">\n'
    '  <DEMOGRAPHIC_FIELDS>\n'
    '    <DEMOGRAPHIC_FIELD ID="1" LABEL="Last:" VALUE="Doe" UNITS="" />\n'
    '    <DEMOGRAPHIC_FIELD ID="7" LABEL="First:" VALUE="John" UNITS="" />\n'
    '    <DEMOGRAPHIC_FIELD ID="2" LABEL="ID:" VALUE="PAT001" UNITS="" />\n'
    '  </DEMOGRAPHIC_FIELDS>\n'
    '  <SITE ID="3" />\n'
    '  <SUBJECT LAST_NAME="Doe" FIRST_NAME="Jos\u00e9 O\'Neil" GENDER="Male" ID="PAT001" '
    'DOB="19900515" DOB_XML="1990-05-15" />\n'
    '  <MEDICATION CLASS_CODE="2" DRUG_CODE="7" NAME="Bisoprolol" />\n'
    '  <MEDICATION CLASS_CODE="0" DRUG_CODE="0" NAME="" />\n'
    '  <SOURCE TYPE="RESTING" MANUFACTURER="Mortara Instrument, Inc." MODEL="ELI250c" '
    'TRANSMISSION_TIME="20231202080000" TRANSMISSION_TIME_XML="2023-12-02T08:00:00" '
    'BASELINE_ROLL_FILTER="5" LOW_PASS_FILTER="150" PRINT_FILTER="150" FILTER_BITMAP="2" '
    'ACQUIRING_DEVICE_SW_VERSION="V2.0.4.0" ACQUIRING_DEVICE_INTERPRETATION_SW_VERSION="7.2.3" '
    'ACQUIRING_DEVICE_SERIAL_NUMBER="SN123" ACQUIRING_DEVICE_SITE_NAME="Main Hospital" />\n'
    '  <AUTOMATIC_INTERPRETATION>\n'
    '    <STATEMENT STATEMENT_NUMBER="1" TEXT="SINUS RHYTHM" REASON="" />\n'
    '    <STATEMENT STATEMENT_NUMBER="2" TEXT="LEFT AXIS DEVIATION " REASON="QRS AXIS &lt; -30" />\n'
    '    <STATEMENT STATEMENT_NUMBER="3" TEXT="" REASON="" />\n'
    '    <STATEMENT STATEMENT_NUMBER="4" TEXT="UNCONFIRMED REPORT" REASON="" />\n'
    '  </AUTOMATIC_INTERPRETATION>\n'
    '  <TYPICAL_CYCLE R_PEAK="150" P_ONSET="-208" P_OFFSET="-97" Q_ONSET="-45" Q_OFFSET="48" '
    'T_OFFSET="411" P_DURATION="111" PR_DURATION="163" QRS_DURATION="93" QT="456" QTC="463" '
    'QTB="469" QTF="464" P_AXIS="57" QRS_AXIS="-35" T_AXIS="40" BITS="16" FORMAT="SIGNED" '
    f'UNITS_PER_MV="200" DURATION="{MORTARA_MEDIAN_N}" SAMPLE_FREQ="250" ENCODING="BASE64">\n'
    f'    <TYPICAL_CYCLE_CHANNEL NAME="I" DATA="{_mortara_b64(0, MORTARA_MEDIAN_N)}" />\n'
    f'    <TYPICAL_CYCLE_CHANNEL NAME="II" DATA="{_mortara_b64(1, MORTARA_MEDIAN_N)}" />\n'
    '  </TYPICAL_CYCLE>\n'
    + "".join(
        f'  <CHANNEL OFFSET="0" BITS="16" FORMAT="SIGNED" UNITS_PER_MV="400" '
        f'DURATION="{MORTARA_RHYTHM_N}" SAMPLE_FREQ="500" NAME="{name}" ENCODING="BASE64" '
        f'DATA="{_mortara_b64(i, MORTARA_RHYTHM_N)}" />\n'
        for i, name in enumerate(("I", "II", "V1"))
    )
    + '</ECG>\n'
)


# ---------------------------------------------------------------------------
# ISHNE Holter binary fixture helpers
# ---------------------------------------------------------------------------

def create_ishne_binary(
    nleads: int = 2,
    sr: int = 200,
    samples_per_lead: int = 400,
) -> bytes:
    """Build a minimal valid ISHNE Holter binary file in memory."""
    import struct
    import datetime as dt

    var_block = b""
    var_block_size = len(var_block)
    ecg_block_offset = 522 + var_block_size
    ecg_size = samples_per_lead  # per-lead sample count

    header = bytearray(512)

    # var_block_size (offset 0 in header = offset 10 in file)
    struct.pack_into("<i", header, 0, var_block_size)
    # ecg_size
    struct.pack_into("<i", header, 4, ecg_size)
    # var_block_offset
    struct.pack_into("<i", header, 8, 522)
    # ecg_block_offset
    struct.pack_into("<i", header, 12, ecg_block_offset)
    # file_version
    struct.pack_into("<h", header, 16, 1)
    # first_name (offset 18 in header = 28 in file)
    header[18:18+4] = b"Test"
    # last_name (offset 58 in header = 68 in file)
    header[58:58+3] = b"Ecg"
    # id (offset 98 in header = 108 in file)
    header[98:98+5] = b"P0001"
    # sex (offset 118 = 128 in file): 1=male
    struct.pack_into("<h", header, 118, 1)
    # race
    struct.pack_into("<h", header, 120, 0)
    # birth_date: day=15, month=6, year=1980 (offset 122 = 132)
    struct.pack_into("<hhh", header, 122, 15, 6, 1980)
    # record_date: day=1, month=12, year=2023 (offset 128 = 138)
    struct.pack_into("<hhh", header, 128, 1, 12, 2023)
    # file_date (offset 134 = 144)
    struct.pack_into("<hhh", header, 134, 1, 12, 2023)
    # start_time: 10, 30, 0 (offset 140 = 150)
    struct.pack_into("<hhh", header, 140, 10, 30, 0)
    # nleads (offset 146 = 156)
    struct.pack_into("<h", header, 146, nleads)
    # lead_spec (offset 148 = 158): I=5, II=6
    for i in range(nleads):
        struct.pack_into("<h", header, 148 + i * 2, 5 + i)
    for i in range(12 - nleads):
        struct.pack_into("<h", header, 148 + nleads * 2 + i * 2, -9)
    # lead_quality (offset 172 = 182)
    for i in range(nleads):
        struct.pack_into("<h", header, 172 + i * 2, 1)
    for i in range(12 - nleads):
        struct.pack_into("<h", header, 172 + nleads * 2 + i * 2, -9)
    # ampl_res in nV/count (offset 196 = 206): e.g. 1000 nV = 1 µV
    for i in range(nleads):
        struct.pack_into("<h", header, 196 + i * 2, 1000)
    for i in range(12 - nleads):
        struct.pack_into("<h", header, 196 + nleads * 2 + i * 2, -9)
    # pacemaker (offset 220 = 230)
    struct.pack_into("<h", header, 220, 0)
    # recorder_type (offset 222 = 232)
    header[222:222+7] = b"digital"
    # sr (offset 262 = 272)
    struct.pack_into("<h", header, 262, sr)

    # Build ECG data: interleaved int16 samples
    data = np.zeros(nleads * samples_per_lead, dtype=np.int16)
    for lead_idx in range(nleads):
        for s in range(samples_per_lead):
            data[s * nleads + lead_idx] = np.int16((lead_idx + 1) * 100 + (s % 50))

    # Compute CRC-CCITT (0xFFFF initial) of header — inline to avoid PyCRC dep in tests
    header_bytes = bytes(header) + var_block
    crc = 0xFFFF
    for byte in header_bytes:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    checksum = np.uint16(crc)

    # Assemble file
    pre_header = b"ISHNE1.0" + checksum.tobytes()
    return pre_header + header_bytes + data.tobytes()


@pytest.fixture
def hl7_aecg_file(tmp_path: Path) -> Path:
    """Write a minimal HL7 aECG XML file and return its path."""
    p = tmp_path / "test_hl7.xml"
    p.write_text(HL7_AECG_XML, encoding="utf-8")
    return p


@pytest.fixture
def mortara_file(tmp_path: Path) -> Path:
    """Write a minimal Mortara EL250 XML file and return its path."""
    p = tmp_path / "test_mortara.xml"
    p.write_text(MORTARA_XML, encoding="utf-8")
    return p


@pytest.fixture
def ishne_file(tmp_path: Path) -> Path:
    """Write a minimal ISHNE Holter binary file and return its path."""
    p = tmp_path / "test_holter.ecg"
    p.write_bytes(create_ishne_binary())
    return p


# ---------------------------------------------------------------------------
# Minimal EDF fixture
# ---------------------------------------------------------------------------

def create_edf_binary(
    num_signals: int = 2,
    num_records: int = 2,
    samples_per_record: int = 500,
    sampling_rate: int = 500,
) -> bytes:
    """Build a minimal valid EDF file in memory."""
    record_duration = samples_per_record / sampling_rate  # 1 second

    # Main header (256 bytes)
    hdr = bytearray(256)
    hdr[0:8] = b"0       "                                          # version
    hdr[8:88] = b"P001 M 15-JUN-1980 TestPatient" + b" " * 50      # patient
    hdr[8:88] = f"{'P001 M 15-JUN-1980 TestPatient':<80}".encode("ascii")
    hdr[88:168] = f"{'Startdate 01-DEC-2023 TestRecording':<80}".encode("ascii")
    hdr[168:176] = b"01.12.23"                                      # start date
    hdr[176:184] = b"10.30.00"                                      # start time
    header_bytes = 256 + num_signals * 256
    hdr[184:192] = f"{header_bytes:<8}".encode("ascii")
    hdr[192:236] = f"{'EDF+C':<44}".encode("ascii")                 # reserved (EDF+)
    hdr[236:244] = f"{num_records:<8}".encode("ascii")
    hdr[244:252] = f"{record_duration:<8}".encode("ascii")
    hdr[252:256] = f"{num_signals:<4}".encode("ascii")

    # Signal sub-headers
    labels = [f"{'ECG I':<16}", f"{'ECG II':<16}"][:num_signals]
    transducer = [f"{'AgAgCl':<80}"] * num_signals
    phys_dim = [f"{'mV':<8}"] * num_signals
    phys_min = [f"{-3.2:<8}"] * num_signals
    phys_max = [f"{3.2:<8}"] * num_signals
    dig_min = [f"{-32768:<8}"] * num_signals
    dig_max = [f"{32767:<8}"] * num_signals
    prefilter = [f"{'':<80}"] * num_signals
    samps = [f"{samples_per_record:<8}"] * num_signals
    reserved_sig = [f"{'':<32}"] * num_signals

    sig_hdr = bytearray()
    for group in [labels, transducer, phys_dim, phys_min, phys_max,
                  dig_min, dig_max, prefilter, samps, reserved_sig]:
        for item in group:
            sig_hdr.extend(item.encode("ascii"))

    # Data records: interleaved per record
    data = bytearray()
    for rec in range(num_records):
        for sig in range(num_signals):
            for s in range(samples_per_record):
                val = np.int16((sig + 1) * 100 + (s % 50))
                data.extend(struct.pack("<h", val))

    return bytes(hdr) + bytes(sig_hdr) + bytes(data)


@pytest.fixture
def edf_file(tmp_path: Path) -> Path:
    """Write a minimal EDF file and return its path."""
    p = tmp_path / "test.edf"
    p.write_bytes(create_edf_binary())
    return p


# ---------------------------------------------------------------------------
# SCP-ECG fixture (EN 1064 / ISO 11073-91064 layout, real CRCs)
# ---------------------------------------------------------------------------

def scp_crc(data: bytes) -> int:
    """CRC-CCITT (polynomial 0x1021, initial value 0xFFFF) used by SCP-ECG."""
    return binascii.crc_hqx(data, 0xFFFF)


def scp_section(sec_id: int, body: bytes, protocol: int = 20, reserved: bytes = b"\x00" * 6) -> bytes:
    """Build one section: CRC, ID, length, version, protocol, reserved, body."""
    body = bytes(body)
    if len(body) % 2:
        body += b"\x00"  # sections have an even length
    rest = struct.pack("<HIBB", sec_id, 16 + len(body), protocol, protocol) + reserved + body
    return struct.pack("<H", scp_crc(rest)) + rest


def scp_tag(tag: int, value: bytes) -> bytes:
    """Section 1 tagged field."""
    return bytes([tag]) + struct.pack("<H", len(value)) + value


def scp_huffman_encode(values) -> bytes:
    """Encode integers with the SCP default Huffman table."""
    bits = []
    for v in values:
        v = int(v)
        if v == 0:
            bits.append("0")
        elif -8 <= v <= 8:
            bits.append("1" * abs(v) + "0" + ("0" if v > 0 else "1"))
        elif -128 <= v <= 127:
            bits.append("1111111110" + format(v & 0xFF, "08b"))
        else:
            bits.append("1111111111" + format(v & 0xFFFF, "016b"))
    stream = "".join(bits)
    stream += "0" * (-len(stream) % 8)
    return bytes(int(stream[i:i + 8], 2) for i in range(0, len(stream), 8))


def scp_encode_differences(values, diff: int) -> list[int]:
    """Apply SCP first (1) or second (2) difference encoding (16-bit arithmetic)."""
    x = [int(v) for v in values]
    if diff == 1:
        x = x[:1] + [x[i] - x[i - 1] for i in range(1, len(x))]
    elif diff == 2:
        x = x[:2] + [x[i] - 2 * x[i - 1] + x[i - 2] for i in range(2, len(x))]
    return [(v + 0x8000) % 0x10000 - 0x8000 for v in x]


def scp_signal_data(avm: int, interval: int, diff: int, signals, huffman: bool, flag: int = 0) -> bytes:
    """Body of Section 5 or 6: header, per-lead byte counts, encoded leads."""
    blocks = []
    for sig in signals:
        enc = scp_encode_differences(sig, diff)
        blocks.append(scp_huffman_encode(enc) if huffman else struct.pack(f"<{len(enc)}h", *enc))
    head = struct.pack("<HHBB", avm, interval, diff, flag)
    return head + b"".join(struct.pack("<H", len(b)) for b in blocks) + b"".join(blocks)


def scp_default_section1() -> bytes:
    return (
        scp_tag(0, b"TestSCP\x00")
        + scp_tag(1, b"Jos\xe9\x00")  # Latin-1 text
        + scp_tag(2, b"SCP001\x00")
        + scp_tag(4, struct.pack("<HB", 43, 1))  # 43 years
        + scp_tag(5, struct.pack("<HBB", 1980, 6, 15))
        + scp_tag(6, struct.pack("<HB", 180, 1))  # cm
        + scp_tag(7, struct.pack("<HB", 75, 1))  # kg
        + scp_tag(8, bytes([1]))  # male
        + scp_tag(9, bytes([1]))  # caucasian
        + scp_tag(25, struct.pack("<HBB", 2024, 3, 9))
        + scp_tag(26, bytes([14, 30, 5]))
        + scp_tag(27, struct.pack("<H", 5))  # 0.05 Hz
        + scp_tag(28, struct.pack("<H", 150))  # 150 Hz
        + scp_tag(29, bytes([0x02]))  # 50 Hz notch
        + scp_tag(255, b"")
    )


def scp_test_signal(lead_index: int, samples: int) -> np.ndarray:
    return np.array([(lead_index + 1) * 100 + (s % 50) for s in range(samples)], dtype=np.int64)


def create_scp_ecg_binary(
    num_leads: int = 2,
    samples_per_lead: int = 500,
    sampling_rate: int = 500,
    *,
    lead_codes: list[int] | None = None,
    section1: bytes | None = None,
    avm: int = 1000,
    diff: int = 0,
    huffman: bool = False,
    signals: list | None = None,
    lead_flags: int = 0x04,
    sections: dict[int, bytes] | None = None,
    protocol: int = 20,
) -> bytes:
    """Build an SCP-ECG record following EN 1064.

    Defaults: leads I and II (codes 1, 2), uncompressed amplitudes, AVM
    1000 nV, Sections 0, 1, 3 and 6. *sections* adds or replaces section
    bodies by ID (e.g. 4, 5, 7, 8).
    """
    codes = lead_codes if lead_codes is not None else list(range(1, num_leads + 1))
    if signals is None:
        signals = [scp_test_signal(i, samples_per_lead) for i in range(len(codes))]
    interval = round(1_000_000 / sampling_rate)

    sec3 = bytes([len(codes), lead_flags]) + b"".join(
        struct.pack("<IIB", 1, len(sig), code) for code, sig in zip(codes, signals)
    )
    bodies = {
        1: scp_default_section1() if section1 is None else section1,
        3: sec3,
        6: scp_signal_data(avm, interval, diff, signals, huffman),
    }
    if huffman:
        bodies[2] = struct.pack("<H", 19999)
    bodies.update(sections or {})
    built = {sid: scp_section(sid, body, protocol) for sid, body in bodies.items()}

    # Section 0 holds one pointer per section 0 to 11 (empty ones are zero)
    sec0_len = 16 + 10 * 12
    pos = 6 + sec0_len + 1  # 1-based byte index
    pointers = struct.pack("<HII", 0, sec0_len, 7)
    offsets = {}
    for sid in sorted(built):
        offsets[sid] = pos
        pos += len(built[sid])
    for sid in range(1, 12):
        if sid in built:
            pointers += struct.pack("<HII", sid, len(built[sid]), offsets[sid])
        else:
            pointers += struct.pack("<HII", sid, 0, 0)
    sec0 = scp_section(0, pointers, protocol, reserved=b"SCPECG")
    body = sec0 + b"".join(built[sid] for sid in sorted(built))
    rest = struct.pack("<I", 6 + len(body)) + body
    return struct.pack("<H", scp_crc(rest)) + rest


@pytest.fixture
def scp_ecg_file(tmp_path: Path) -> Path:
    """Write a minimal SCP-ECG binary file and return its path."""
    p = tmp_path / "test.scp"
    p.write_bytes(create_scp_ecg_binary())
    return p


# ---------------------------------------------------------------------------
# GE MUSE XML fixture (element names and order per MUSE restecg.dtd)
# ---------------------------------------------------------------------------


def _muse_lead(lead_id: str, samples: list[int]) -> str:
    """Build a LeadData element: Base64 little-endian int16 with CRC32."""
    import zlib

    payload = np.asarray(samples, dtype="<i2").tobytes()
    return (
        "    <LeadData>\n"
        f"      <LeadByteCountTotal>{len(payload)}</LeadByteCountTotal>\n"
        "      <LeadTimeOffset>0</LeadTimeOffset>\n"
        f"      <LeadSampleCountTotal>{len(samples)}</LeadSampleCountTotal>\n"
        "      <LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>\n"
        "      <LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>\n"
        "      <LeadHighLimit>32767</LeadHighLimit>\n"
        "      <LeadLowLimit>-32768</LeadLowLimit>\n"
        f"      <LeadID>{lead_id}</LeadID>\n"
        "      <LeadOffsetFirstSample>0</LeadOffsetFirstSample>\n"
        "      <FirstSampleBaseline>0</FirstSampleBaseline>\n"
        "      <LeadSampleSize>2</LeadSampleSize>\n"
        "      <LeadOff>FALSE</LeadOff>\n"
        "      <BaselineSway>FALSE</BaselineSway>\n"
        f"      <LeadDataCRC32>{zlib.crc32(payload)}</LeadDataCRC32>\n"
        f"      <WaveFormData>{base64.b64encode(payload).decode()}</WaveFormData>\n"
        "    </LeadData>\n"
    )


MUSE_RHYTHM_I = [100, 50, -20, 0]
MUSE_RHYTHM_II = [60, 80, 10, -4]


def _muse_waveform(kind: str, leads: dict[str, list[int]]) -> str:
    return (
        "  <Waveform>\n"
        f"    <WaveformType>{kind}</WaveformType>\n"
        "    <WaveformStartTime>0</WaveformStartTime>\n"
        f"    <NumberofLeads>{len(leads)}</NumberofLeads>\n"
        "    <SampleType>CONTINUOUS_SAMPLES</SampleType>\n"
        "    <SampleBase>500</SampleBase>\n"
        "    <SampleExponent>0</SampleExponent>\n"
        "    <HighPassFilter>16</HighPassFilter>\n"
        "    <LowPassFilter>150</LowPassFilter>\n"
        "    <ACFilter>60</ACFilter>\n"
        + "".join(_muse_lead(k, v) for k, v in leads.items())
        + "  </Waveform>\n"
    )


GE_MUSE_XML = (
    '<?xml version="1.0" encoding="ISO-8859-1"?>\n'
    '<!DOCTYPE RestingECG SYSTEM "restecg.dtd">\n'
    "<RestingECG>\n"
    "  <MuseInfo>\n"
    "    <MuseVersion>9.0.7.17363</MuseVersion>\n"
    "  </MuseInfo>\n"
    "  <PatientDemographics>\n"
    "    <PatientID>MUSE001</PatientID>\n"
    "    <PatientAge>38</PatientAge>\n"
    "    <AgeUnits>YEARS</AgeUnits>\n"
    "    <DateofBirth>03-15-1985</DateofBirth>\n"
    "    <Gender>FEMALE</Gender>\n"
    "    <Race>CAUCASIAN</Race>\n"
    "    <HeightCM>165</HeightCM>\n"
    "    <WeightKG>60</WeightKG>\n"
    "    <PatientLastName>Smith</PatientLastName>\n"
    "    <PatientFirstName>Jane</PatientFirstName>\n"
    "  </PatientDemographics>\n"
    "  <TestDemographics>\n"
    "    <DataType>RESTING</DataType>\n"
    "    <Site>1</Site>\n"
    "    <SiteName>Main Hospital</SiteName>\n"
    "    <AcquisitionDevice>MAC5500</AcquisitionDevice>\n"
    "    <Status>UNCONFIRMED</Status>\n"
    "    <Location>12</Location>\n"
    "    <LocationName>Cardiology</LocationName>\n"
    "    <RoomID>4B</RoomID>\n"
    "    <AcquisitionTime>14:30:00</AcquisitionTime>\n"
    "    <AcquisitionDate>12-01-2023</AcquisitionDate>\n"
    "    <CartNumber>7</CartNumber>\n"
    "    <AcquisitionSoftwareVersion>010A</AcquisitionSoftwareVersion>\n"
    "    <ReferringMDLastName>House</ReferringMDLastName>\n"
    "    <ReferringMDFirstName>Greg</ReferringMDFirstName>\n"
    "    <AcquisitionTechLastName>Tech</AcquisitionTechLastName>\n"
    "    <AcquisitionTechFirstName>Alex</AcquisitionTechFirstName>\n"
    "  </TestDemographics>\n"
    "  <RestingECGMeasurements>\n"
    "    <VentricularRate>72</VentricularRate>\n"
    "    <AtrialRate>72</AtrialRate>\n"
    "    <PRInterval>160</PRInterval>\n"
    "    <QRSDuration>88</QRSDuration>\n"
    "    <QTInterval>380</QTInterval>\n"
    "    <QTCorrected>416</QTCorrected>\n"
    "    <PAxis>51</PAxis>\n"
    "    <RAxis>20</RAxis>\n"
    "    <TAxis>40</TAxis>\n"
    "    <QRSCount>12</QRSCount>\n"
    "    <ECGSampleBase>500</ECGSampleBase>\n"
    "    <ECGSampleExponent>0</ECGSampleExponent>\n"
    "    <QTcFrederica>404</QTcFrederica>\n"
    "  </RestingECGMeasurements>\n"
    "  <Diagnosis>\n"
    "    <Modality>RESTING</Modality>\n"
    "    <DiagnosisStatement>\n"
    "      <StmtFlag>ENDSLINE</StmtFlag>\n"
    "      <StmtText>Normal sinus rhythm</StmtText>\n"
    "    </DiagnosisStatement>\n"
    "    <DiagnosisStatement>\n"
    "      <StmtFlag>ENDSLINE</StmtFlag>\n"
    "      <StmtText>Normal ECG</StmtText>\n"
    "    </DiagnosisStatement>\n"
    "  </Diagnosis>\n"
    "  <QRSTimesTypes>\n"
    "    <QRS><Number>1</Number><Type>0</Type><Time>614</Time></QRS>\n"
    "    <GlobalRR>833</GlobalRR>\n"
    "  </QRSTimesTypes>\n"
    + _muse_waveform("Median", {"I": [1, 2, 3], "II": [3, 2, 1]})
    + _muse_waveform("Rhythm", {"I": MUSE_RHYTHM_I, "II": MUSE_RHYTHM_II})
    + "</RestingECG>\n"
)


@pytest.fixture
def ge_muse_xml_file(tmp_path: Path) -> Path:
    """Write a minimal GE MUSE XML file and return its path."""
    p = tmp_path / "test_muse.xml"
    p.write_text(GE_MUSE_XML, encoding="iso-8859-1")
    return p


# ---------------------------------------------------------------------------
# DICOM Waveform fixtures (PS3.3 C.10.9, 12-lead ECG IOD)
# ---------------------------------------------------------------------------

DICOM_12LEAD_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.9.1.1"

# MDC codes and meanings from PS3.16 CID 3001
DICOM_LEAD_SOURCES = [
    ("2:1", "Lead I"), ("2:2", "Lead II"), ("2:61", "Lead III"),
    ("2:62", "aVR, augmented voltage, right"),
    ("2:63", "aVL, augmented voltage, left"),
    ("2:64", "aVF, augmented voltage, foot"),
    ("2:3", "Lead V1"), ("2:4", "Lead V2"), ("2:5", "Lead V3"),
    ("2:6", "Lead V4"), ("2:7", "Lead V5"), ("2:8", "Lead V6"),
]


def dicom_code(value: str, scheme: str, meaning: str):
    """Return a single code sequence item."""
    from pydicom.dataset import Dataset

    item = Dataset()
    item.CodeValue = value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return item


def dicom_channel(
    source: tuple[str, str] | None,
    sensitivity: str | None = "2.5",
    unit: tuple[str, str] | None = ("uV", "microvolt"),
    baseline: str | None = "0",
    correction: str | None = "1",
    filters: tuple[str, str, str] | None = ("0.05", "150", "50"),
):
    """Build one Channel Definition Sequence item."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    ch = Dataset()
    if source is not None:
        ch.ChannelSourceSequence = Sequence([dicom_code(source[0], "MDC", source[1])])
    if sensitivity is not None:
        ch.ChannelSensitivity = sensitivity
    if unit is not None:
        ch.ChannelSensitivityUnitsSequence = Sequence([dicom_code(unit[0], "UCUM", unit[1])])
    if correction is not None:
        ch.ChannelSensitivityCorrectionFactor = correction
    if baseline is not None:
        ch.ChannelBaseline = baseline
    ch.WaveformBitsStored = 16
    if filters is not None:
        ch.FilterLowFrequency, ch.FilterHighFrequency, ch.NotchFilterFrequency = filters
    return ch


def dicom_group(
    samples: np.ndarray,
    channels: list,
    sampling_rate: str = "500",
    interpretation: str = "SS",
    bits: int = 16,
    label: str = "RHYTHM",
    data: bytes | None = None,
    padding: bytes | None = None,
):
    """Build one multiplex group; *samples* has shape (samples, channels)."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    wf = Dataset()
    wf.MultiplexGroupLabel = label
    wf.WaveformOriginality = "ORIGINAL"
    wf.NumberOfWaveformChannels = samples.shape[1]
    wf.NumberOfWaveformSamples = samples.shape[0]
    wf.SamplingFrequency = sampling_rate
    wf.ChannelDefinitionSequence = Sequence(channels)
    wf.WaveformBitsAllocated = bits
    wf.WaveformSampleInterpretation = interpretation
    if padding is not None:
        wf.WaveformPaddingValue = padding
    # Interleaved by channel then by sample (C.10.9.1.5)
    wf.WaveformData = data if data is not None else samples.tobytes()
    return wf


def write_dicom_ecg(
    path: Path,
    groups: list,
    annotations: list | None = None,
    little_endian: bool = True,
    **attrs,
) -> Path:
    """Write a 12-lead ECG Waveform Storage file with the given groups."""
    pytest.importorskip("pydicom")
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRBigEndian, ExplicitVRLittleEndian

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = DICOM_12LEAD_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian if little_endian else ExplicitVRBigEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = DICOM_12LEAD_SOP_CLASS
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "ECG"
    for key, value in attrs.items():
        setattr(ds, key, value)
    ds.WaveformSequence = Sequence(groups)
    if annotations:
        ds.WaveformAnnotationSequence = Sequence(annotations)
    ds.save_as(str(path), enforce_file_format=True, little_endian=little_endian, implicit_vr=False)
    return path


def dicom_12lead_samples(num_samples: int = 1000) -> np.ndarray:
    """Deterministic int16 test signal of shape (samples, 12)."""
    s = np.arange(num_samples)[:, None]
    ch = np.arange(12)[None, :]
    return ((ch + 1) * 100 + (s % 50) - 25 * (ch % 2)).astype("<i2")


@pytest.fixture
def dicom_file(tmp_path: Path) -> Path:
    """Minimal DICOM waveform: 2 channels, 100 samples, no units."""
    num_samples = 100
    samples = np.zeros((num_samples, 2), dtype="<i2")
    samples[:, 0] = 100 + np.arange(num_samples) % 50
    samples[:, 1] = 200 + np.arange(num_samples) % 50
    channels = [
        dicom_channel(("2:1", "Lead I"), sensitivity="1.0", unit=None, filters=None),
        dicom_channel(("2:2", "Lead II"), sensitivity="1.0", unit=None, filters=None),
    ]
    return write_dicom_ecg(
        tmp_path / "test.dcm",
        [dicom_group(samples, channels)],
        PatientName="Doe^John",
        PatientID="DCM001",
        PatientSex="M",
        PatientBirthDate="19900101",
        StudyDate="20231201",
        StudyTime="103000",
    )


@pytest.fixture
def dicom_file_baseline(tmp_path: Path) -> Path:
    """One channel, sensitivity 2.5 uV, baseline 5 uV, constant sample 100."""
    samples = np.full((10, 1), 100, dtype="<i2")
    channels = [dicom_channel(("2:1", "Lead I"), sensitivity="2.5", baseline="5", filters=None)]
    return write_dicom_ecg(
        tmp_path / "test_baseline.dcm", [dicom_group(samples, channels)], PatientID="DCMBL",
    )


@pytest.fixture
def dicom_12lead_file(tmp_path: Path) -> Path:
    """12-lead ECG with a rhythm group, a median group and annotations."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    rhythm = dicom_group(
        dicom_12lead_samples(1000),
        [dicom_channel(src, sensitivity="2.5", correction="1.02") for src in DICOM_LEAD_SOURCES],
        label="RHYTHM",
    )
    median = dicom_group(
        dicom_12lead_samples(600),
        [dicom_channel(src, sensitivity="2.5") for src in DICOM_LEAD_SOURCES],
        sampling_rate="1000",
        label="MEDIAN BEAT",
    )

    def measurement(code, meaning, value, unit):
        a = Dataset()
        a.ConceptNameCodeSequence = Sequence([dicom_code(code, "MDC", meaning)])
        a.NumericValue = value
        a.MeasurementUnitsCodeSequence = Sequence([dicom_code(unit, "UCUM", unit)])
        a.AnnotationGroupNumber = 1
        return a

    def text(value):
        a = Dataset()
        a.UnformattedTextValue = value
        a.AnnotationGroupNumber = 2
        return a

    fiducial = Dataset()
    fiducial.ConceptNameCodeSequence = Sequence([dicom_code("2:15844", "MDC", "QRS onset")])
    fiducial.ReferencedWaveformChannels = [1, 0]
    fiducial.TemporalRangeType = "POINT"
    fiducial.ReferencedSamplePositions = [120]

    annotations = [
        measurement("2:16770", "ECG Heart Rate", "72", "/min"),
        measurement("2:15872", "PR interval global", "0.1604", "s"),
        measurement("2:16156", "QRS duration global", "98", "ms"),
        measurement("2:16160", "QT interval global", "398", "ms"),
        measurement("2:15880", "QTc global using Bazett formula", "421.6", "ms"),
        measurement("2:16132", "QRS axis", "45", "deg"),
        text("Sinus rhythm"),
        text("Normal ECG"),
        fiducial,
    ]
    return write_dicom_ecg(
        tmp_path / "ecg12.dcm",
        [rhythm, median],
        annotations,
        PatientName="Doe^Jane^Q",
        PatientID="DCM12",
        PatientSex="F",
        PatientBirthDate="19800615",
        PatientAge="099Y",
        PatientWeight="61.5",
        PatientSize="1.68",
        AcquisitionDateTime="20240102030405.250000+0100",
        StudyDate="20240101",
        StudyTime="000000",
        InstanceCreationDate="20240103",
        Manufacturer="ACME",
        ManufacturerModelName="ECG-1",
        StationName="CART7",
        DeviceSerialNumber="SN42",
        SoftwareVersions=["1.2", "3.4"],
        InstitutionName="General Hospital",
        InstitutionalDepartmentName="Cardiology",
        OperatorsName="Tech^Terry",
        ReferringPhysicianName="Ref^Rita",
    )


# ---------------------------------------------------------------------------
# Minimal WFDB fixture
# ---------------------------------------------------------------------------

def create_wfdb_files(directory: Path) -> Path:
    """Create minimal WFDB .hea + .dat files and return the .hea path."""
    num_signals = 2
    num_samples = 500
    sampling_rate = 500
    gain = 200.0

    # Build .dat file: Format 16, interleaved int16
    dat = np.zeros(num_samples * num_signals, dtype="<i2")
    for s in range(num_samples):
        dat[s * num_signals + 0] = np.int16(100 + (s % 50))
        dat[s * num_signals + 1] = np.int16(200 + (s % 50))

    dat_path = directory / "test_wfdb.dat"
    dat_path.write_bytes(dat.tobytes())

    # Build .hea file; checksum is the 16-bit signed sum of each signal
    def checksum(ch: int) -> int:
        total = int(dat[ch::num_signals].astype(np.int64).sum()) & 0xFFFF
        return total - 0x10000 if total >= 0x8000 else total

    hea_lines = [
        f"test_wfdb {num_signals} {sampling_rate} {num_samples} 10:30:00 01/12/2023",
        f"test_wfdb.dat 16 {gain}(0)/mV 12 0 {dat[0]} {checksum(0)} 0 I",
        f"test_wfdb.dat 16 {gain}(0)/mV 12 0 {dat[1]} {checksum(1)} 0 II",
        "# Age: 43",
        "# Sex: M",
    ]
    hea_path = directory / "test_wfdb.hea"
    hea_path.write_text("\n".join(hea_lines) + "\n", encoding="ascii")

    return hea_path


@pytest.fixture
def wfdb_file(tmp_path: Path) -> Path:
    """Create a minimal WFDB file set and return the .hea path."""
    return create_wfdb_files(tmp_path)


# ---------------------------------------------------------------------------
# MFER fixtures (MFER Part 1 v1.05 / ISO 22077-1)
# ---------------------------------------------------------------------------

def mfer_tlv(tag: int, value: bytes = b"") -> bytes:
    """Encode one MFER definition (tag, BER length, value)."""
    length = len(value)
    if length < 0x80:
        return bytes([tag, length]) + value
    len_bytes = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([tag, 0x80 | len(len_bytes)]) + len_bytes + value


def mfer_att(channel: int, *definitions: bytes, indefinite: bool = False) -> bytes:
    """Encode MWF_ATT (0x3F) channel attributes for a 0-based channel <= 127."""
    body = b"".join(definitions)
    if indefinite:
        return bytes([0x3F, channel, 0x80]) + body + b"\x00\x00"
    return bytes([0x3F, channel]) + mfer_tlv(0, body)[1:]


def mfer_scaled(unit: int, exponent: int, mantissa: int, size: int = 2, big: bool = True) -> bytes:
    """MWF_IVL / MWF_SEN value: unit code, signed exponent, signed mantissa."""
    order = "big" if big else "little"
    return bytes([unit, exponent & 0xFF]) + mantissa.to_bytes(size, order, signed=True)


def mfer_frame_data(channels: list[np.ndarray], block: int, dtype: str = ">i2") -> bytes:
    """Lay out samples as sequence x channel x block (Part 1 Fig. 5-16)."""
    n = len(channels[0])
    out = bytearray()
    for start in range(0, n, block):
        for ch in channels:
            out += np.asarray(ch[start:start + block], dtype=dtype).tobytes()
    return bytes(out)


def mfer_test_signal(num_channels: int, num_samples: int) -> list[np.ndarray]:
    """Deterministic int16 test signal per channel."""
    s = np.arange(num_samples)
    return [((ch + 1) * 100 + (s % 50) - 20 * ch).astype(np.int16) for ch in range(num_channels)]


def create_mfer_binary(
    num_channels: int = 2,
    num_samples: int = 500,
    sampling_rate: int = 500,
    block: int = 50,
) -> bytes:
    """Build an MFER standard 12-lead style file following Part 1 Annex A."""
    interval_ms = 1000 // sampling_rate
    lead_codes = [1, 2, 61, 62, 63, 64, 3, 4, 5, 6, 7, 8]
    parts = [
        mfer_tlv(0x40, b"MFR " + b"Standard 12 leads ECG".ljust(28, b" ")),  # MWF_PRE
        mfer_tlv(0x01, b"\x00"),                                # MWF_BLE big endian
        mfer_tlv(0x02, bytes([1, 5, 0])),                       # MWF_VER 1.5.0
        mfer_tlv(0x03, b"UTF-8"),                               # MWF_TXC
        mfer_tlv(0x17, b"ACME^ECG-2003^1.02.33^SN-77"),         # MWF_MAN
        mfer_tlv(0x08, struct.pack(">H", 1)),                   # MWF_WFM standard 12-lead
        mfer_tlv(0x0B, mfer_scaled(1, -3, interval_ms)),        # MWF_IVL interval in ms
        mfer_tlv(0x0C, mfer_scaled(0, -6, 5)),                  # MWF_SEN 5 uV
        mfer_tlv(0x0A, b"\x00"),                                # MWF_DTP int16
        mfer_tlv(0x04, struct.pack(">I", block)),               # MWF_BLK
        mfer_tlv(0x05, struct.pack(">I", num_channels)),        # MWF_CHN
        mfer_tlv(0x06, struct.pack(">I", num_samples // block)),  # MWF_SEQ
    ]
    for ch in range(num_channels):
        parts.append(mfer_att(ch, mfer_tlv(0x09, bytes([lead_codes[ch]]))))  # MWF_LDN
    parts += [
        mfer_tlv(0x11, b"HPF=0.05"),                            # MWF_FLT
        mfer_tlv(0x11, b"LPF=150^2nd order Butterworth"),
        mfer_tlv(0x11, b"BEF=50^Hum filter"),
        mfer_tlv(0x16, "Comment é".encode("utf-8")),            # MWF_NTE
        mfer_tlv(0x81, b"Doe^DOE^John^JOHN"),                   # MWF_PNM
        mfer_tlv(0x82, b"MFER001^EXAM9"),                       # MWF_PID
        mfer_tlv(0x83, bytes([99]) + struct.pack(">HHBB", 0, 1980, 6, 15)),  # MWF_AGE
        mfer_tlv(0x84, b"\x01"),                                # MWF_SEX male
        mfer_tlv(0x85, struct.pack(">HBBBBBHH", 2023, 12, 1, 10, 30, 0, 250, 0)),  # MWF_TIM
        mfer_tlv(0x41, struct.pack(">HII", 7, 100, 20) + b"Calibration"),  # MWF_EVT
        mfer_tlv(0x1E, mfer_frame_data(mfer_test_signal(num_channels, num_samples), block)),
        b"\x80\x00",                                            # MWF_END
    ]
    return b"".join(parts)


@pytest.fixture
def mfer_file(tmp_path: Path) -> Path:
    """Write a spec-compliant MFER file (2 channels, 500 Hz, 1 s)."""
    p = tmp_path / "test.mwf"
    p.write_bytes(create_mfer_binary())
    return p


# ---------------------------------------------------------------------------
# BeneHeart R12 XML fixture (no public spec: inferred layout, synthetic data)
# ---------------------------------------------------------------------------

def int16_b64(values) -> str:
    """Base64 of little-endian int16 *values*."""
    return base64.b64encode(np.asarray(values, dtype="<i2").tobytes()).decode()


BENEHEART_R12_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<BeneHeartR12>
  <PatientInfo>
    <PatientID>BH001</PatientID>
    <FirstName>Alice</FirstName>
    <LastName>Wonder</LastName>
    <Sex>F</Sex>
    <DateOfBirth>1992-03-20</DateOfBirth>
    <Age>31</Age>
    <Height Unit="cm">165</Height>
    <Weight Unit="kg">60</Weight>
  </PatientInfo>
  <AcquisitionInfo>
    <AcquisitionDate>2023-12-01</AcquisitionDate>
    <AcquisitionTime>09:15:00</AcquisitionTime>
    <SampleRate>500</SampleRate>
    <Device>BeneHeart R12</Device>
    <SerialNumber>R12-0001</SerialNumber>
    <SoftwareVersion>01.02</SoftwareVersion>
  </AcquisitionInfo>
  <FilterSettings>
    <HighPass>0.05</HighPass>
    <LowPass>150</LowPass>
    <Notch>0</Notch>
  </FilterSettings>
  <Leads>
    <Lead Name="I" Data="ZAAyAA=="/>
    <Lead Name="II" Data="ZAAyAA=="/>
  </Leads>
  <Measurements>
    <HeartRate>72.6</HeartRate>
    <PRInterval>150</PRInterval>
    <QTcBazett>410</QTcBazett>
  </Measurements>
  <Diagnosis>
    <Statement>Sinus rhythm</Statement>
    <Statement>Normal ECG</Statement>
  </Diagnosis>
</BeneHeartR12>
""")


@pytest.fixture
def beneheart_r12_file(tmp_path: Path) -> Path:
    """Write a minimal BeneHeart R12 XML file and return its path."""
    p = tmp_path / "test_beneheart.xml"
    p.write_text(BENEHEART_R12_XML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# GE MAC 2000 XML fixture (GE MUSE element conventions, synthetic data)
# ---------------------------------------------------------------------------

GE_MAC2000_RHYTHM = [np.arange(5000) % 100 - 50, np.arange(5000) % 80 - 40]
GE_MAC2000_MEDIAN = np.arange(600) % 60 - 30

GE_MAC2000_XML = textwrap.dedent(f"""\
<?xml version="1.0" encoding="UTF-8"?>
<MAC2000>
  <PatientDemographics>
    <PatientID>MAC001</PatientID>
    <PatientFirstName>Bob</PatientFirstName>
    <PatientLastName>Builder</PatientLastName>
    <Gender>MALE</Gender>
    <DateofBirth>07-22-1978</DateofBirth>
    <PatientAge>99</PatientAge>
    <AgeUnits>YEARS</AgeUnits>
    <PatientHeightCM>180</PatientHeightCM>
    <PatientWeightKG>82</PatientWeightKG>
  </PatientDemographics>
  <TestDemographics>
    <AcquisitionDevice>MAC2000</AcquisitionDevice>
    <AcquisitionSoftwareVersion>010A</AcquisitionSoftwareVersion>
    <AcquisitionDate>12-01-2023</AcquisitionDate>
    <AcquisitionTime>11:00:00</AcquisitionTime>
    <SiteName>General Hospital</SiteName>
    <LocationName>Cardiology</LocationName>
    <RoomID>4B</RoomID>
    <AcquisitionTechID>TK</AcquisitionTechID>
    <ReferringMDLastName>House</ReferringMDLastName>
    <EditorFirstName>Ann</EditorFirstName>
    <EditorLastName>Smith</EditorLastName>
    <EditDate>12-02-2023</EditDate>
    <EditTime>08:30:00</EditTime>
  </TestDemographics>
  <Waveform>
    <WaveformType>Median</WaveformType>
    <SampleBase>500</SampleBase>
    <LeadData>
      <LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>
      <LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>
      <LeadID>I</LeadID>
      <WaveFormData>{int16_b64(GE_MAC2000_MEDIAN)}</WaveFormData>
    </LeadData>
  </Waveform>
  <Waveform>
    <WaveformType>Rhythm</WaveformType>
    <SampleBase>500</SampleBase>
    <HighPassFilter>0.16</HighPassFilter>
    <LowPassFilter>150</LowPassFilter>
    <ACFilter>60</ACFilter>
    <LeadData>
      <LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>
      <LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>
      <LeadID>I</LeadID>
      <WaveFormData>{int16_b64(GE_MAC2000_RHYTHM[0])}</WaveFormData>
    </LeadData>
    <LeadData>
      <LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit>
      <LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>
      <LeadID>II</LeadID>
      <WaveFormData>{int16_b64(GE_MAC2000_RHYTHM[1])}</WaveFormData>
    </LeadData>
  </Waveform>
  <RestingECGMeasurements>
    <VentricularRate>68</VentricularRate>
    <PRInterval>160</PRInterval>
    <QRSDuration>92</QRSDuration>
    <QTInterval>380</QTInterval>
    <QTCorrected>404.6</QTCorrected>
    <PAxis>50</PAxis>
    <RAxis>-10</RAxis>
    <TAxis>30</TAxis>
  </RestingECGMeasurements>
  <OriginalDiagnosis>
    <DiagnosisStatement><StmtFlag>ENDSLINE</StmtFlag><StmtText>Sinus rhythm</StmtText></DiagnosisStatement>
    <DiagnosisStatement><StmtText>Borderline ECG</StmtText></DiagnosisStatement>
  </OriginalDiagnosis>
  <Diagnosis>
    <DiagnosisStatement><StmtFlag>ENDSLINE</StmtFlag><StmtText>Sinus rhythm</StmtText></DiagnosisStatement>
    <DiagnosisStatement><StmtText>Normal ECG</StmtText></DiagnosisStatement>
  </Diagnosis>
</MAC2000>
""")


@pytest.fixture
def ge_mac2000_file(tmp_path: Path) -> Path:
    """Write a GE MAC 2000 XML file and return its path."""
    p = tmp_path / "test_mac2000.xml"
    p.write_text(GE_MAC2000_XML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Minimal EDAN ARC Holter fixture
# ---------------------------------------------------------------------------

# Header layout follows https://paulbourke.net/dataformats/edan/
# The patient header is 3672 bytes with fields at fixed offsets.
_EDAN_HEA_SIZE = 3672


def create_edan_arc_hea(
    channel_count: int = 3,
    sampling_rate: int = 200,
    start_epoch: int = 1_701_421_800,   # 2023-12-01 09:10:00 UTC
    end_epoch: int = 1_701_421_805,     # +5 seconds (1000 samples at 200 Hz)
    lead_labels: tuple[str, ...] = ("I", "II", "III"),
    patient_name: bytes = b"Edan TestPatient",
) -> bytes:
    """Build a minimal EDAN patient.hea buffer in memory."""
    buf = bytearray(_EDAN_HEA_SIZE)
    # Timestamps at offsets 4 and 8
    struct.pack_into("<I", buf, 4, start_epoch)
    struct.pack_into("<I", buf, 8, end_epoch)
    # Channel count (byte 12)
    buf[12] = channel_count
    # Sampling rate (s16 at 32)
    struct.pack_into("<h", buf, 32, sampling_rate)
    # Height & weight (s16 at 60 and 64)
    struct.pack_into("<h", buf, 60, 175)
    struct.pack_into("<h", buf, 64, 70)
    # Telephone at 68
    buf[68:68 + 11] = b"555-1234567"
    # Patient ID at 108
    buf[108:108 + 8] = b"EDAN0001"
    # Diagnosis at 140
    buf[140:140 + 16] = b"Sinus rhythm    "
    # Medication at 242
    buf[242:242 + 7] = b"Aspirin"
    # Lead labels (8 bytes each from offset 1796)
    for i, lbl in enumerate(lead_labels[:channel_count]):
        encoded = lbl.encode("ascii")[:8]
        buf[1796 + i * 8:1796 + i * 8 + len(encoded)] = encoded
    # Department at 1960
    buf[1960:1960 + 10] = b"Cardiology"
    # Recorder ID at 2304
    buf[2304:2304 + 6] = b"SE2012"
    # Version at 2314
    buf[2314:2314 + 4] = b"1.0a"
    # Low-pass filter at 2596
    struct.pack_into("<h", buf, 2596, 75)
    # DFT filter at 2628
    buf[2628:2628 + 4] = b"50Hz"
    # Patient name at 2637 (up to the physician field at 2700)
    buf[2637:2637 + len(patient_name)] = patient_name
    # Physician name at 2700
    buf[2700:2700 + 6] = b"Dr Liu"
    # Technician at 2764
    buf[2764:2764 + 5] = b"TechA"
    # Medical history at 2892
    buf[2892:2892 + 12] = b"Hypertension"
    return bytes(buf)


def create_edan_arc_dat(
    channel_count: int = 3,
    samples_per_channel: int = 1000,
    stored_channels: int | None = None,
    pad_value: int = 0,
) -> bytes:
    """Build a minimal EDAN ecgraw.dat buffer (interleaved uint16, ADC zero 16384).

    *stored_channels* (default: *channel_count*) is the number of channels
    interleaved in the file; the reference states that the recorder always
    writes 12. Channels past *channel_count* are filled with *pad_value*.
    """
    stored = stored_channels or channel_count
    raw = np.full((samples_per_channel, stored), pad_value, dtype="<u2")
    s = np.arange(samples_per_channel)
    for ch in range(channel_count):
        # Centre on 16384 so the decoded signed value is small.
        raw[:, ch] = 16384 + (ch + 1) * 10 + (s % 50)
    return raw.tobytes()


@pytest.fixture
def edan_arc_dir(tmp_path: Path) -> Path:
    """Write minimal EDAN patient.hea + ecgraw.dat into ``tmp_path``."""
    hea_path = tmp_path / "patient.hea"
    dat_path = tmp_path / "ecgraw.dat"
    hea_path.write_bytes(create_edan_arc_hea())
    dat_path.write_bytes(create_edan_arc_dat())
    return hea_path


@pytest.fixture
def edan_arc_archive(tmp_path: Path) -> Path:
    """Write a minimal EDAN .arc bundle (filename markers + payloads) to disk."""
    hea = create_edan_arc_hea()
    dat = create_edan_arc_dat()
    # A plausible (invented) wrapper: filename literal + 8-byte size + payload.
    arc = bytearray()
    arc.extend(b"EDANARC\x00")  # fake archive magic
    for name, payload in [("patient.hea", hea), ("ecgraw.dat", dat)]:
        arc.extend(name.encode("ascii"))
        arc.extend(b"\x00")  # NUL terminator
        arc.extend(struct.pack("<Q", len(payload)))  # 8-byte size
        arc.extend(payload)
    p = tmp_path / "test.arc"
    p.write_bytes(bytes(arc))
    return p


def create_neutral_holter_arc(
    channels: int = 3,
    sampling_rate: int = 250,
    duration_s: int = 10,
    filename_stamp: str = "DT-06_05_2026-11_38_39",
) -> tuple[str, bytes]:
    """Build a minimal NEUTRAL HOLTER RECORDING .arc buffer in memory."""
    HEADER_SIZE = 0x1000
    header = bytearray(HEADER_SIZE)
    # Constant record-count u32 at offset 0
    struct.pack_into("<I", header, 0, 3)
    # Magic at offset 4
    sig = b"##NEUTRAL HOLTER RECORDING##"
    header[4:4 + len(sig)] = sig
    # UUID-style id at offset 0x36
    header[0x36:0x36 + 13] = b"6a0c3d93-3b15"
    # Filename literal at offset 0x5A
    header[0x5A:0x5A + 15] = b"patientdata.dat"
    # Index pointer at offset 0x56 — point past the ECG section
    samples_per_ch = duration_s * sampling_rate
    payload_bytes = samples_per_ch * channels * 2
    ann_offset = HEADER_SIZE + payload_bytes + 1024
    struct.pack_into("<I", header, 0x56, ann_offset)

    # ECG payload — small sinusoid that clearly stays under the std threshold
    t = np.arange(samples_per_ch) / sampling_rate
    payload = bytearray()
    for s in range(samples_per_ch):
        for ch in range(channels):
            val = int(20 * np.sin(2 * np.pi * 1.2 * t[s] + ch * np.pi / 4))
            payload.extend(struct.pack("<h", val))

    # Annotation tail — high-amplitude noise to ensure the std-jump scan
    # locates the boundary cleanly
    rng = np.random.default_rng(42)
    tail = rng.integers(-20000, 20000, size=8192, dtype=np.int16).tobytes()

    arc = bytes(header) + bytes(payload) + b"\x00" * 1024 + tail
    return f"{filename_stamp}.arc", arc


@pytest.fixture
def neutral_holter_arc_file(tmp_path: Path) -> Path:
    """Write a minimal NEUTRAL HOLTER .arc to disk."""
    name, blob = create_neutral_holter_arc()
    p = tmp_path / name
    p.write_bytes(blob)
    return p
