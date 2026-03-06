"""Shared test fixtures for ECGDataKit."""

from __future__ import annotations

import base64
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
# Minimal HL7 aECG XML fixture
# ---------------------------------------------------------------------------

HL7_AECG_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<AnnotatedECG>
  <id root="test-uuid-1234"/>
  <effectiveTime>
    <low value="20230615103000"/>
    <high value="20230615103010"/>
  </effectiveTime>
  <subject>
    <administrativeGenderCode code="M"/>
    <birthTime value="19800101"/>
  </subject>
  <component>
    <series>
      <component>
        <sequenceSet>
          <component>
            <sequence>
              <code code="MDC_ECG_LEAD_I"/>
              <value>
                <digits>100 200 300 400 500</digits>
              </value>
            </sequence>
          </component>
          <component>
            <sequence>
              <code code="MDC_ECG_LEAD_II"/>
              <value>
                <digits>110 210 310 410 510</digits>
              </value>
            </sequence>
          </component>
        </sequenceSet>
      </component>
    </series>
  </component>
</AnnotatedECG>
""")


# ---------------------------------------------------------------------------
# Minimal Mortara EL250 XML fixture
# ---------------------------------------------------------------------------

MORTARA_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<ECG ACQUISITION_TIME="20231201120000" ACQUISITION_TIME_XML="2023-12-01T12:00:00"
     NUM_QRS="75" VENT_RATE="75">
  <SOURCE MODEL="EL250" MANUFACTURER="MORTARA" TYPE="STD-12"
          ACQUIRING_DEVICE_SERIAL_NUMBER="SN123" LOW_PASS_FILTER="150"/>
  <DEMOGRAPHIC_FIELDS>
    <DEMOGRAPHIC_FIELD LABEL="First:" VALUE="John"/>
    <DEMOGRAPHIC_FIELD LABEL="Last:" VALUE="Doe"/>
    <DEMOGRAPHIC_FIELD LABEL="ID:" VALUE="PAT001"/>
    <DEMOGRAPHIC_FIELD LABEL="DOB:" VALUE="19900515"/>
    <DEMOGRAPHIC_FIELD LABEL="Sex:" VALUE="M"/>
    <DEMOGRAPHIC_FIELD LABEL="Age:" VALUE="33"/>
  </DEMOGRAPHIC_FIELDS>
  <CHANNEL NAME="I" DURATION="5000" SAMPLE_FREQ="500" DATA="ZAAyAA=="/>
  <CHANNEL NAME="II" DURATION="5000" SAMPLE_FREQ="500" DATA="ZAAyAA=="/>
  <TYPICAL_CYCLE BITS="16" FORMAT="signed" UNITS_PER_MV="200"
                 DURATION="400" SAMPLE_FREQ="500" ENCODING="base64"
                 PR_INTERVAL="160" QRS_DURATION="90">
    <TYPICAL_CYCLE_CHANNEL NAME="I" DATA="ZAAyAA=="/>
  </TYPICAL_CYCLE>
</ECG>
""")


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
    sample_rate: int = 500,
) -> bytes:
    """Build a minimal valid EDF file in memory."""
    record_duration = samples_per_record / sample_rate  # 1 second

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
    labels = [f"{'EDF ECG I':<16}", f"{'EDF ECG II':<16}"][:num_signals]
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
# Minimal SCP-ECG fixture
# ---------------------------------------------------------------------------

def create_scp_ecg_binary(
    num_leads: int = 2,
    samples_per_lead: int = 500,
    sample_rate: int = 500,
) -> bytes:
    """Build a minimal valid SCP-ECG binary file in memory."""
    # We'll build sections 0, 1, 3, 6

    # Section 1: Patient demographics
    sec1_data = bytearray()
    # Tag 0: Last name
    name = b"TestSCP\x00"
    sec1_data.append(0)
    sec1_data.extend(struct.pack("<H", len(name)))
    sec1_data.extend(name)
    # Tag 2: Patient ID
    pid = b"SCP001\x00"
    sec1_data.append(2)
    sec1_data.extend(struct.pack("<H", len(pid)))
    sec1_data.extend(pid)
    # Tag 8: Birth date (YYYY, MM, DD)
    sec1_data.append(8)
    sec1_data.extend(struct.pack("<H", 4))
    sec1_data.extend(struct.pack("<H", 1980))
    sec1_data.extend(bytes([6, 15]))
    # Tag 9: Sex (1=M)
    sec1_data.append(9)
    sec1_data.extend(struct.pack("<H", 1))
    sec1_data.append(1)
    # Tag 255: Terminator
    sec1_data.append(255)
    sec1_data.extend(struct.pack("<H", 0))

    # Section 3: Lead definition
    sec3_data = bytearray()
    sec3_data.append(num_leads)  # number of leads
    sec3_data.append(0)          # flags
    for i in range(num_leads):
        sec3_data.extend(struct.pack("<I", 1))                # start sample
        sec3_data.extend(struct.pack("<I", samples_per_lead)) # end sample
        sec3_data.append(i)                                    # lead ID

    # Section 6: Rhythm data (raw int16, no Huffman)
    sec6_data = bytearray()
    avm = 1000   # nV per unit
    sample_time = int(1_000_000 / sample_rate)  # µs
    sec6_data.extend(struct.pack("<H", avm))
    sec6_data.extend(struct.pack("<H", sample_time))
    sec6_data.append(0)   # encoding: first difference
    sec6_data.append(1)   # compression: none (non-bimodal = raw)

    # Per-lead byte lengths
    lead_bytes = samples_per_lead * 2
    for i in range(num_leads):
        sec6_data.extend(struct.pack("<H", lead_bytes))

    # Lead data as raw int16 (first-difference encoded)
    for lead_idx in range(num_leads):
        prev = 0
        for s in range(samples_per_lead):
            val = (lead_idx + 1) * 100 + (s % 50)
            diff = val - prev
            sec6_data.extend(struct.pack("<h", diff))
            prev = val

    def build_section(sec_id: int, data: bytes) -> bytes:
        """Build a section with 16-byte header."""
        sec = bytearray(16)
        sec[0] = sec_id
        sec[1:3] = struct.pack("<H", 0)  # CRC
        sec[3] = 0  # reserved
        struct.pack_into("<I", sec, 4, len(data) + 16)  # section length
        sec[8:10] = struct.pack("<H", 20)  # version
        sec[10:12] = struct.pack("<H", 20)  # protocol version
        sec[12:16] = b"\x00" * 4  # reserved
        return bytes(sec) + data

    sec1_bytes = build_section(1, bytes(sec1_data))
    sec3_bytes = build_section(3, bytes(sec3_data))
    sec6_bytes = build_section(6, bytes(sec6_data))

    # Section 0: Pointer table (after 6-byte file preamble)
    # We'll place sections sequentially:
    # [preamble 6] [sec0] [sec1] [sec3] [sec6]
    preamble_size = 6
    sec0_header_size = 16

    # Estimate sec0 size: header + entries (3 entries * 10 bytes each)
    num_entries = 3
    sec0_data_size = num_entries * 10
    sec0_total = sec0_header_size + sec0_data_size

    sec1_offset = preamble_size + sec0_total + 1  # 1-indexed
    sec3_offset = sec1_offset + len(sec1_bytes)
    sec6_offset = sec3_offset + len(sec3_bytes)

    sec0_data = bytearray()
    # Entry for section 1
    sec0_data.extend(struct.pack("<H", 1))
    sec0_data.extend(struct.pack("<I", len(sec1_bytes)))
    sec0_data.extend(struct.pack("<I", sec1_offset))
    # Entry for section 3
    sec0_data.extend(struct.pack("<H", 3))
    sec0_data.extend(struct.pack("<I", len(sec3_bytes)))
    sec0_data.extend(struct.pack("<I", sec3_offset))
    # Entry for section 6
    sec0_data.extend(struct.pack("<H", 6))
    sec0_data.extend(struct.pack("<I", len(sec6_bytes)))
    sec0_data.extend(struct.pack("<I", sec6_offset))

    sec0_bytes = build_section(0, bytes(sec0_data))

    file_size = preamble_size + len(sec0_bytes) + len(sec1_bytes) + len(sec3_bytes) + len(sec6_bytes)

    # File preamble: CRC (2) + file size (4)
    preamble = bytearray(6)
    struct.pack_into("<H", preamble, 0, 0)  # CRC placeholder
    struct.pack_into("<I", preamble, 2, file_size)

    return bytes(preamble) + sec0_bytes + sec1_bytes + sec3_bytes + sec6_bytes


@pytest.fixture
def scp_ecg_file(tmp_path: Path) -> Path:
    """Write a minimal SCP-ECG binary file and return its path."""
    p = tmp_path / "test.scp"
    p.write_bytes(create_scp_ecg_binary())
    return p


# ---------------------------------------------------------------------------
# Minimal GE MUSE XML fixture
# ---------------------------------------------------------------------------

GE_MUSE_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<RestingECG>
  <MuseInfo>
    <MuseVersion>9.0</MuseVersion>
  </MuseInfo>
  <PatientDemographics>
    <PatientID>MUSE001</PatientID>
    <PatientFirstName>Jane</PatientFirstName>
    <PatientLastName>Smith</PatientLastName>
    <Gender>FEMALE</Gender>
    <DateofBirth>03/15/1985</DateofBirth>
    <PatientAge>38</PatientAge>
  </PatientDemographics>
  <TestDemographics>
    <AcquisitionDate>12/01/2023</AcquisitionDate>
    <AcquisitionTime>14:30:00</AcquisitionTime>
    <AcquisitionDevice>MAC5500</AcquisitionDevice>
  </TestDemographics>
  <Waveform>
    <WaveformType>Rhythm</WaveformType>
    <SampleBase>500</SampleBase>
    <LeadData>
      <LeadID>I</LeadID>
      <WaveFormData>ZAAyAA==</WaveFormData>
    </LeadData>
    <LeadData>
      <LeadID>II</LeadID>
      <WaveFormData>ZAAyAA==</WaveFormData>
    </LeadData>
  </Waveform>
  <Diagnosis>
    <DiagnosisStatement>
      <StmtText>Normal sinus rhythm</StmtText>
    </DiagnosisStatement>
  </Diagnosis>
  <OriginalRestingECGMeasurements>
    <VentricularRate>72</VentricularRate>
    <QRSDuration>88</QRSDuration>
  </OriginalRestingECGMeasurements>
</RestingECG>
""")


@pytest.fixture
def ge_muse_xml_file(tmp_path: Path) -> Path:
    """Write a minimal GE MUSE XML file and return its path."""
    p = tmp_path / "test_muse.xml"
    p.write_text(GE_MUSE_XML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Minimal DICOM Waveform fixture
# ---------------------------------------------------------------------------

def create_dicom_binary() -> bytes:
    """Build a minimal DICOM file with waveform data.

    This creates the bare minimum structure pydicom can read.
    """
    # 128-byte preamble + DICM magic
    preamble = b"\x00" * 128 + b"DICM"
    # For actual testing with pydicom, we build a real DICOM dataset
    # Here we just return the magic; the test will use pydicom to create a proper file
    return preamble


@pytest.fixture
def dicom_file(tmp_path: Path) -> Path:
    """Write a minimal DICOM waveform file using pydicom and return its path."""
    pytest.importorskip("pydicom")
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian

    p = tmp_path / "test.dcm"

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(p), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PatientName = "Doe^John"
    ds.PatientID = "DCM001"
    ds.PatientSex = "M"
    ds.PatientBirthDate = "19900101"
    ds.StudyDate = "20231201"
    ds.StudyTime = "103000"
    ds.Modality = "ECG"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    ds.SOPInstanceUID = pydicom.uid.generate_uid()

    # Create waveform data — 2 channels, 100 samples each
    num_channels = 2
    num_samples = 100
    sample_rate = 500.0

    wf = Dataset()
    wf.NumberOfWaveformChannels = num_channels
    wf.NumberOfWaveformSamples = num_samples
    wf.SamplingFrequency = sample_rate
    wf.WaveformBitsAllocated = 16

    # Channel definitions
    ch1 = Dataset()
    ch1_source = Dataset()
    ch1_source.CodeMeaning = "I"
    ch1.ChannelSourceSequence = Sequence([ch1_source])
    ch1.ChannelSensitivity = "1.0"
    ch1.ChannelBaseline = "0"
    ch1.ChannelSensitivityCorrectionFactor = "1.0"

    ch2 = Dataset()
    ch2_source = Dataset()
    ch2_source.CodeMeaning = "II"
    ch2.ChannelSourceSequence = Sequence([ch2_source])
    ch2.ChannelSensitivity = "1.0"
    ch2.ChannelBaseline = "0"
    ch2.ChannelSensitivityCorrectionFactor = "1.0"

    wf.ChannelDefinitionSequence = Sequence([ch1, ch2])

    # Build multiplexed waveform data
    samples = np.zeros(num_samples * num_channels, dtype=np.int16)
    for s in range(num_samples):
        samples[s * num_channels + 0] = np.int16(100 + (s % 50))
        samples[s * num_channels + 1] = np.int16(200 + (s % 50))
    wf.WaveformData = samples.tobytes()

    ds.WaveformSequence = Sequence([wf])

    ds.save_as(str(p))
    return p


# ---------------------------------------------------------------------------
# Minimal WFDB fixture
# ---------------------------------------------------------------------------

def create_wfdb_files(directory: Path) -> Path:
    """Create minimal WFDB .hea + .dat files and return the .hea path."""
    num_signals = 2
    num_samples = 500
    sample_rate = 500
    gain = 200.0

    # Build .dat file: Format 16, interleaved int16
    dat = np.zeros(num_samples * num_signals, dtype="<i2")
    for s in range(num_samples):
        dat[s * num_signals + 0] = np.int16(100 + (s % 50))
        dat[s * num_signals + 1] = np.int16(200 + (s % 50))

    dat_path = directory / "test_wfdb.dat"
    dat_path.write_bytes(dat.tobytes())

    # Build .hea file
    hea_lines = [
        f"test_wfdb {num_signals} {sample_rate} {num_samples} 10:30:00 01/12/2023",
        f"test_wfdb.dat 16 {gain}(0)/mV 12 0 0 0 0 I",
        f"test_wfdb.dat 16 {gain}(0)/mV 12 0 0 0 0 II",
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
# Minimal MFER fixture
# ---------------------------------------------------------------------------

def create_mfer_binary(
    num_channels: int = 2,
    num_samples: int = 500,
    sample_rate: int = 500,
) -> bytes:
    """Build a minimal valid MFER binary file in memory."""
    data = bytearray()

    def add_tlv(tag: int, value: bytes) -> None:
        data.append(tag)
        length = len(value)
        if length < 0x80:
            data.append(length)
        else:
            # Long form BER length
            len_bytes = length.to_bytes((length.bit_length() + 7) // 8, "big")
            data.append(0x80 | len(len_bytes))
            data.extend(len_bytes)
        data.extend(value)

    # Byte order: big endian (0x00)
    add_tlv(0x01, b"\x00")

    # Data type: int16 (0x00)
    add_tlv(0x02, b"\x00")

    # Waveform type: ECG (0x00)
    add_tlv(0x03, b"\x00")

    # Sampling interval in µs
    interval_us = int(1_000_000 / sample_rate)
    add_tlv(0x04, struct.pack(">h", interval_us))

    # Number of channels
    add_tlv(0x05, struct.pack(">B", num_channels))

    # Number of samples per channel
    add_tlv(0x06, struct.pack(">H", num_samples))

    # Resolution
    add_tlv(0x0B, struct.pack(">h", 1))

    # Channel labels
    add_tlv(0x0A, b"I\x00")
    add_tlv(0x0A, b"II\x00")

    # Patient ID
    add_tlv(0x31, b"MFER001\x00")

    # Patient name
    add_tlv(0x32, b"MferTest\x00")

    # Sex: male (1)
    add_tlv(0x34, b"\x01")

    # Birth date: 1980-06-15
    add_tlv(0x33, struct.pack(">HBB", 1980, 6, 15))

    # Recording time: 2023-12-01 10:30:00
    add_tlv(0x41, struct.pack(">HBBBBB", 2023, 12, 1, 10, 30, 0))

    # Data block: interleaved int16 samples (big-endian)
    waveform = bytearray()
    for s in range(num_samples):
        for ch in range(num_channels):
            val = (ch + 1) * 100 + (s % 50)
            waveform.extend(struct.pack(">h", val))
    add_tlv(0x09, bytes(waveform))

    return bytes(data)


@pytest.fixture
def mfer_file(tmp_path: Path) -> Path:
    """Write a minimal MFER binary file and return its path."""
    p = tmp_path / "test.mwf"
    p.write_bytes(create_mfer_binary())
    return p


# ---------------------------------------------------------------------------
# Minimal BeneHeart R12 XML fixture
# ---------------------------------------------------------------------------

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
  </PatientInfo>
  <AcquisitionInfo>
    <AcquisitionDate>2023-12-01</AcquisitionDate>
    <AcquisitionTime>09:15:00</AcquisitionTime>
    <SampleRate>500</SampleRate>
    <Device>BeneHeart R12</Device>
  </AcquisitionInfo>
  <Leads>
    <Lead Name="I" Data="ZAAyAA=="/>
    <Lead Name="II" Data="ZAAyAA=="/>
  </Leads>
</BeneHeartR12>
""")


@pytest.fixture
def beneheart_r12_file(tmp_path: Path) -> Path:
    """Write a minimal BeneHeart R12 XML file and return its path."""
    p = tmp_path / "test_beneheart.xml"
    p.write_text(BENEHEART_R12_XML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Minimal GE MAC 2000 XML fixture
# ---------------------------------------------------------------------------

GE_MAC2000_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<MAC2000>
  <PatientDemographics>
    <PatientID>MAC001</PatientID>
    <PatientFirstName>Bob</PatientFirstName>
    <PatientLastName>Builder</PatientLastName>
    <Gender>MALE</Gender>
    <DateofBirth>07/22/1978</DateofBirth>
    <PatientAge>45</PatientAge>
  </PatientDemographics>
  <TestDemographics>
    <AcquisitionDate>12/01/2023</AcquisitionDate>
    <AcquisitionTime>11:00:00</AcquisitionTime>
    <SampleRate>500</SampleRate>
  </TestDemographics>
  <Waveform>
    <LeadData>
      <LeadID>I</LeadID>
      <WaveFormData>ZAAyAA==</WaveFormData>
    </LeadData>
    <LeadData>
      <LeadID>II</LeadID>
      <WaveFormData>ZAAyAA==</WaveFormData>
    </LeadData>
  </Waveform>
  <Measurements>
    <VentricularRate>68</VentricularRate>
    <QRSDuration>92</QRSDuration>
  </Measurements>
</MAC2000>
""")


@pytest.fixture
def ge_mac2000_file(tmp_path: Path) -> Path:
    """Write a minimal GE MAC 2000 XML file and return its path."""
    p = tmp_path / "test_mac2000.xml"
    p.write_text(GE_MAC2000_XML, encoding="utf-8")
    return p
