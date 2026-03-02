---
title: "Supported Formats"
weight: 20
---

ECGDataKit parses 12 ECG file formats via content-based detection — no file extension guessing.

<table>
  <thead>
    <tr><th>Format</th><th>File Types</th><th>Parser Class</th><th>Detection</th></tr>
  </thead>
  <tbody>
    <tr><td>HL7 aECG</td><td><code>.xml</code></td><td><code>HL7aECGParser</code></td><td><code>&lt;AnnotatedECG</code> in header</td></tr>
    <tr><td>Philips Sierra XML</td><td><code>.xml</code></td><td><code>SierraXMLParser</code></td><td><code>&lt;restingecgdata</code> in header</td></tr>
    <tr><td>GE MUSE XML</td><td><code>.xml</code></td><td><code>GEMuseXMLParser</code></td><td><code>&lt;RestingECG&gt;</code> in header</td></tr>
    <tr><td>ISHNE Holter</td><td><code>.ecg</code>, <code>.hol</code></td><td><code>ISHNEHolterParser</code></td><td><code>ISHNE1.0</code> or <code>ANN  1.0</code> magic bytes</td></tr>
    <tr><td>Mortara EL250</td><td><code>.xml</code></td><td><code>MortaraEL250Parser</code></td><td><code>&lt;ECG</code> + <code>&lt;CHANNEL</code> in header</td></tr>
    <tr><td>EDF / EDF+</td><td><code>.edf</code></td><td><code>EDFParser</code></td><td><code>"0       "</code> at offset 0 + valid structure</td></tr>
    <tr><td>SCP-ECG</td><td><code>.scp</code></td><td><code>SCPECGParser</code></td><td>Valid Section 0 pointer table at offset 6</td></tr>
    <tr><td>DICOM Waveform</td><td><code>.dcm</code></td><td><code>DICOMWaveformParser</code></td><td><code>DICM</code> at offset 128</td></tr>
    <tr><td>WFDB (PhysioNet)</td><td><code>.hea</code> + <code>.dat</code></td><td><code>WFDBParser</code></td><td><code>.hea</code> extension + valid header</td></tr>
    <tr><td>MFER</td><td><code>.mwf</code>, <code>.mfer</code></td><td><code>MFERParser</code></td><td>Valid MFER tag (0x01–0x3F) + BER length</td></tr>
    <tr><td>Mindray BeneHeart R12</td><td><code>.xml</code></td><td><code>BeneHeartR12Parser</code></td><td><code>&lt;BeneHeartR12&gt;</code> or <code>&lt;MindrayECG&gt;</code></td></tr>
    <tr><td>GE MAC 2000</td><td><code>.xml</code></td><td><code>GEMAC2000Parser</code></td><td><code>&lt;MAC2000&gt;</code> or <code>&lt;GE_MAC&gt;</code></td></tr>
  </tbody>
</table>

## HL7 aECG

The HL7 annotated ECG standard (**ANSI/HL7 V3 aECG**) is an XML-based format widely used for regulatory ECG submissions to the FDA. Files contain waveform data, patient demographics, annotations, and measurements in a structured XML hierarchy.

**Dependencies:** None (uses built-in XML parsing).

## Philips Sierra XML

The proprietary XML format produced by Philips/Sierra ECG management systems. Found on devices like the PageWriter TC series and IntelliSpace ECG. Contains lead data, patient info, and interpretation statements.

**Dependencies:** None.

## GE MUSE XML

XML export format from GE Healthcare's MUSE ECG management system. Contains base64-encoded waveform data, patient demographics, measurements, and interpretation text. Common in hospital GE environments.

**Dependencies:** None.

## ISHNE Holter

A binary format defined by the International Society for Holter and Noninvasive Electrocardiology. Designed for long-duration Holter recordings. Contains a fixed-length header with patient info and signal metadata, followed by raw sample data.

**Dependencies:** Optional `PyCRC` for CRC-16 checksum validation.

```bash
pip install "ecgdatakit[holter]"
```

## Mortara EL250

XML format produced by Mortara/Welch Allyn ELI 250 series ECG devices. Stores multi-channel waveform data with per-channel metadata in a straightforward XML structure.

**Dependencies:** None.

## EDF / EDF+

The **European Data Format** is an open standard for multi-channel biosignal data exchange. EDF+ extends the original EDF with annotations and discontinuous recordings. Used across polysomnography, EEG, and ECG.

**Dependencies:** None (pure binary parsing).

## SCP-ECG

The **Standard Communications Protocol for computer-assisted Electrocardiography** (EN 1064 / ISO 11073-91064). A compact binary format with section-based structure storing patient data, Huffman-compressed waveforms, measurements, and interpretation. Supports both reference-beat and full-disclosure storage.

**Dependencies:** None.

## DICOM Waveform

DICOM (Digital Imaging and Communications in Medicine) can store ECG waveforms in the Waveform Information Object Definition. Contains structured patient, study, and series data alongside encoded signal channels.

**Dependencies:** Requires `pydicom` for DICOM file reading.

```bash
pip install "ecgdatakit[dicom]"
```

## WFDB (PhysioNet)

The format used by **PhysioNet** and the WFDB (WaveForm DataBase) toolset. Consists of a header file (`.hea`) describing the signal layout and one or more binary data files (`.dat`) containing raw samples. Commonly used in research datasets like MIT-BIH, PTB-XL, and MIMIC.

**Dependencies:** None. Requires both `.hea` and `.dat` files in the same directory.

## MFER

The **Medical waveform Format Encoding Rules** (ISO 22077) is a lightweight binary format popular in Japan and Asia. Uses a tag-length-value (BER) encoding with compact representation. Supports multiple channels and various data types.

**Dependencies:** None.

## Mindray BeneHeart R12

XML format exported by Mindray BeneHeart R12 and similar Mindray ECG devices. Contains patient data, waveforms, and interpretation in a device-specific XML schema.

**Dependencies:** None.

## GE MAC 2000

XML format from GE Healthcare MAC 2000 resting ECG devices. Contains patient demographics, waveform data, and device-generated measurements and interpretation.

**Dependencies:** None.
