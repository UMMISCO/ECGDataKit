# Supported Formats

ECGDataKit parses 13 ECG file formats via content-based detection, not file extensions. Parsers with a reliable magic number are tried first and loose sniffers last (`Parser.PRIORITY`).

| Format | File Types | Parser Class | Detection |
|--------|-----------|--------------|-----------|
| HL7 aECG | `.xml` | `HL7aECGParser` | `<AnnotatedECG` in header |
| Philips Sierra XML | `.xml` | `SierraXMLParser` | `<restingecgdata` in header |
| GE MUSE XML | `.xml` | `GEMuseXMLParser` | `<RestingECG>` in header |
| ISHNE Holter | `.ecg` | `ISHNEHolterParser` | `ISHNE1.0` magic bytes |
| Mortara ELI (EL250/ELI280) | `.xml` | `MortaraEL250Parser` | `<ECG` root + ELI elements or attributes |
| EDF / EDF+ | `.edf` | `EDFParser` | `"0       "` at offset 0 + valid structure |
| SCP-ECG | `.scp` | `SCPECGParser` | Section 0 header at offset 6 (ID at byte 8) and a valid pointer table |
| DICOM Waveform | `.dcm` | `DICOMWaveformParser` | `DICM` at offset 128 |
| WFDB (PhysioNet) | `.hea` + `.dat` | `WFDBParser` | valid `.hea` header, or a `.dat` with a sibling `.hea` |
| MFER | `.mwf`, `.mfer`, `.mfr` | `MFERParser` | `MFR ` preamble or a chain of known MFER tags |
| Mindray BeneHeart R12 | `.xml` | `BeneHeartR12Parser` | `<BeneHeartR12>` or `<MindrayECG>` |
| GE MAC 2000 | `.xml` | `GEMAC2000Parser` | `<MAC2000>` or `<GE_MAC>` |
| EDAN ARC Holter | recording folder, `patient.hea` + `ecgraw.dat`, `.arc` | `EDANARCHolterParser` | `patient.hea` with sibling `ecgraw.dat`, or an `.arc` with a known signature |

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

**Dependencies:** None (the CRC-16 check uses the standard library). Some recorders store a constant instead of a real checksum: the mismatch is reported as `ChecksumWarning` and `raw_metadata["checksum_valid"]`.

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
