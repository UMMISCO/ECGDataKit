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
| EDAN ARC Holter (experimental) | recording folder, `patient.hea` + `ecgraw.dat`, `.arc` | `EDANARCHolterParser` | `patient.hea` with sibling `ecgraw.dat`, or an `.arc` with a known signature |
| AliveCor Kardia JSON | `.json` | `AliveCorKardiaParser` | JSON object with Kardia recording keys (`recordedAt`, `algorithmDetermination`, ...) |
| XLE ECG (experimental) | `.xle` | `XLEParser` | `.xle` file with a 256 KiB header holding the patient field labels, and whole 16-byte frames after it |

## HL7 aECG

The HL7 annotated ECG standard (**ANSI/HL7 V3 aECG**) is an XML-based format widely used for regulatory ECG submissions to the FDA. Files contain waveform data, patient demographics, annotations, and measurements in a structured XML hierarchy.

**Dependencies:** No optional dependency (XML is read with the core dependency `defusedxml`).

## Philips Sierra XML

The proprietary XML format produced by Philips/Sierra ECG management systems. Found on devices like the PageWriter TC series and IntelliSpace ECG. Contains lead data, patient info, and interpretation statements.

**Dependencies:** No optional dependency (XML is read with the core dependencies `defusedxml` and `xmltodict`).

## GE MUSE XML

XML export format from GE Healthcare's MUSE ECG management system. Contains base64-encoded waveform data, patient demographics, measurements, and interpretation text. Common in hospital GE environments.

**Dependencies:** No optional dependency (XML is read with the core dependency `defusedxml`).

## ISHNE Holter

A binary format defined by the International Society for Holter and Noninvasive Electrocardiology. Designed for long-duration Holter recordings. Contains a fixed-length header with patient info and signal metadata, followed by raw sample data.

**Dependencies:** None (the CRC-16 check uses the standard library). Some recorders store a constant instead of a real checksum: the mismatch is reported as `ChecksumWarning` and `raw_metadata["checksum_valid"]`.

## Mortara ELI (EL250/ELI280)

XML format produced by Mortara/Welch Allyn ELI series ECG devices (ELI 250, ELI 280). Stores multi-channel waveform data with per-channel metadata in a straightforward XML structure.

**Dependencies:** No optional dependency (XML is read with the core dependency `defusedxml`).

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

**Dependencies:** None. Requires both `.hea` and `.dat` files in the same directory. FLAC storage formats 508/516/524 (used by MIMIC-IV waveforms) are not supported and raise `UnsupportedFormatError`. When a signal gives no unit, the WFDB default mV is used and flagged with a warning.

## MFER

The **Medical waveform Format Encoding Rules** (ISO 22077) is a lightweight binary format popular in Japan and Asia. Uses a tag-length-value (BER) encoding with compact representation. Supports multiple channels and various data types. Only the first frame's channel layout is joined into leads: later frames that redefine channels, sampling or scaling are reported in `raw_metadata["frames"]` with a warning.

**Dependencies:** None.

## EDAN ARC Holter (experimental)

```{warning}
EDAN support is experimental. EDAN does not publish this format: the parser follows a partial reverse-engineered description and has not been checked on a wide range of recorders. Every parse issues a `UserWarning`. Check sample values and metadata against the recorder's own export before using the data.
```

Holter recordings from EDAN devices, either as a recording folder (`patient.hea` with a sibling `ecgraw.dat`) or as a single `.arc` archive. The format is not publicly documented and the file does not state the count to voltage factor, so samples stay in raw counts (`is_raw=True`, no unit). "NEUTRAL" `.arc` archives are reverse engineered (`raw_metadata["reverse_engineered"]` is `True`) and their 250 Hz rate is not read from the file (`raw_metadata["sampling_rate_stated"]` is `False`).

**Dependencies:** None.

## AliveCor Kardia JSON

ECG recordings from KardiaMobile (single lead) and KardiaMobile 6L devices, as returned by the [Kardia API](https://developers.kardia.com) (`GET /v1/recordings/:id`). The file holds the signal twice: `raw` goes to `record.leads` and `enhanced` (filtered by AliveCor) to `record.leads_enhanced`. Samples are scaled with `amplitudeResolution` (nanovolts per count). `algorithmDetermination` is the device's automatic analysis (`interpretation.source` is `"machine"`). Kardia 12L recordings use another layout and raise `UnsupportedFormatError`.

**Dependencies:** None.

## XLE ECG (experimental)

```{warning}
XLE support is experimental. There is no public description of this format, and the device that writes it is not known. The layout was worked out from sample files and has not been checked against a specification or the recorder's own software. Every parse issues a `UserWarning`.
```

8-channel ECG recordings in `.xle` files: a 256 KiB header, then the signal up to the end of the file. What the parser reads:

- **Signal.** 8 channels of signed 16-bit samples. Only the low 16 bits of each value are stored, so a baseline that drifts past the 16-bit range wraps around. The parser undoes these wraps to give a continuous signal (`raw_metadata["wraps_undone"]` counts them).
- **Patient fields.** Given name, family name, ID, sex and history go to `record.patient`. The birth date is free text in the file, kept as stored in `raw_metadata["birth_date_text"]` and not converted to a date. The comment goes to `raw_metadata["comment"]`.

What the file does not store:

- **The voltage scale.** Samples stay in raw ADC counts (`is_raw=True`, no unit), also with `auto_scale=True`. They are never converted to millivolts.
- **The lead labels.** Channels are named `Ch1` to `Ch8`. In the sample files their waveforms match I, II, V1 to V6 in that order. This order is given in `raw_metadata["likely_lead_order"]` but not applied.
- **The sampling rate.** It is set to 1000 Hz, the rate that matches the sample files (heart cycle intervals, mains interference at 50 Hz), and `raw_metadata["sampling_rate_stated"]` is `False`.
- **The recording date.**

**Dependencies:** None.
