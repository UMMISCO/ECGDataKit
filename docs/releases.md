# Releases

All published releases of ECGDataKit, fetched from [GitHub Releases](https://github.com/UMMISCO/ECGDataKit/releases) at build time.

---

(v2-0-3)=
## v2.0.3 - XLE 12 leads and format detection

*Released 2026-10-14* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v2.0.3)

## ECGDataKit 2.0.3

- XLE (experimental): `XLEParser(wilson_correction=True, compute_missing_leads=True)` gives the standard 12 leads. III, aVR, aVL and aVF are computed from I and II, with a warning. The option requires the Wilson correction.
- New `FileParser.detect_format(path)`: tells the format of a file and the parser that reads it, without parsing it.
- `FileParser.parse(path)` no longer needs an instance. `FileParser().parse(path)` still works.

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v2.0.2...v2.0.3

(v2-0-2)=
## v2.0.2 - XLE voltage scale, chest leads and recording date

*Released 2026-10-14* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v2.0.2)

## ECGDataKit 2.0.2

Updates to the experimental XLE parser.

- Samples are now converted to mV with a scale of 0.9375 µV per count. The file does not store this scale and the manufacturer has not validated it, so the conversion issues a warning.
- Leads are named I, II and CR1 to CR6, since the chest leads are stored against the right arm. An optional Wilson correction, `XLEParser(wilson_correction=True)`, turns them into V1 to V6, with a warning.
- The recording date and time are now read from the file.

XLE support remains experimental. Check amplitudes against the recorder's own printout before use.

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v2.0.1...v2.0.2

(v2-0-1)=
## v2.0.1

*Released 2026-10-13* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v2.0.1)

## ECGDataKit 2.0.1

This release adds a new ECG format.

### Parsing

- **New format (experimental): XLE ECG (`.xle`).** 8-channel ECG recordings can now be read. The format is not documented, so its layout was worked out from sample files. The files do not state the voltage scale, so signals are kept in raw values and never converted with a guessed factor.

### Anonymization

- XLE files can now be anonymized like the other formats.

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v2.0.0...v2.0.1

(v2-0-0)=
## v2.0.0 - New anonymization module

*Released 2026-10-12* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v2.0.0)

## ECGDataKit 2.0.0

This release adds anonymization of ECG files, for every format ECGDataKit reads.

### Anonymization

- **New `ecgdatakit.anonymize` module with a command line tool (`ecgdatakit anonymize`).** Anonymizes ECG files by replacing the patient's identity (patient names, patient ID and ECG ID) with pseudonyms.
- **All supported formats.** Every format ECGDataKit reads can be anonymized, except EDAN Holter. Each anonymized file is checked to hold the same signal as the original.
- **Patients and folders.** Files can be grouped by patient folder, so all the recordings of a patient share one pseudonym. Patient folders can also be renamed.
- **Catalog and checks.** A CSV catalog links each original file to its anonymized copy and flags the files to review. 
- **Continuous mode.** The anonymizer can run in the background and handle new files as they arrive, also in a Docker container.

### Documentation

- The documentation has a new layout and new pages on anonymization.

Full documentation: https://ecgdatakit.ummisco.fr

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v1.1.1...v2.0.0

(v1-1-1)=
## v1.1.1

*Released 2026-10-09* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v1.1.1)

### New format: AliveCor Kardia JSON

ECG recordings exported from the Kardia API (KardiaMobile and KardiaMobile 6L) can now be read.

- The unfiltered signal goes to `leads`. The signal filtered by AliveCor goes to a new `leads_enhanced` field. Other formats leave this field empty.
- Samples are scaled with `amplitudeResolution` (nanovolts per count), as described in the Kardia API documentation.
- The AliveCor algorithm result (for example `afib`) is reported as a machine interpretation.
- Kardia 12L recordings use a different layout and are not supported.

ECGDataKit now reads 12 formats.

### Detected format and parser in `file_format`

`record.file_format` now holds the name of the detected format (`name`) and the parser that read the file (`parser`). They are filled in when the file is read through `FileParser` or `parse_batch`.

### Annotation changes

These changes rename or remove annotation keys. Code that reads them may need updating.

- **No more repeated measurements.** The Sierra and Mortara parsers stored some values both in `measurements` and in `annotations`. A value with a measurement field is now stored only in `measurements`. All other values stay in `annotations`.
- **One key per wave boundary.** HL7 aECG and SCP-ECG wrote all wave boundaries into one text annotation, `wave_boundaries_ms`. Each boundary now has its own key.
- **Representative beat boundaries start with `median_`** in every parser, for example `median_p_onset_ms` or `median_q_onset`. This covers HL7 aECG, SCP-ECG, Mortara, GE MUSE and Sierra.
- **Keys end with their unit when the format states it:** `_ms`, `_samples` or `_s`.
- **DICOM time-referenced annotations** moved from `raw_metadata["waveform_annotations"]` to `annotations`, with one key per wave type, for example `p_onset_samples`.

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v1.1.0...v1.1.1

(v1-1-0)=
## v1.1.0

*Released 2026-10-09* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v1.1.0)

### Breaking changes

- **Only stored values are reported**: intervals, heart rate, QTc, end dates and derived leads are no longer computed, and pacemaker status or severity are no longer inferred from text. Missing fields stay empty.
- **Stricter error handling**: unknown files raise `UnsupportedFormatError`, empty or damaged files raise `CorruptedFileError`, and a missing optional dependency raises `ImportError`.
- **Processing input validation**: NaN, infinite values and 2-D arrays raise `ValueError`. `clean_ecg` rejects unknown keyword arguments and propagates backend errors.
- **Normalized leads are unitless**: min-max and z-score outputs carry no physical unit and are not rescaled by `to_physical()`.
- **`Lead.to_physical()`** raises `RawSamplesError` when the file provides no physical unit.

### Parsing

- **Scaling contract**: `FileParser.parse()` returns leads in mV by default. `auto_scale=False` returns the stored values together with the scale needed to convert them.
- **Interpretation source**: `interpretation.source` distinguishes device statements (`"machine"`) from physician statements (`"confirmed"`, `"overread"`). When both are present, the device statements are kept in `annotations["machine_interpretation"]`.
- **SCP-ECG and MFER** rewritten according to EN 1064 and MFER Part 1.
- **WFDB, EDF, DICOM, HL7 aECG, GE MUSE, Philips Sierra, Mortara ELI and ISHNE**: numerous decoding and metadata fixes. Decoded signals match wfdb-python, pyedflib, pydicom, the FDA aECG reader and sierraecg on real recordings.
- **EDAN ARC Holter** marked experimental, as the format is undocumented.
- **New fields**: `ECGRecord.file_format`, `PatientInfo.has_pacemaker` and `PatientInfo.pacemaker_type`. New `ChecksumWarning` for checksum mismatches.

### Signal processing

- **R-peak detection**: Pan-Tompkins (default) no longer stops after artifacts, ignores flat segments and no longer counts T waves. MIT-BIH: 99.5 % sensitivity, 98.1 % positive predictive value at 250 to 1000 Hz.
- **Lead derivation**, resamplingw and physical leads correctly.
- **Signal quality and HRV** are consistent across sampling rates, and warn on recordings
too short for frequency-domain
- **DeepFADE** preserves input amplitude, refuses raw counts and is marked experimental.

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v1.0.2...v1.1.0

(v1-0-2)=
## v1.0.2

*Released 2026-07-06* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v1.0.2)

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v1.0.0...v1.0.2

(v1-0-0)=
## v1.0.0 - Stable Release

*Released 2026-03-19* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v1.0.0)

### New features

- **Per-lead normalization** — `normalize_minmax`, `normalize_zscore`, and `normalize_amplitude` now accept a `list[Lead]` and return a `list[Lead]`, normalizing each lead independently. Pass `record.leads` directly instead of looping manually
- Resampling and normalization (min-max, z-score, amplitude) now support batch processing across all leads (independently) of an ECG record


**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/compare/v0.0.9...v1.0.0

(v0-0-9)=
## v0.0.9 - ADC resolution pipeline

*Released 2026-03-07* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v0.0.9)

### Breaking changes

  - **`sample_rate` renamed to `sampling_rate`** in `Lead` and `SignalCharacteristics`
  - **`Lead.units` semantics changed** — now empty (`""`) when `is_raw=True` (raw ADC counts); set to the physical unit (voltages) only after `to_physical()` or when data is already in physical units
  - **`Lead.is_raw` is now auto-detected** — parsers no longer hardcode `is_raw=True`; instead `is_raw = not (resolution == 1.0 and offset == 0.0)`

  ### New features

  - **Automatic ADC-to-physical scaling** — `FileParser.parse(auto_scale=True)` (default) converts raw ADC samples to mV via `to_physical()` + `convert_units("mV")`. Disable with `auto_scale=False`
  - **`Lead.to_physical()`** — converts raw ADC samples using `physical = samples × resolution + offset`
  - **`Lead.convert_units(target)`** — converts between voltage units (uV, mV, V)
  - **`ECGRecord.to_physical()` / `ECGRecord.convert_units(target)`** — batch conversion for all leads and median beats
  - **`FileParser.supported_formats()`** — returns format metadata for all 12 parsers
  - **`ECGRecord.__repr__()` and `ECGRecord.plot()`** — YAML-style console display and quick plotting
  - **Multi-lead numpy array support** — `plot_leads`, `plot_12lead`, `iplot_leads`, `iplot_12lead` accept raw numpy arrays with `fs=` parameter
  - **`LeadsLike` type alias** — for multi-lead inputs (list of Lead, ECGRecord, 2D array, list of arrays)

  ### New Lead fields

  - **`resolution_unit`** — unit of the resolution scale factor (e.g. `"uV"`, `"mV"`); what samples will be in after `to_physical()`
  - **`adc_resolution`** — original ADC resolution as stored in the source file (e.g. `153.0` for 153 nV/count in ISHNE)
  - **`adc_resolution_unit`** — unit of `adc_resolution` as defined by the format (e.g. `"nV"` for ISHNE and SCP-ECG)

  ### Improvements

  - Simplified multi-lead plots: full signal by default with configurable `rows`/`cols` grid layout
  - `plot_12lead` and `iplot_12lead` assign standard 12-lead names to unnamed inputs
  - Static plots no longer force matplotlib Agg backend
  

**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/commits/v0.0.9

(v0-0-8)=
## v0.0.8 - Visualization

*Released 2026-03-03* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v0.0.8)

### Visualization

- Multi-lead plotting functions (`plot_leads`, `plot_12lead`, `plot_quality`, `iplot_leads`, `iplot_12lead`) now accept raw numpy arrays directly with `fs=` parameter — pass a 2D array (n_leads × n_samples) or a list of 1D arrays alongside the sample rate
- New `LeadsLike` type alias for multi-lead inputs: `list[Lead] | ECGRecord | NDArray | list[NDArray]`
- Static plots no longer force the `Agg` backend — plots display inline in Jupyter and GUI environments by default


**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/commits/v0.0.8

(v0-0-0-7)=
## v0.0.7 - Signal Characteristics, Flexible Inputs & Sphinx Docs

*Released 2026-03-02* &middot; [View on GitHub](https://github.com/UMMISCO/ECGDataKit/releases/tag/v0.0.0.7)

## What's New

  ### Signal Characteristics Model

  New `SignalCharacteristics` dataclass on every `ECGRecord` — captures technical signal encoding metadata directly from each format:

  - `bits_per_sample`, `signal_offset`, `signal_signed`
  - `data_encoding` (e.g. `base64_int16le`, `format_212`, `int16`)
  - `compression` (e.g. `none`, `huffman`)
  - `number_channels_allocated` / `number_channels_valid`
  - `electrode_placement`, `acsetting`, and signal processing flags

  All 12 parsers now populate `record.signal` from format-specific fields.

  ### Expanded Metadata Extraction

  Parsers now extract significantly more metadata from each format:

  - **Technician** — Sierra XML, GE MUSE, DICOM, GE MAC 2000
  - **Referring physician** — Sierra XML, EDF, GE MUSE, GE MAC 2000
  - **Patient weight** — SCP-ECG (Tag 12), BeneHeart R12, GE MAC 2000
  - **Patient height** — BeneHeart R12, GE MAC 2000
  - **Patient race** — HL7 aECG, Mortara EL250, GE MAC 2000
  - **RR interval & QRS count** — Mortara EL250, BeneHeart R12, GE MAC 2000
  - **Device serial number** — HL7 aECG, GE MAC 2000
  - **Notch filter active flag** — now properly set across all parsers

  ### Flexible Inputs — Numpy Arrays Everywhere

  All processing and plotting functions now accept **raw numpy arrays** directly alongside `Lead` objects. Pass the sample rate via `fs=`:

  ```python
  import numpy as np
  from ecgdatakit.processing import diagnostic_filter, detect_r_peaks
  from ecgdatakit.plotting import plot_lead

  signal = np.random.randn(5000)
  filtered = diagnostic_filter(signal, fs=500)
  peaks = detect_r_peaks(filtered)
  plot_lead(filtered, peaks=peaks)
```

  ### Plotting Improvements

  - Auto-display: static plots now call plt.show() by default — pass show=False to get the figure for saving
  - Sample index axis: plot_lead(lead, x_axis="samples") for sample indices instead of time
  - Cleaner layout: reduced margins, integer time ticks on x-axis

  ### Documentation Migration

  - Migrated from Hugo to Sphinx with the Furo theme
  - Full autodoc API reference generated from source docstrings
  - Hosted at https://ecgdatakit.ummisco.fr

  Bug Fixes

  - BeneHeart R12: fixed assignment to non-existent RecordingInfo.device field
  - SCP-ECG: now reads patient weight from Tag 12 (was silently skipped)
  - MFER: notch filter now correctly extracted from filter tag when 12+ bytes present


**Full Changelog**: https://github.com/UMMISCO/ECGDataKit/commits/v0.0.0.7
