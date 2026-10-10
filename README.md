# ECGDataKit

[![PyPI](https://img.shields.io/pypi/v/ecgdatakit)](https://pypi.org/project/ecgdatakit/)
[![Release](https://img.shields.io/github/v/release/UMMISCO/ECGDataKit)](https://github.com/UMMISCO/ECGDataKit/releases)
[![Tests](https://github.com/UMMISCO/ECGDataKit/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/UMMISCO/ECGDataKit/actions/workflows/tests.yml)
[![Docs](https://github.com/UMMISCO/ECGDataKit/actions/workflows/docs.yml/badge.svg?branch=main)](https://ecgdatakit.ummisco.fr)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/UMMISCO/ECGDataKit/blob/main/LICENSE)
[![GitLab](https://img.shields.io/badge/GitLab-mirror-orange?logo=gitlab)](https://git.ummisco.fr/open/ecgdatakit)

**A Python library for parsing, processing, and visualizing multi-format ECG files.**

Developed at [UMMISCO](https://www.ummisco.fr) / [IRD](https://www.ird.fr) by Ahmad Fall.

> **[ecgdatakit.ummisco.fr](https://ecgdatakit.ummisco.fr)**: Full documentation, API reference, and getting started guide.

---

## Features

### Parsing - 12 ECG formats, one unified data model

Formats are detected from the file content, not the extension.

| Format | File Types | Detection |
|--------|-----------|-----------|
| HL7 aECG | `.xml` | `<AnnotatedECG` in header |
| Philips Sierra XML | `.xml` | `<restingecgdata` in header |
| GE MUSE XML | `.xml` | `<RestingECG>` in header |
| ISHNE Holter | `.ecg` | `ISHNE1.0` magic bytes |
| Mortara ELI (EL250/ELI280) | `.xml` | `<ECG` root + ELI elements or attributes |
| EDF/EDF+ | `.edf` | `"0       "` at offset 0 + valid structure |
| SCP-ECG | `.scp` | Section 0 header at offset 6 and a valid pointer table |
| DICOM Waveform | `.dcm` | `DICM` at offset 128 |
| WFDB (PhysioNet) | `.hea` + `.dat` | valid `.hea` header, or a `.dat` with a sibling `.hea` |
| MFER | `.mwf`, `.mfer`, `.mfr` | `MFR ` preamble or a chain of known MFER tags |
| EDAN ARC Holter (experimental) | `patient.hea` + `ecgraw.dat`, `.arc` | `patient.hea` with sibling `ecgraw.dat`, or an `.arc` with a known signature |
| AliveCor Kardia JSON | `.json` | JSON object with Kardia recording keys (`recordedAt`, `algorithmDetermination`, ...) |

Parsers report what the file stores. They never compute clinical values (intervals, heart rate, QTc, axes, pacemaker status) or invent dates. The only derived values are those a format requires (duration from the sample count, age from the birth date when no age is stored), and a default taken from a format specification is flagged with a warning and in `raw_metadata`.

### Signal Processing

| Category | Capabilities |
|----------|-------------|
| **Filtering** | Butterworth (lowpass, highpass, bandpass, notch), baseline removal, diagnostic & monitoring presets |
| **Peak Detection** | Pan-Tompkins, Shannon energy |
| **Heart Rate** | Average HR, RR intervals, instantaneous beat-by-beat HR |
| **HRV Analysis** | Time-domain (SDNN, RMSSD, pNN50), frequency-domain (VLF/LF/HF), Poincaré (SD1/SD2) |
| **Spectral** | FFT, Welch PSD, beat segmentation, ensemble averaging |
| **Quality** | Signal quality index (SQI), SNR estimation |
| **Leads** | Derive III, aVR/aVL/aVF, full 12-lead assembly |
| **Cleaning** | Built-in, BioSPPy, NeuroKit2, combined pipelines |
| **Deep Denoising** | DeepFADE (experimental), a DenseNet encoder-decoder denoising autoencoder trained on a large private ECG database (weights bundled). Not validated for clinical use |

### Visualization

| Type | Plots |
|------|-------|
| **ECG Waveforms** | Single lead, multi-lead, 12-lead, optional ECG paper grid (0.2 s / 0.5 mV squares) |
| **Annotations** | R-peak markers, RR intervals, heart rate overlay |
| **Beat Analysis** | Segmented beats, ensemble-averaged beat with SD shading |
| **Spectral** | Power spectrum (PSD/FFT), spectrogram |
| **HRV** | Tachogram, Poincaré plot, frequency bands, metrics dashboard |
| **Reports** | Signal quality per lead, ECG report page with patient info and the standard 3 x 4 layout (2.5 s per column) plus a 10 s rhythm strip |
| **Interactive** | Plotly versions (zoom, pan, hover) of single lead, multi-lead, 12-lead, R-peaks, spectrum, RR tachogram, Poincaré plot and report. Beat, spectrogram, HRV summary and quality plots are static only |

### Anonymization

`ecgdatakit anonymize` copies the ECG files of a source folder into an `ANONYMIZED` folder with the patient's name, the ECG ID and, when it can be trusted, the patient ID replaced by pseudonyms, and keeps the link in a CSV catalog. Raw files are only read. Each copy is read back and must give the same samples and fields before it is published. It runs once or as a daemon that follows new files. See [Anonymization](https://ecgdatakit.ummisco.fr/api/anonymization.html) in the docs.

```bash
ecgdatakit anonymize run /path/to/source
```

## Installation

```bash
# Core (parsing only)
pip install ecgdatakit

# With signal processing
pip install "ecgdatakit[processing]"

# With static plots (matplotlib)
pip install "ecgdatakit[plotting]"

# With interactive plots (plotly)
pip install "ecgdatakit[plotting-interactive]"

# With ECG cleaning backends
pip install "ecgdatakit[cleaning]"

# With DICOM waveform support (pydicom)
pip install "ecgdatakit[dicom]"

# With anonymization (ecgdatakit anonymize)
pip install "ecgdatakit[anonymize]"

# With the experimental DeepFADE denoising autoencoder (requires torch)
pip install "ecgdatakit[denoising]"

# Everything except torch (install it separately if needed)
pip install "ecgdatakit[all]"

# Development version from GitLab
pip install "ecgdatakit @ git+https://git.ummisco.fr/open/ecgdatakit.git#subdirectory=lib"
```

## Quick Start

### Parse an ECG file

```python
from ecgdatakit import FileParser

record = FileParser().parse("path/to/ecg_file.xml")   # leads in mV by default

print(record.source_format)            # "sierra_xml"
print(record.patient.first_name)       # "John"
print(record.patient.age)              # 55
print(record.recording.acquisition.signal.sampling_rate)  # 500
print(record.measurements.heart_rate)  # 75
print(record.recording.device.manufacturer)               # "Philips"
print(record.recording.acquisition.signal.data_encoding)  # "Base64"
print(len(record.leads))               # 12

json_str = record.to_json()
```

### Visualize

```python
from ecgdatakit.plotting import plot_12lead, plot_lead, plot_report, iplot_12lead

plot_12lead(record)                # static, one row per lead (auto-displays)
plot_lead(record.leads[0], show_grid=True)   # single lead on ECG paper grid
plot_report(record)                # report page, standard 3 x 4 layout

fig = plot_12lead(record, show=False)   # get the figure without displaying
fig.savefig("ecg_12lead.png", dpi=150)

iplot_12lead(record)               # interactive (plotly), opens in browser
```

### Batch processing

`parse_batch` uses worker processes, so scripts need the `__main__` guard (required on macOS and Windows).

```python
from pathlib import Path
from ecgdatakit import parse_batch

if __name__ == "__main__":
    files = list(Path("ecg_data/").glob("*.xml"))
    # on_error="warn" skips unreadable files with a BatchParseWarning
    for record in parse_batch(files, max_workers=4, on_error="warn"):
        print(record.raw_metadata["filepath"], record.measurements.heart_rate)
```

## Data Model

Every parser returns the same `ECGRecord`, so downstream code stays identical no matter which format was read. `FileParser().parse(path)` returns leads in mV by default (`auto_scale=True`; pass `units="uV"` or `"V"` for another unit). Leads whose file gives no voltage scale stay in raw counts, with a warning. With `auto_scale=False`, `lead.samples` are the values stored in the file and `physical = samples * resolution + offset` in `resolution_unit`; `record.to_physical()` and `record.convert_units("mV")` do that conversion. Export the whole record with `record.to_dict()` or `record.to_json()`.

```
ECGRecord
  patient          PatientInfo             ID, name, birth date, sex, race, age, weight, height, medications, history, pacemaker
  recording        RecordingInfo           date, end date, duration, technician, physician, room, location
    ├─ device      DeviceInfo              manufacturer, model, serial number, software version, institution
    └─ acquisition AcquisitionSetup
         ├─ signal SignalCharacteristics   sampling rate, resolution, bits/sample, encoding, compression, channels
         └─ filters FilterSettings         highpass, lowpass, notch frequencies
  leads            list[Lead]              label, samples (float64), sampling rate, resolution, units, is_raw
  measurements     GlobalMeasurements      HR, PR, QRS, QT, QTc (Bazett/Fridericia), P/QRS/T axes, RR interval
  interpretation   Interpretation          statements, severity, source, interpreter
  median_beats     list[Lead]              median/template beats, when available
  leads_enhanced   list[Lead]              the same leads filtered by the device, when the file stores them (AliveCor)
  annotations      dict[str, str]          additional key-value annotations
  source_format    str                     parser identifier (e.g. "sierra_xml")
  file_format      FileFormatInfo          file format version and creation date
  raw_metadata     dict                    original format-specific metadata
```

## Author

**Ahmad Fall**, [UMMISCO](https://www.ummisco.fr) / [IRD](https://www.ird.fr)

## License

Apache 2.0. See [LICENSE](https://github.com/UMMISCO/ECGDataKit/blob/main/LICENSE) for details.
