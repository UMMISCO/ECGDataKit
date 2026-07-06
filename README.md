# ECGDataKit

[![PyPI](https://img.shields.io/pypi/v/ecgdatakit)](https://pypi.org/project/ecgdatakit/)
[![Release](https://img.shields.io/github/v/release/UMMISCO/ECGDataKit)](https://github.com/UMMISCO/ECGDataKit/releases)
[![Tests](https://github.com/UMMISCO/ECGDataKit/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/UMMISCO/ECGDataKit/actions/workflows/tests.yml)
[![Docs](https://github.com/UMMISCO/ECGDataKit/actions/workflows/docs.yml/badge.svg?branch=main)](https://ecgdatakit.ummisco.fr)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![GitLab](https://img.shields.io/badge/GitLab-mirror-orange?logo=gitlab)](https://git.ummisco.fr/open/ecgdatakit)

**A Python library for parsing, processing, and visualizing multi-format ECG files.**

Developed at [UMMISCO](https://www.ummisco.fr) / [IRD](https://www.ird.fr) by Ahmad Fall.

> **[ecgdatakit.ummisco.fr](https://ecgdatakit.ummisco.fr)**: Full documentation, API reference, and getting started guide.

---

## Features

### Parsing - 13 ECG formats, one unified data model

| Format | File Types | Detection |
|--------|-----------|-----------|
| HL7 aECG | `.xml` | `<AnnotatedECG` in header |
| Philips Sierra XML | `.xml` | `<restingecgdata` in header |
| ISHNE Holter | `.ecg`, `.hol` | `ISHNE1.0` or `ANN  1.0` magic bytes |
| Mortara EL250 | `.xml` | `<ECG` + `<CHANNEL` in header |
| EDF/EDF+ | `.edf` | `"0       "` at offset 0 |
| SCP-ECG | `.scp` | Valid Section 0 pointer table at offset 6 |
| GE MUSE XML | `.xml` | `<RestingECG>` in header |
| DICOM Waveform | `.dcm` | `DICM` at offset 128 |
| WFDB (PhysioNet) | `.hea` + `.dat` | `.hea` extension + valid header |
| MFER | `.mwf`, `.mfer` | Valid MFER tag + BER length |
| Mindray BeneHeart R12 | `.xml` | `<BeneHeartR12>` or `<MindrayECG>` |
| GE MAC 2000 | `.xml` | `<MAC2000>` or `<GE_MAC>` |
| EDAN ARC Holter | `patient.hea` + `ecgraw.dat` | filename `patient.hea` + sibling `ecgraw.dat` |

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
| **Deep Denoising** | DeepFADE, a DenseNet encoder-decoder denoising autoencoder trained on a large private ECG database (weights bundled) |

### Visualization

| Type | Plots |
|------|-------|
| **ECG Waveforms** | Single lead, multi-lead, standard 12-lead grid with paper background |
| **Annotations** | R-peak markers, RR intervals, heart rate overlay |
| **Beat Analysis** | Segmented beats, ensemble-averaged beat with SD shading |
| **Spectral** | Power spectrum (PSD/FFT), spectrogram |
| **HRV** | Tachogram, Poincaré plot, frequency bands, metrics dashboard |
| **Reports** | Signal quality per lead, full ECG report with patient info |
| **Interactive** | All plots available as interactive Plotly versions (zoom, pan, hover) |

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

# With DeepFADE denoising autoencoder (requires torch)
pip install "ecgdatakit[denoising]"

# Everything (except torch, install separately if needed)
pip install "ecgdatakit[all]"
```

Optional extras for specific formats:

```bash
pip install "ecgdatakit[holter]"   # ISHNE Holter CRC validation
pip install "ecgdatakit[dicom]"    # DICOM waveform support
```

## Quick Start

### Parse an ECG file

```python
from ecgdatakit import FileParser

record = FileParser().parse("path/to/ecg_file.xml")

print(record.source_format)            # "sierra_xml"
print(record.patient.first_name)       # "John"
print(record.patient.age)              # 55
print(record.recording.acquisition.signal.sampling_rate)  # 500
print(record.measurements.heart_rate)  # 75
print(record.recording.device.manufacturer)               # "Philips"
print(record.recording.acquisition.signal.data_encoding)  # "base64_int16le"
print(len(record.leads))               # 12

json_str = record.to_json()
```

### Visualize

```python
from ecgdatakit.plotting import plot_12lead, plot_lead, iplot_12lead

plot_12lead(record)                # static 12-lead grid (auto-displays)
plot_lead(record.leads[0])         # single lead

fig = plot_12lead(record, show=False)   # get the figure without displaying
fig.savefig("ecg_12lead.png", dpi=150)

iplot_12lead(record).show()        # interactive (plotly), opens in browser
```

### Batch processing

```python
from pathlib import Path
from ecgdatakit import parse_batch

files = list(Path("ecg_data/").glob("*.xml"))
for record in parse_batch(files, max_workers=4):
    print(record.patient.patient_id, record.measurements.heart_rate)
```

## Data Model

Every parser returns the same `ECGRecord`, so downstream code stays identical no matter which format was read. Leads hold raw ADC counts by default. Call `record.to_physical()` to scale them to voltage, then `record.convert_units("mV")` to switch between `uV`, `mV`, or `V`. Export the whole record with `record.to_dict()` or `record.to_json()`.

```
ECGRecord
  patient          PatientInfo             ID, name, birth date, sex, age, weight, height, medications, history
  recording        RecordingInfo           date, end date, duration, technician, physician, room, location
    ├─ device      DeviceInfo              manufacturer, model, serial number, software version, institution
    └─ acquisition AcquisitionSetup
         ├─ signal SignalCharacteristics   sampling rate, resolution, bits/sample, encoding, compression, channels
         └─ filters FilterSettings         highpass, lowpass, notch frequencies
  leads            list[Lead]              label, samples (float64), sampling rate, resolution, units, is_raw
  measurements     GlobalMeasurements      HR, PR, QRS, QT, QTc (Bazett/Fridericia), P/QRS/T axes, RR interval
  interpretation   Interpretation          statements, severity, source, interpreter
  median_beats     list[Lead]              median/template beats, when available
  annotations      dict[str, str]          additional key-value annotations
  source_format    str                     parser identifier (e.g. "sierra_xml")
  raw_metadata     dict                    original format-specific metadata
```

## Author

**Ahmad Fall**, [UMMISCO](https://www.ummisco.fr) / [IRD](https://www.ird.fr)

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
