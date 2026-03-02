---
title: "Getting Started"
weight: 10
---

This guide walks you through installing ECGDataKit, parsing your first ECG file, processing signals, and creating visualizations.

## Installation

Install the core library with pip:

```bash
pip install .
```

ECGDataKit ships several optional dependency groups. Install only what you need:

| Extra | Command | Description |
|-------|---------|-------------|
| `processing` | `pip install ".[processing]"` | Signal filtering, peak detection, HRV analysis |
| `plotting` | `pip install ".[plotting]"` | Static Matplotlib-based plots |
| `plotting-interactive` | `pip install ".[plotting-interactive]"` | Interactive Plotly-based plots |
| `holter` | `pip install ".[holter]"` | ISHNE Holter format CRC validation |
| `dicom` | `pip install ".[dicom]"` | DICOM waveform parsing via pydicom |
| `cleaning` | `pip install ".[cleaning]"` | BioSPPy + NeuroKit2 ECG cleaning backends |
| `denoising` | `pip install ".[denoising]"` | DeepFADE neural-net denoiser (torch) |
| `all` | `pip install ".[all]"` | Everything above (except torch) |

## Parsing an ECG file

`FileParser` auto-detects the file format and returns a unified `ECGRecord` object.

```python
from ecgdatakit.parsing import FileParser

record = FileParser().parse("path/to/ecg_file.xml")
```

### Patient information

```python
patient = record.patient
print(patient.name)        # Patient name
print(patient.id)          # Patient ID
print(patient.birth_date)  # Date of birth
print(patient.sex)         # Sex
```

### Recording information

```python
recording = record.recording
print(recording.date)         # Acquisition date
print(recording.sample_rate)  # Sampling frequency in Hz
print(recording.duration)     # Duration in seconds
```

### Lead data

```python
leads = record.leads
for lead in leads:
    print(lead.name, lead.samples[:5])
```

### Measurements and device info

```python
# Automated measurements (if present in the file)
measurements = record.measurements
print(measurements.heart_rate)
print(measurements.pr_interval)
print(measurements.qrs_duration)

# Device that recorded the ECG
device = record.device
print(device.manufacturer)
print(device.model)
```

### Serialization

```python
# Export to JSON string
json_str = record.to_json()

# Export to Python dict
data = record.to_dict()
```

## Processing signals

The `ecgdatakit.processing` module provides filtering, peak detection, heart-rate analysis, HRV metrics, signal quality assessment, and lead derivation.

### Filtering

```python
from ecgdatakit.processing import diagnostic_filter, remove_baseline

# Apply a diagnostic-grade bandpass filter (0.05 - 150 Hz)
filtered = diagnostic_filter(lead.samples, fs=recording.sample_rate)

# Remove baseline wander
corrected = remove_baseline(lead.samples, fs=recording.sample_rate)
```

### Peak detection and heart rate

```python
from ecgdatakit.processing import detect_r_peaks, heart_rate, rr_intervals

# Detect R-peaks
peaks = detect_r_peaks(filtered, fs=recording.sample_rate)

# Compute heart rate in BPM
hr = heart_rate(peaks, fs=recording.sample_rate)

# Get RR intervals in milliseconds
rr = rr_intervals(peaks, fs=recording.sample_rate)
```

### Heart-rate variability (HRV)

```python
from ecgdatakit.processing import time_domain

# Time-domain HRV metrics (SDNN, RMSSD, pNN50, etc.)
hrv = time_domain(rr)
print(hrv)
```

### Signal quality

```python
from ecgdatakit.processing import signal_quality_index

sqi = signal_quality_index(filtered, fs=recording.sample_rate)
print(f"Signal quality: {sqi}")
```

### Lead derivation

```python
from ecgdatakit.processing import derive_augmented

# Derive augmented limb leads (aVR, aVL, aVF) from I and II
augmented = derive_augmented(lead_I.samples, lead_II.samples)
```

### ECG cleaning

```python
from ecgdatakit.processing import clean_ecg

# Built-in cleaning (bandpass + notch, no extra deps)
cleaned = clean_ecg(lead)

# NeuroKit2 backend (pip install ecgdatakit[cleaning])
cleaned = clean_ecg(lead, method="neurokit2")
```

### Neural-net denoising

```python
from ecgdatakit.processing import clean_ecg

# DeepFADE DenseNet denoiser (pip install ecgdatakit[denoising])
denoised = clean_ecg(lead, method="deepfade")

# Use MPS acceleration on Apple Silicon
denoised = clean_ecg(lead, method="deepfade", device="mps")
```

## Visualizing

ECGDataKit provides both static (Matplotlib) and interactive (Plotly) plotting functions.

### Static plots

```python
from ecgdatakit.plotting import plot_12lead, plot_peaks, plot_hrv_summary

# Standard 12-lead ECG layout
plot_12lead(record)

# Overlay detected R-peaks on a filtered signal
plot_peaks(filtered, peaks)

# HRV summary plot (Poincare, histogram, tachogram)
plot_hrv_summary(rr)
```

### Interactive plots

```python
from ecgdatakit.plotting import iplot_lead, iplot_12lead

# Interactive single-lead viewer with zoom and pan
iplot_lead(filtered, peaks)

# Interactive 12-lead viewer
iplot_12lead(record)
```

## Batch processing

Parse multiple files in parallel with `parse_batch`:

```python
from ecgdatakit.parsing import parse_batch
from pathlib import Path

files = list(Path("data/").glob("*.xml"))
records = parse_batch(files, max_workers=4)

for rec in records:
    print(rec.patient.name, rec.recording.date)
```

## Adding a new parser

To add support for a new ECG format:

1. Create a new file in `src/ecgdatakit/parsing/parsers/`.
2. Subclass `Parser` and implement two methods: `can_parse` and `parse`.

```python
from ecgdatakit.parsing.parsers.base import Parser
from ecgdatakit.parsing.models import ECGRecord

class MyFormatParser(Parser):
    """Parser for the MyFormat ECG file type."""

    def can_parse(self, path: str) -> bool:
        """Return True if this parser can handle the given file."""
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic == b"MYFT"

    def parse(self, path: str) -> ECGRecord:
        """Parse the file and return an ECGRecord."""
        # Read and decode the file
        # Build and return an ECGRecord
        ...
```

The `FileParser` will automatically discover your new parser and try it during format detection.
