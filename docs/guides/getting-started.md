# Getting Started

This guide walks you through installing ECGDataKit, parsing your first ECG file, processing signals, and creating visualizations.

## Installation

### From PyPI

```bash
pip install ecgdatakit
```

### From GitHub (latest stable version)

```bash
pip install "ecgdatakit @ git+https://github.com/UMMISCO/ECGDataKit.git#subdirectory=lib"
```

### From GitLab (development version)

```bash
pip install "ecgdatakit @ git+https://git.ummisco.fr/open/ecgdatakit.git#subdirectory=lib"
```

### From source (local clone)

```bash
# GitHub (stable)
git clone https://github.com/UMMISCO/ECGDataKit.git

# GitLab (development)
git clone https://git.ummisco.fr/open/ecgdatakit.git

cd ECGDataKit/lib
pip install .
```

### Optional extras

ECGDataKit ships several optional dependency groups. Install only what you need:

| Extra | Command | Description |
|-------|---------|-------------|
| `processing` | `pip install "ecgdatakit[processing]"` | Signal filtering, peak detection, HRV analysis |
| `plotting` | `pip install "ecgdatakit[plotting]"` | Static Matplotlib-based plots (includes scipy) |
| `plotting-interactive` | `pip install "ecgdatakit[plotting-interactive]"` | Interactive Plotly-based plots (includes scipy) |
| `dicom` | `pip install "ecgdatakit[dicom]"` | DICOM waveform parsing via pydicom |
| `cleaning` | `pip install "ecgdatakit[cleaning]"` | BioSPPy + NeuroKit2 ECG cleaning backends |
| `denoising` | `pip install "ecgdatakit[denoising]"` | Experimental DeepFADE denoising autoencoder (torch) |
| `all` | `pip install "ecgdatakit[all]"` | Everything above except torch |

## Parsing an ECG file

`FileParser` auto-detects the file format and returns a unified `ECGRecord` object.

```python
from ecgdatakit.parsing import FileParser

record = FileParser().parse("path/to/ecg_file.xml")
```

Leads are returned in millivolts by default (`auto_scale=True`). Leads whose
file gives no voltage scale stay in raw counts, with a warning. Pass
`auto_scale=False` to get the values stored in the file; see {doc}`../api/scaling`.

### Patient information

```python
patient = record.patient
print(patient.first_name)   # First name
print(patient.last_name)    # Last name
print(patient.patient_id)   # Patient ID
print(patient.birth_date)   # Date of birth
print(patient.sex)          # Sex ("M", "F", or "U")
```

### Recording information

```python
recording = record.recording
print(recording.date)         # Acquisition date
print(recording.acquisition.signal.sampling_rate)  # Sampling frequency in Hz
print(recording.duration)     # Duration as timedelta
```

### Lead data

```python
leads = record.leads
for lead in leads:
    print(lead.label, lead.samples[:5])
```

### Measurements and device info

```python
# Automated measurements (if present in the file)
measurements = record.measurements
print(measurements.heart_rate)
print(measurements.pr_interval)
print(measurements.qrs_duration)

# Device that recorded the ECG
device = record.recording.device
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

lead = record.leads[0]  # Lead object from a parsed record

# Apply a diagnostic-grade bandpass filter (0.05 - 150 Hz). Below 333 Hz
# sampling the upper cutoff is lowered to 0.45 * fs, with a warning.
filtered = diagnostic_filter(lead)

# Remove baseline wander
corrected = remove_baseline(lead)
```

You can also pass numpy arrays directly with `fs=`:

```python
filtered = diagnostic_filter(my_numpy_array, fs=500)
```

### Peak detection and heart rate

```python
from ecgdatakit.processing import detect_r_peaks, heart_rate, rr_intervals

# Detect R-peaks (returns a numpy array of sample indices)
peaks = detect_r_peaks(filtered)

# Compute heart rate in BPM
hr = heart_rate(filtered, peaks)

# Get RR intervals in milliseconds
rr = rr_intervals(filtered, peaks)
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

sqi = signal_quality_index(filtered)
print(f"Signal quality: {sqi}")
```

### Lead derivation

```python
from ecgdatakit.processing import derive_augmented, find_lead

lead_i = find_lead(record.leads, "I")
lead_ii = find_lead(record.leads, "II")

# Derive augmented limb leads (aVR, aVL, aVF) from I and II
avr, avl, avf = derive_augmented(lead_i, lead_ii)
```

### ECG cleaning

```python
from ecgdatakit.processing import clean_ecg

# Built-in cleaning (bandpass + notch, no extra deps)
cleaned = clean_ecg(lead)

# NeuroKit2 backend (pip install "ecgdatakit[cleaning]")
cleaned = clean_ecg(lead, method="neurokit2")
```

### Neural-net denoising

```{warning}
DeepFADE is experimental: it emits a warning when used, it is not validated
for clinical use, and its output can change between releases.
```

DeepFADE is a denoising autoencoder developed as part of ECGDataKit. It uses a symmetric DenseNet encoder-decoder architecture trained on a large private ECG database with extensive noise augmentations. Pre-trained weights are bundled with the package.

```python
from ecgdatakit.processing import clean_ecg

# DeepFADE denoising autoencoder (pip install "ecgdatakit[denoising]")
denoised = clean_ecg(lead, method="deepfade")

# Use MPS acceleration on Apple Silicon
denoised = clean_ecg(lead, method="deepfade", device="mps")
```

## Visualizing

ECGDataKit provides both static (Matplotlib) and interactive (Plotly) plotting functions.

### Static plots

Static plots auto-display by default. Pass `show=False` to get the figure without displaying.

```python
from ecgdatakit.plotting import plot_12lead, plot_peaks, plot_lead, plot_hrv_summary

# Displays automatically (one row per lead, amplitude unit in each title)
plot_12lead(record)
plot_peaks(filtered, peaks)
plot_hrv_summary(rr)

# Suppress display to save to file
fig = plot_12lead(record, show=False)
fig.savefig("ecg_12lead.png", dpi=150)

# Use sample indices instead of time on x-axis
plot_lead(filtered, x_axis="samples")

# ECG paper grid: 0.2 s and 0.5 mV squares (amplitude lines need a voltage unit)
plot_lead(record.leads[1], show_grid=True)
```

`plot_report(record)` draws a report page with the standard 3 x 4 layout
(each column shows its own 2.5 s of the first 10 s), a 10 s lead II rhythm
strip and signal quality for the same 10 s. For long recordings such as
Holters, plot a slice of the lead to look past the first 10 s.

### Interactive plots

```python
from ecgdatakit.plotting import iplot_lead, iplot_12lead

# Interactive single-lead viewer with zoom and pan
iplot_lead(filtered, peaks)

# Interactive 12-lead viewer
iplot_12lead(record)
```

Interactive traces longer than 200 000 samples are drawn with min/max
decimation (peaks are kept) and a warning, so a 24 h Holter stays usable in
the browser.

## Batch processing

Parse multiple files in parallel with `parse_batch`. It uses worker
processes, so a script must call it under `if __name__ == "__main__":`
(required on macOS and Windows):

```python
from pathlib import Path

from ecgdatakit.parsing import parse_batch

if __name__ == "__main__":
    files = list(Path("data/").glob("*.xml"))
    records = parse_batch(files, max_workers=4, on_error="warn")

    for rec in records:
        print(rec.raw_metadata["filepath"], rec.recording.date)
```

`on_error="raise"` (default) stops at the first failure, `"warn"` skips the
file with a `BatchParseWarning` and `"ignore"` skips it silently.

## Adding a new parser

`FileParser` discovers every `Parser` subclass in the
`ecgdatakit.parsing.parsers` package, so a new format is one module there:

1. Create `src/ecgdatakit/parsing/parsers/my_format.py`.
2. Subclass `Parser`, set `FORMAT_NAME`, `FILE_EXTENSIONS` and `PRIORITY`
   (lower runs first: use a low value for a reliable magic number, a high one
   for a loose sniffer), and implement the static `can_parse(file_path, header)`
   and `parse(self, file_path)`.

The example reads a made-up format: the magic `MYFT`, then the sampling rate,
the number of leads and the scale in nV per count (three little-endian
`uint16`), then a 4-byte ASCII label per lead, then interleaved `int16` samples.

```python
from pathlib import Path

import numpy as np

from ecgdatakit.exceptions import CorruptedFileError
from ecgdatakit.models import ECGRecord, Lead, derive_is_raw
from ecgdatakit.parsing.helpers import normalize_lead_label, unique_labels
from ecgdatakit.parsing.parser import Parser


class MyFormatParser(Parser):
    """Parser for the MyFormat ECG file type."""

    FORMAT_NAME = "MyFormat"
    FORMAT_DESCRIPTION = "Example binary format with a MYFT magic number"
    FILE_EXTENSIONS = [".myf"]
    PRIORITY = 10

    @staticmethod
    def can_parse(file_path: Path, header: bytes) -> bool:
        # header holds the first 4096 bytes of the file
        return header[:4] == b"MYFT"

    def parse(self, file_path: Path) -> ECGRecord:
        data = Path(file_path).read_bytes()
        if len(data) < 10:
            raise CorruptedFileError(f"MyFormat: {file_path.name} is truncated")
        rate, n_leads, nv_per_count = np.frombuffer(data, "<u2", 3, offset=4).tolist()
        start = 10 + 4 * n_leads
        if rate == 0 or n_leads == 0 or len(data) < start:
            raise CorruptedFileError(f"MyFormat: {file_path.name} has an invalid header")
        names = [data[10 + 4 * i:14 + 4 * i].decode("ascii").strip() for i in range(n_leads)]
        body = data[start:]
        body = body[: len(body) // (2 * n_leads) * 2 * n_leads]
        samples = np.frombuffer(body, "<i2").reshape(-1, n_leads).T.astype(np.float64)

        resolution = nv_per_count / 1000.0  # uV per count
        leads = [
            Lead(
                label=label,
                samples=samples[i],
                sampling_rate=rate,
                resolution=resolution,
                resolution_unit="uV",
                adc_resolution=float(nv_per_count),
                adc_resolution_unit="nV",
                is_raw=derive_is_raw(resolution, 0.0, "uV"),
            )
            for i, label in enumerate(unique_labels([normalize_lead_label(n) for n in names]))
        ]
        return ECGRecord(source_format="my_format", leads=leads)
```

Return the samples exactly as stored with their scale in `resolution` and
`resolution_unit`: `FileParser` converts them to mV when `auto_scale=True`
and fills the signal summary. Report only values the file stores, raise
`CorruptedFileError` for invalid input, and warn (and flag it in
`raw_metadata`) when you use a default from the format specification.
