# ECGDataKit

**A pure-Python library for parsing, processing, and visualizing multi-format ECG files.**

Developed at [UMMISCO](https://www.ummisco.fr) / [IRD](https://www.ird.fr) by Ahmad Fall.

---

## Features

### Parsing — 12 ECG formats, one unified data model

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

### Signal Processing — powered by SciPy

- Butterworth filters (lowpass, highpass, bandpass, notch, baseline removal)
- Diagnostic (0.05–150 Hz) and monitoring (0.67–40 Hz) presets
- R-peak detection (Pan-Tompkins and Shannon energy)
- Heart rate, RR intervals, instantaneous HR
- HRV: time-domain (SDNN, RMSSD, pNN50), frequency-domain (VLF/LF/HF), Poincaré (SD1/SD2)
- FFT, PSD, beat segmentation, ensemble averaging
- Signal quality index, SNR estimation
- Lead derivation (III, aVR/aVL/aVF, full 12-lead assembly)
- Resampling, normalization (min-max, z-score, amplitude)
- ECG cleaning (built-in, BioSPPy, NeuroKit2, combined pipelines)
- Deep learning denoising via DeepFADE (DenseNet encoder-decoder)

### Visualization — static and interactive

- Standard 12-lead grid with paper-style background
- R-peak annotations with RR intervals and heart rate
- Beat segmentation overlay, ensemble-averaged beat with SD shading
- Power spectrum (Welch PSD, FFT), spectrogram
- HRV dashboard (tachogram, Poincaré plot, frequency bands, metrics table)
- Signal quality bar chart per lead
- Full ECG report page with patient info, measurements, interpretation
- Interactive versions of all plots via Plotly (zoom, pan, hover)

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

# With DeepFADE neural-net denoising (requires torch)
pip install "ecgdatakit[denoising]"

# Everything (except torch — install separately if needed)
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
print(record.recording.sample_rate)    # 500
print(record.measurements.heart_rate)  # 75
print(record.device.manufacturer)      # "Philips"
print(len(record.leads))               # 12

json_str = record.to_json()
```

### Process signals

```python
from ecgdatakit.processing import (
    diagnostic_filter, detect_r_peaks, heart_rate,
    rr_intervals, time_domain, signal_quality_index, clean_ecg,
)

lead = record.leads[1]

filtered = diagnostic_filter(lead)

peaks = detect_r_peaks(filtered)
peaks_se = detect_r_peaks(filtered, method="shannon_energy")

hr = heart_rate(filtered, peaks)
rr = rr_intervals(filtered, peaks)

hrv = time_domain(rr)
print(hrv["sdnn"], hrv["rmssd"], hrv["pnn50"])

sqi = signal_quality_index(lead)

cleaned = clean_ecg(lead)
cleaned = clean_ecg(lead, method="neurokit2")
cleaned = clean_ecg(lead, method="deepfade")
```

### Visualize

```python
from ecgdatakit.plotting import (
    plot_12lead, plot_peaks, plot_hrv_summary,
    iplot_lead, iplot_12lead,
)

# Static 12-lead ECG report (matplotlib)
fig = plot_12lead(record)
fig.savefig("ecg_12lead.png", dpi=150)

# R-peak annotations
fig = plot_peaks(filtered, peaks)

# HRV dashboard
fig = plot_hrv_summary(rr)

# Interactive plots (plotly) — opens in browser
iplot_lead(filtered, peaks).show()
iplot_12lead(record).show()
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

All parsers produce the same `ECGRecord`:

```
ECGRecord
  patient: PatientInfo        # ID, name, birth date, sex, age, weight, height, medications
  recording: RecordingInfo    # date, duration, sample rate, ADC gain, technician, physician
  device: DeviceInfo          # manufacturer, model, serial number, software version, institution
  filters: FilterSettings     # highpass, lowpass, notch frequencies
  leads: list[Lead]           # label, samples (float64 array), sample rate, units
  interpretation: Interpretation  # statements, severity, source, interpreter
  measurements: GlobalMeasurements  # HR, PR, QRS, QT, QTc, axes
  median_beats: list[Lead]    # median/template beats if available
  annotations: dict[str, str] # additional key-value annotations
  source_format: str          # parser identifier
  raw_metadata: dict          # original format-specific metadata
```

## Exceptions

All exceptions inherit from `ECGDataKitError`:

| Exception | When raised |
|-----------|-------------|
| `UnsupportedFormatError` | File format not recognized |
| `CorruptedFileError` | File is truncated or structurally invalid |
| `MissingElementError` | Required element or field is missing |
| `ChecksumError` | Checksum validation failed |

## Testing

```bash
pip install -e ".[all,dev,holter,dicom]"
pytest tests/ -v
```

## Author

**Ahmad Fall** — [UMMISCO](https://www.ummisco.fr) / [IRD](https://www.ird.fr)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
