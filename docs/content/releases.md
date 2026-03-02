---
title: "Releases"
weight: 50
---

# Releases

All notable changes to ECGDataKit are documented here.

---

## v0.0.6 — Initial Release

**Release date:** March 2, 2026

The first public release of ECGDataKit.

### Parsing

- 12 ECG format parsers: HL7 aECG, Philips Sierra XML, ISHNE Holter, Mortara EL250, EDF/EDF+, SCP-ECG, GE MUSE XML, DICOM Waveform, WFDB (PhysioNet), MFER, Mindray BeneHeart R12, GE MAC 2000
- Unified `ECGRecord` data model across all formats
- Auto-detection of file format via `FileParser`
- Batch parsing with `parse_batch()` and multiprocessing support
- JSON serialization via `ECGRecord.to_json()`

### Signal Processing

- Butterworth filters: lowpass, highpass, bandpass, notch, baseline removal
- Diagnostic (0.05–150 Hz) and monitoring (0.67–40 Hz) filter presets
- R-peak detection: Pan-Tompkins and Shannon energy algorithms
- Heart rate, RR intervals, instantaneous heart rate
- HRV analysis: time-domain (SDNN, RMSSD, pNN50), frequency-domain (VLF/LF/HF), Poincaré (SD1/SD2)
- FFT, power spectral density, beat segmentation, ensemble averaging
- Signal quality index (SQI) and SNR estimation
- Lead derivation (III, aVR/aVL/aVF, full 12-lead assembly)
- Resampling and normalization (min-max, z-score, amplitude)
- ECG cleaning: built-in, BioSPPy, NeuroKit2, combined pipelines
- DeepFADE neural-net denoising (DenseNet encoder-decoder)

### Visualization

- Static plots (matplotlib): single lead, multi-lead, 12-lead grid, R-peak annotations, beat overlay, averaged beat, spectrum, spectrogram, RR tachogram, Poincaré plot, HRV dashboard, signal quality chart, full ECG report
- Interactive plots (plotly): all of the above with zoom, pan, hover, and range slider

### Infrastructure

- Documentation site at [ecgdatakit.ummisco.fr](https://ecgdatakit.ummisco.fr)
- GitHub Actions CI: tests across Python 3.10–3.13
- Apache 2.0 license
