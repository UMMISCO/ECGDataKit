# Releases

All notable changes to ECGDataKit are documented here.

---

## v0.0.7

### Parsing

- New `SignalCharacteristics` dataclass on `ECGRecord` — captures bits per sample, data encoding, compression, signed/unsigned, channel counts, electrode placement, and signal processing flags
- All 12 parsers now populate `record.signal` with format-specific signal characteristics
- New `DeviceInfo.name` field (distinct from `model`)
- Expanded metadata extraction across all parsers:
  - `recording.technician` — now extracted from Sierra XML, GE MUSE, DICOM, GE MAC 2000
  - `recording.referring_physician` — now extracted from Sierra XML, EDF, GE MUSE, GE MAC 2000
  - `patient.weight` — now extracted from SCP-ECG, BeneHeart R12, GE MAC 2000
  - `patient.height` — now extracted from BeneHeart R12, GE MAC 2000
  - `patient.race` — now extracted from HL7 aECG, Mortara EL250, GE MAC 2000
  - `patient.clinical_history` — now extracted from WFDB comment lines
  - `filters.notch_active` — now properly set when notch filter is present (all parsers)
  - `measurements.rr_interval` — now extracted from Mortara EL250, BeneHeart R12, GE MAC 2000
  - `measurements.qrs_count` — now extracted from Mortara EL250, BeneHeart R12, GE MAC 2000
  - `device.serial_number` — now extracted from HL7 aECG, GE MAC 2000
  - `device.institution` — now extracted from HL7 aECG
- Fix: BeneHeart R12 no longer assigns device name to non-existent `RecordingInfo.device` field
- Fix: SCP-ECG now reads patient weight from Tag 12

### Visualization

- Static plots now auto-display by default (`show=True`); pass `show=False` to get the figure without displaying
- New `x_axis` parameter on single-lead plots: `"time"` (default, seconds) or `"samples"` (sample indices)
- Reduced plot margins and borders
- X-axis time labels now use integer seconds

### Processing & Plotting

- All processing and plotting functions now accept raw numpy arrays directly with `fs=` (sample rate) parameter, in addition to `Lead` objects

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
- DeepFADE denoising autoencoder — a symmetric DenseNet encoder-decoder trained on a large private ECG database with noise augmentations, with pre-trained weights bundled

### Visualization

- Static plots (matplotlib): single lead, multi-lead, 12-lead grid, R-peak annotations, beat overlay, averaged beat, spectrum, spectrogram, RR tachogram, Poincaré plot, HRV dashboard, signal quality chart, full ECG report
- Interactive plots (plotly): all of the above with zoom, pan, hover, and range slider

### Infrastructure

- Documentation site at [ecgdatakit.ummisco.fr](https://ecgdatakit.ummisco.fr)
- GitHub Actions CI: tests across Python 3.10–3.13
- Apache 2.0 license
