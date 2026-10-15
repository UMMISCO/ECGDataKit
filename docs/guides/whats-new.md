# What's new

## 2.0.4

### Parsing

- **XLE: representative beat.** The averaged beat stored in the XLE
  header is now read into `record.median_beats`, 1 s at 5000 Hz, with a
  warning. The file stores the sum of the averaged beats as
  sample-to-sample changes. The parser adds them up and divides by the
  number of beats averaged. The beat follows the leads: `CR1` to `CR6` by
  default, the Wilson correction and the missing leads when they are
  requested.
- **XLE: scale of the representative beat.** Its scale is not stored. It
  is set to the scale of the leads divided by 1.6, the ratio found between
  the beat and the median beat of the leads on sample files. The beat is
  converted to mV with the leads, and the scale warning mentions the
  factor. See [XLE ECG](formats.md#xle-ecg-experimental) for how it was
  found.
- **XLE: values stored with the beat.** Its QRS onset and offset and the
  number of beats averaged go to `record.annotations`. The number of QRS
  complexes detected goes to `record.measurements.qrs_count`.

### Anonymization

- **File names are kept by default.** Each anonymized file keeps its raw
  name. A name holding a known identity value (the patient's name or ID,
  the ECG ID, or the patient folder name with
  `--anonymize-patient-folders`) is marked as risk in the catalog.
- **New `--anonymize-file-names` option** (`anonymize_file_names=True`).
  Each anonymized file is named `<patient pseudonym>_<ECG pseudonym>`,
  as 2.0.0 to 2.0.3 always did. Turning it on or off renames the existing
  copies on the next run, with the same pseudonyms. The catalog has a new
  `file_name_anonymized` column.

## 2.0.3

### Parsing

- **XLE: standard 12 leads.**
  `XLEParser(wilson_correction=True, compute_missing_leads=True)` adds
  III, aVR, aVL and aVF, computed from I and II, with a warning. The
  record then holds the standard 12 leads. `compute_missing_leads`
  requires the Wilson correction.
- **`FileParser.detect_format(path)`** tells the format of a file and the
  parser that reads it, without parsing it.
- **`FileParser` needs no instance.** `FileParser.parse(path)` works like
  `FileParser().parse(path)`.

## 2.0.2

### Parsing

- **XLE (experimental): voltage scale.** Samples are converted to mV with
  0.9375 µV per count. The file does not store this scale and it is not
  validated by the manufacturer's documentation, so the conversion warns
  about it.
- **XLE: chest leads.** Leads are named I, II and `CR1` to `CR6`, since
  the chest leads are stored against the right arm.
  `XLEParser(wilson_correction=True)` refers them to Wilson's central
  terminal (`V1` to `V6`), with a warning.
- **XLE: recording date.** The recording date and time are now read from
  the file.

## 2.0.1

### Parsing

- **New format (experimental): XLE ECG (`.xle`).** 8-channel ECG
  recordings read from their reverse-engineered layout. The file does not
  state the voltage scale, so samples stay in raw ADC counts. ECGDataKit
  now reads 13 formats (see {doc}`formats`).
- XLE files can be anonymized: given name emptied, family name and
  patient ID replaced by the patient pseudonym, names and IDs replaced in
  the history and comment fields. Birth date and sex are kept.

## 2.0.0

### Anonymization

- **New `ecgdatakit.anonymize` package and `ecgdatakit anonymize`
  command.** ECG files are copied into an `ANONYMIZED` folder with the
  patient's name, the ECG ID and, when it can be trusted, the patient ID
  replaced by pseudonyms, and gets a new file name made of the
  pseudonyms. Raw files are only read. A CSV catalog links every raw file
  to its anonymized copy. See {doc}`../api/anonymization`.
- Every format ecgdatakit reads is supported, except EDAN Holter. Only the
  bytes of the identity values change: each file keeps its format, and
  each copy is read back and must give the same samples and fields before
  it is kept.
- A single file, a folder, or several datasets (`--datasets`) can be
  anonymized. Runs only process new, changed and deleted files.
- Patients are grouped only by patient folder
  (`--group-by-patient-folders`, `--patients-dir-name`), never by an ID
  read in the files. Patient folders can be renamed to their pseudonyms
  (`--anonymize-patient-folders`).
- Identical files are the same ECG: they share one ECG pseudonym, and the
  catalog lists the copies of each file (`duplicate` column).
- `ecgdatakit anonymize daemon` keeps a folder anonymized as files arrive,
  `ecgdatakit anonymize shell` shows what it is doing, and a Dockerfile
  runs it in a container.
- After each run, the raw files and the anonymized files are counted
  again from disk and checked (folders, patient folders, ECG files, patient
  IDs, duplicates, ECG IDs). The counts are printed, written to
  `anonymization_stats.csv`, and shown by `ecgdatakit anonymize stats`.
- Install with `pip install "ecgdatakit[anonymize]"` (also part of
  `[all]`).

### Documentation

- New layout: header bar with search, grouped sidebar, "On this page"
  panel, and edit and report links on every page.
- "API Reference" is renamed "Docs". The function reference becomes
  "API Reference", with Parsing, Processing, Plotting and Anonymization
  pages.
- New Anonymization and Usage pages.

## 1.1.1

### New format: AliveCor Kardia JSON

- ECG recordings exported from the Kardia API (KardiaMobile and
  KardiaMobile 6L) can now be read. The library reads 12 formats (see
  {doc}`formats`).
- The unfiltered signal goes to `record.leads`. The signal filtered by
  AliveCor goes to the new `record.leads_enhanced` field, which other
  formats leave empty.
- Samples are scaled with `amplitudeResolution` (nanovolts per count), as
  described in the Kardia API documentation.
- The AliveCor algorithm result (for example `afib`) is a machine
  interpretation.
- Kardia 12L recordings use another layout and raise
  `UnsupportedFormatError`.

### Detected format and parser

`record.file_format.name` holds the name of the detected format and
`record.file_format.parser` the parser that read the file. They are filled
in when the file is read through `FileParser` or `parse_batch`.

### Annotations

These changes rename or remove annotation keys. Code that reads them may
need updating.

- A value with a field in `record.measurements` is stored only there, not
  also in `record.annotations` (Sierra and Mortara repeated some values).
- Each wave boundary has its own key. HL7 aECG and SCP-ECG no longer write
  them into one `wave_boundaries_ms` text.
- Boundaries of the representative beat start with `median_`, for example
  `median_p_onset_ms` or `median_q_onset` (HL7 aECG, SCP-ECG, Mortara,
  GE MUSE, Sierra).
- Keys end with their unit when the format states it: `_ms`, `_samples`
  or `_s`.
- DICOM time-referenced annotations moved from
  `raw_metadata["waveform_annotations"]` to `record.annotations`, one key
  per wave type, for example `p_onset_samples`.

## 1.1.0

This release makes the library stricter and more honest about what it
reads. Some code that worked with 1.0 may now raise an error or give a
different result. This page lists every change you may notice.

### Reading files

- **Two formats removed.** GE MAC 2000 and Mindray BeneHeart R12 are no
  longer supported. These files now raise `UnsupportedFormatError`. The
  library reads 11 formats (see {doc}`formats`).
- **EDAN Holter support is experimental.** EDAN does not document its
  format, so every EDAN file now gives a warning. Check the values against
  the recorder's own export before using them.
- **Leads come in millivolts by default.** `FileParser().parse(path)`
  converts every lead whose file gives a voltage scale to mV
  (`auto_scale=True`). Choose another unit with `units="uV"` or `units="V"`.
  Leads whose file gives no scale stay in raw counts, and you get a warning.
  With `auto_scale=False` you get the numbers exactly as stored in the
  file, plus the scale needed to convert them (`resolution`,
  `resolution_unit`, `offset`).
- **Only what the file stores.** The library no longer works out values
  the file does not contain: no intervals computed from wave boundaries, no
  heart rate from RR intervals, no QTc from a guessed formula, no end date
  from start plus duration, no extra leads (such as III or aVR) built from
  the others, and no pacemaker or severity guessed from the text. A field
  the file does not give stays empty. Where a format specification defines
  a default (for example a sampling rate), it is used with a warning and
  marked in `raw_metadata`.
- **Who wrote the interpretation.** `record.interpretation.source` is
  `"machine"` for the device's automatic statements, and `"confirmed"` or
  `"overread"` when a physician reviewed them (with the physician's name
  and date when stored). When both exist, the device statements are kept
  in `record.annotations["machine_interpretation"]`.
- **Clearer errors.** A file no parser recognises raises
  `UnsupportedFormatError` (it is also a `ValueError`, so existing
  `except ValueError` code still works). Empty files raise instead of
  returning an empty record. A broken file raises `CorruptedFileError`
  with the reason. A missing optional package (for example pydicom for
  DICOM) raises `ImportError` with the install command.

### Records and leads

- `lead.to_physical()` raises `RawSamplesError` when the file gives no
  unit for the samples: uncalibrated counts cannot be turned into volts.
- `record.to_json()` produces strict JSON: missing or non-finite numbers
  are written as `null`, never `NaN`. For long recordings, write directly
  to a file with `record.to_json(fp=f)` to avoid building a huge string.

### Parsing many files

- `parse_batch(..., on_error="warn")` skips files that fail and reports
  each one as a `BatchParseWarning`, and `on_error="ignore"` skips them
  silently. The default, `"raise"`, stops at the first failure.
- `parse_batch(..., executor="serial")` parses in the current process,
  which is faster for long Holter recordings. The default uses worker
  processes. On macOS and Windows, call it under
  `if __name__ == "__main__":`.

### Signal processing

- Filters, resampling, peak detection, spectra, quality and cleaning raise
  `ValueError` for signals that contain NaN or infinite values and for 2-D
  arrays, instead of returning wrong results. Normalization still accepts
  2-D arrays, one lead per row.
- `clean_ecg` rejects options that the chosen method does not use, and
  raises clear errors when a backend cannot handle the signal.
- Normalized signals (min-max, z-score) have no unit: their `units` is
  empty, so they cannot be mistaken for voltages.
- Leads derived from others (III, aVR, aVL, aVF) stay in raw counts only
  when both inputs are raw on the same scale. Otherwise they come back in
  physical units (the unit of lead I). Raw inputs with different scales and
  no unit raise `ValueError`.
- Filtering a lead that is still in raw counts now takes its `offset` into
  account, so the result converts to the right voltage.
- R-peak detection (`detect_r_peaks`, Pan-Tompkins by default) is more
  reliable: better thresholds while learning, search back for missed beats,
  either QRS polarity, and any sampling rate.
- **DeepFADE is experimental.** It warns when used, is not validated for
  clinical use, and refuses leads that are still in raw counts: convert
  them to volts first.

### Plotting

- Time axes are exact (`sample / sampling rate`). Beat plots show time
  relative to the R-peak, so 0 ms is the R-peak.
- Leads of different lengths are no longer cut to the shortest one.
- Each plot shows the amplitude unit, or "raw counts".
- The optional ECG paper grid uses 0.2 s and 0.5 mV squares.
- `plot_report` uses the standard 3 x 4 layout (2.5 s per column) with a
  10 s rhythm strip.
- Very long leads, such as a 24 h Holter, are drawn with a reduced number
  of points that keeps every peak, and you get a warning. Slice the lead
  to see every sample.
- Plots no longer change your global matplotlib style.
