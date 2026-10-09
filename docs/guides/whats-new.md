# What's new

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
  each one as a `BatchParseWarning`; `on_error="ignore"` skips them
  silently. The default, `"raise"`, stops at the first failure.
- `parse_batch(..., executor="serial")` parses in the current process,
  which is faster for long Holter recordings. The default uses worker
  processes; on macOS and Windows call it under
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
