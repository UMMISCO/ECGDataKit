---
title: "Parsing API Reference"
weight: 10
---

Import from the top-level package: `from ecgdatakit import FileParser, ECGRecord`

## FileParser

The main entry point. Auto-discovers all available parsers and dispatches files to the correct one based on content sniffing.

```python
from ecgdatakit import FileParser

fp = FileParser()
fp.parsers           # list of discovered Parser subclasses
record = fp.parse("ecg_file.xml")
```

<table>
  <thead><tr><th>Method</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>parse(file_path)</code></td><td><code>ECGRecord</code></td><td>Auto-detect format and parse the file</td></tr>
  </tbody>
</table>

## Parser (base class)

Abstract base class for all format-specific parsers. Located at `ecgdatakit.parsing.parser.Parser`.

<table>
  <thead><tr><th>Method</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>can_parse(file_path, header) </code><em>static</em></td><td><code>bool</code></td><td>Return <code>True</code> if this parser handles the file</td></tr>
    <tr><td><code>parse(file_path)</code></td><td><code>ECGRecord</code></td><td>Parse the file and return a unified record</td></tr>
  </tbody>
</table>

## parse_batch

```python
from ecgdatakit import parse_batch

records = list(parse_batch(file_list, max_workers=4))
```

Parse multiple files in parallel using `ProcessPoolExecutor`. Returns an iterator of `ECGRecord` objects.

## Data Models

All models are Python `dataclass` instances defined in `ecgdatakit.models`.

### ECGRecord

The unified output type returned by every parser.

<table>
  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>patient</code></td><td><code>PatientInfo</code></td><td>Patient demographics</td></tr>
    <tr><td><code>recording</code></td><td><code>RecordingInfo</code></td><td>Recording session metadata</td></tr>
    <tr><td><code>device</code></td><td><code>DeviceInfo</code></td><td>Acquisition device info</td></tr>
    <tr><td><code>filters</code></td><td><code>FilterSettings</code></td><td>Filter settings applied during acquisition</td></tr>
    <tr><td><code>leads</code></td><td><code>list[Lead]</code></td><td>ECG lead waveforms</td></tr>
    <tr><td><code>interpretation</code></td><td><code>Interpretation</code></td><td>Machine or physician interpretation</td></tr>
    <tr><td><code>measurements</code></td><td><code>GlobalMeasurements</code></td><td>Global ECG interval/axis measurements</td></tr>
    <tr><td><code>median_beats</code></td><td><code>list[Lead]</code></td><td>Median/template beats if available</td></tr>
    <tr><td><code>annotations</code></td><td><code>dict[str, str]</code></td><td>Additional key-value annotations</td></tr>
    <tr><td><code>source_format</code></td><td><code>str</code></td><td>Parser identifier</td></tr>
    <tr><td><code>raw_metadata</code></td><td><code>dict</code></td><td>Original format-specific metadata</td></tr>
  </tbody>
</table>

<table>
  <thead><tr><th>Method</th><th>Returns</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>to_dict(include_samples=True)</code></td><td><code>dict</code></td><td>JSON-serialisable dictionary</td></tr>
    <tr><td><code>to_json(include_samples=True, indent=2)</code></td><td><code>str</code></td><td>JSON string</td></tr>
  </tbody>
</table>

### PatientInfo

<table>
  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>patient_id</code></td><td><code>str</code></td><td>Patient identifier</td></tr>
    <tr><td><code>first_name</code></td><td><code>str</code></td><td>First name</td></tr>
    <tr><td><code>last_name</code></td><td><code>str</code></td><td>Last name</td></tr>
    <tr><td><code>birth_date</code></td><td><code>datetime | None</code></td><td>Date of birth</td></tr>
    <tr><td><code>sex</code></td><td><code>str</code></td><td><code>"M"</code>, <code>"F"</code>, or <code>"U"</code></td></tr>
    <tr><td><code>race</code></td><td><code>str</code></td><td>Race/ethnicity</td></tr>
    <tr><td><code>age</code></td><td><code>int | None</code></td><td>Age in years</td></tr>
    <tr><td><code>weight</code></td><td><code>float | None</code></td><td>Weight in kg</td></tr>
    <tr><td><code>height</code></td><td><code>float | None</code></td><td>Height in cm</td></tr>
    <tr><td><code>medications</code></td><td><code>list[str]</code></td><td>Current medications</td></tr>
    <tr><td><code>clinical_history</code></td><td><code>str</code></td><td>Clinical history notes</td></tr>
  </tbody>
</table>

### RecordingInfo

<table>
  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>date</code></td><td><code>datetime | None</code></td><td>Recording start time</td></tr>
    <tr><td><code>end_date</code></td><td><code>datetime | None</code></td><td>Recording end time</td></tr>
    <tr><td><code>duration</code></td><td><code>timedelta | None</code></td><td>Recording duration</td></tr>
    <tr><td><code>sample_rate</code></td><td><code>int</code></td><td>Samples per second (Hz)</td></tr>
    <tr><td><code>adc_gain</code></td><td><code>float</code></td><td>ADC gain factor (default 1.0)</td></tr>
    <tr><td><code>technician</code></td><td><code>str</code></td><td>Technician name</td></tr>
    <tr><td><code>referring_physician</code></td><td><code>str</code></td><td>Referring physician name</td></tr>
    <tr><td><code>room</code></td><td><code>str</code></td><td>Room identifier</td></tr>
    <tr><td><code>location</code></td><td><code>str</code></td><td>Facility/location</td></tr>
  </tbody>
</table>

### DeviceInfo

<table>
  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>manufacturer</code></td><td><code>str</code></td><td>Device manufacturer</td></tr>
    <tr><td><code>model</code></td><td><code>str</code></td><td>Device model name</td></tr>
    <tr><td><code>serial_number</code></td><td><code>str</code></td><td>Device serial number</td></tr>
    <tr><td><code>software_version</code></td><td><code>str</code></td><td>Software version</td></tr>
    <tr><td><code>institution</code></td><td><code>str</code></td><td>Institution name</td></tr>
    <tr><td><code>department</code></td><td><code>str</code></td><td>Department name</td></tr>
    <tr><td><code>acquisition_type</code></td><td><code>str</code></td><td>Acquisition type</td></tr>
  </tbody>
</table>

### FilterSettings

<table>
  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>highpass</code></td><td><code>float | None</code></td><td>Highpass cutoff (Hz)</td></tr>
    <tr><td><code>lowpass</code></td><td><code>float | None</code></td><td>Lowpass cutoff (Hz)</td></tr>
    <tr><td><code>notch</code></td><td><code>float | None</code></td><td>Notch frequency (Hz)</td></tr>
    <tr><td><code>notch_active</code></td><td><code>bool | None</code></td><td>Whether notch filter is active</td></tr>
    <tr><td><code>artifact_filter</code></td><td><code>bool | None</code></td><td>Whether artifact filter is active</td></tr>
  </tbody>
</table>

### Interpretation

<table>
  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>statements</code></td><td><code>list[str]</code></td><td>Interpretation text statements</td></tr>
    <tr><td><code>severity</code></td><td><code>str</code></td><td><code>"NORMAL"</code>, <code>"ABNORMAL"</code>, <code>"BORDERLINE"</code></td></tr>
    <tr><td><code>source</code></td><td><code>str</code></td><td><code>"machine"</code>, <code>"overread"</code>, <code>"confirmed"</code></td></tr>
    <tr><td><code>interpreter</code></td><td><code>str</code></td><td>Physician name (if overread)</td></tr>
    <tr><td><code>interpretation_date</code></td><td><code>datetime | None</code></td><td>When interpretation was made</td></tr>
  </tbody>
</table>

### GlobalMeasurements

<table>
  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>heart_rate</code></td><td><code>int | None</code></td><td>Heart rate (bpm)</td></tr>
    <tr><td><code>rr_interval</code></td><td><code>int | None</code></td><td>RR interval (ms)</td></tr>
    <tr><td><code>pr_interval</code></td><td><code>int | None</code></td><td>PR interval (ms)</td></tr>
    <tr><td><code>qrs_duration</code></td><td><code>int | None</code></td><td>QRS duration (ms)</td></tr>
    <tr><td><code>qt_interval</code></td><td><code>int | None</code></td><td>QT interval (ms)</td></tr>
    <tr><td><code>qtc_bazett</code></td><td><code>int | None</code></td><td>QTc Bazett (ms)</td></tr>
    <tr><td><code>qtc_fridericia</code></td><td><code>int | None</code></td><td>QTc Fridericia (ms)</td></tr>
    <tr><td><code>p_axis</code></td><td><code>int | None</code></td><td>P-wave axis (degrees)</td></tr>
    <tr><td><code>qrs_axis</code></td><td><code>int | None</code></td><td>QRS axis (degrees)</td></tr>
    <tr><td><code>t_axis</code></td><td><code>int | None</code></td><td>T-wave axis (degrees)</td></tr>
    <tr><td><code>qrs_count</code></td><td><code>int | None</code></td><td>Total QRS count</td></tr>
  </tbody>
</table>

### Lead

<table>
  <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>label</code></td><td><code>str</code></td><td>Lead name (<code>"I"</code>, <code>"V1"</code>, etc.)</td></tr>
    <tr><td><code>samples</code></td><td><code>NDArray[np.float64]</code></td><td>Signal sample values</td></tr>
    <tr><td><code>sample_rate</code></td><td><code>int</code></td><td>Samples per second (Hz)</td></tr>
    <tr><td><code>resolution</code></td><td><code>float</code></td><td>Resolution (nV/unit, default 1.0)</td></tr>
    <tr><td><code>units</code></td><td><code>str</code></td><td>Signal units (e.g. <code>"mV"</code>)</td></tr>
    <tr><td><code>quality</code></td><td><code>int | None</code></td><td>Signal quality indicator</td></tr>
    <tr><td><code>transducer</code></td><td><code>str</code></td><td>Transducer type</td></tr>
    <tr><td><code>prefiltering</code></td><td><code>str</code></td><td>Pre-filtering description</td></tr>
  </tbody>
</table>

## Exceptions

All exceptions inherit from `ECGDataKitError`.

<table>
  <thead><tr><th>Exception</th><th>When Raised</th></tr></thead>
  <tbody>
    <tr><td><code>ECGDataKitError</code></td><td>Base exception for all errors</td></tr>
    <tr><td><code>UnsupportedFormatError</code></td><td>File format not recognized</td></tr>
    <tr><td><code>CorruptedFileError</code></td><td>File is truncated or structurally invalid</td></tr>
    <tr><td><code>MissingElementError</code></td><td>Required element or field is missing</td></tr>
    <tr><td><code>ChecksumError</code></td><td>Checksum validation failed</td></tr>
  </tbody>
</table>

```python
from ecgdatakit import FileParser, UnsupportedFormatError

try:
    record = FileParser().parse("unknown.bin")
except UnsupportedFormatError as e:
    print(f"Format not supported: {e}")
```
