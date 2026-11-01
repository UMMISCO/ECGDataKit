# Export

`ecgdatakit.exporter` writes the {class}`~ecgdatakit.models.ECGRecord` that every parser returns to another file format. The record can come from any supported format, so ECG files can be converted between formats. EDAN ARC Holter files are the exception: they do not state a voltage scale, which the formats written here need.

Import: `from ecgdatakit.exporter import HL7aECGExporter, fetch_hl7_aecg_schema, validate_hl7_aecg`

| | |
|---|---|
| {class}`~ecgdatakit.exporter.HL7aECGExporter` | Write a record as an HL7 Annotated ECG (aECG) XML file |
| {func}`~ecgdatakit.exporter.fetch_hl7_aecg_schema` | Download the HL7 aECG schema into a cache folder |
| {func}`~ecgdatakit.exporter.validate_hl7_aecg` | Check an HL7 aECG file against the schema |

## HL7 aECG

HL7 Annotated ECG (ANSI/HL7 V3 ECG R1-2004, message type `PORT_MT020001`) is the XML format of the FDA ECG warehouse. The exporter writes documents that conform to its official XML schema.

```python
from ecgdatakit import FileParser
from ecgdatakit.exporter import HL7aECGExporter

record = FileParser.parse("recording.dcm")
HL7aECGExporter().export(record, "recording.xml")
```

`to_bytes(record)` returns the document instead of writing it. The file is written only once the whole document is built, so a record that cannot be exported leaves no partial file.

XLE chest leads are stored against the right arm (`CR1` to `CR6`) and have no HL7 lead code. Read XLE files with the Wilson correction to export them, and add the missing leads to export the standard 12 leads:

```python
from ecgdatakit.parsing.parsers.xle import XLEParser

record = XLEParser(wilson_correction=True, compute_missing_leads=True).parse("recording.xle")
HL7aECGExporter().export(record, "recording.xml")
```

### Checking against the schema

The HL7 schema is not shipped with ECGDataKit. {func}`~ecgdatakit.exporter.fetch_hl7_aecg_schema` downloads it once into a cache folder ({func}`~ecgdatakit.exporter.default_schema_dir`, or the folder you give). The files come from the copy the FDA distributes with its aECG tools, at a fixed version, and each one must match a known SHA-256 hash. A changed file is refused. Checking needs lxml:

```bash
pip install "ecgdatakit[export]"
```

```python
from ecgdatakit.exporter import HL7aECGExporter, fetch_hl7_aecg_schema, validate_hl7_aecg

fetch_hl7_aecg_schema()                                  # once
HL7aECGExporter(validate=True).export(record, "recording.xml")
errors = validate_hl7_aecg("recording.xml")              # [] when the file conforms
```

With `validate=True` every document is checked before it is returned, and one that does not conform raises {class}`~ecgdatakit.exceptions.ExportError` with the schema errors. Checking is off by default since the schema may not be downloaded. The ECGDataKit tests export a record of each supported format except EDAN ARC Holter, and check each document against the schema.

### What is written

| Record | HL7 aECG |
|---|---|
| `leads` | Rhythm series. One sequence set per sampling rate and length, each with its time sequence and one sequence per lead |
| `representative_beats` | Representative beat series, derived from the rhythm series |
| `recording.date`, `recording.duration` | Effective time of the document and of the series, absolute time of the rhythm samples. Without a date the effective time is written with `nullFlavor="NI"` and sample times are relative |
| `patient.patient_id` | Subject ID |
| `patient.first_name`, `last_name`, `sex`, `birth_date` | Subject demographics |
| `patient.age` | Related observation "Reported Age" (LOINC 21612-7) |
| `patient.clinical_history` | Subject finding comment |
| `recording.device` (manufacturer, model, software version, serial number) | Series author device |
| `recording.device.institution` | Trial site |
| `recording.location` | Testing site |
| `recording.technician` | Series performer |
| `recording.acquisition.filters` | Low-pass, high-pass and notch filter control variables |
| `measurements` | Global measurements: heart rate, RR, PR, QRS, QT, QTc Bazett and Fridericia, P, QRS and T axes |
| `interpretation.statements`, `interpretation.interpretation_date` | Interpretation statements and their date, with the interpreter as author when the interpretation was over-read or confirmed |
| `annotations["rhythm"]` | Rhythm statement |
| `annotations["representative_beat_p_onset_ms"]` and the other wave boundaries in ms | P, QRS and T wave boundaries of the representative beat |

Leads, measurements and waves are coded with the MDC codes of ISO/IEEE 11073. A lead with no MDC code (`CR1` for example) raises `ExportError`. With `skip_unknown_leads=True` it is left out with a warning.

Not written: the detail of each interpretation statement, the severity, the QRS count, the wave boundaries whose unit is not stated (keys without `_ms`), `leads_enhanced`, `ecg_files`, the other patient and recording fields (race, weight, height, medications, pacemaker, referring physician, room), other annotations and `raw_metadata`.

Leads at different sampling rates or lengths go to separate sequence sets, since a sequence set has a single time sequence. ECGDataKit's HL7 aECG parser reads only the first sequence set of a series, so it reads back only the leads of the first one.

### Samples

HL7 aECG stores each sample as an integer with an origin and a scale. The exporter writes them in µV:

- A lead whose values are a whole number of ADC counts is written exactly, with the ADC resolution as scale. This also holds for leads already converted to mV.
- Values that are a fraction of a count, down to 1/12, are written exactly too. The Wilson correction gives thirds of a count and Goldberger leads give halves.
- Other values, such as averaged beats, are rounded to 0.01 µV, or to 1/100 of the ADC resolution when that is finer. A warning gives the largest rounding error. `step_uv` sets the step for every lead instead.

A lead with no voltage scale (uncalibrated counts) or with missing (NaN) samples raises `ExportError`.

### Subject and trial IDs

The HL7 schema requires a subject ID and a clinical trial ID. The subject ID is the patient ID, and the trial ID is `trial_id`. Both are scoped by `id_root`, the OID or UUID of the organisation that assigned them. By default `id_root` is {data}`~ecgdatakit.exporter.DEFAULT_ID_ROOT`, a fixed UUID for IDs exported by ECGDataKit. An ID with no value is written with `nullFlavor="NI"` (no information).

```python
exporter = HL7aECGExporter(id_root="2.16.840.1.113883.3.1234", trial_id="STUDY-01")
```

## Full API

```{eval-rst}
.. currentmodule:: ecgdatakit.exporter

.. autoclass:: HL7aECGExporter
   :members: to_bytes, export

.. autofunction:: fetch_hl7_aecg_schema

.. autofunction:: validate_hl7_aecg

.. autofunction:: default_schema_dir

.. autodata:: DEFAULT_ID_ROOT

.. autoclass:: Exporter
   :members:
```
