# Anonymization

`ecgdatakit anonymize` copies the ECG files of a source folder into an
`ANONYMIZED` folder, with the patient's name and ID and the ECG ID replaced
by pseudonyms, and records the link in a CSV catalog. Raw files are only
read, never modified.

```bash
pip install "ecgdatakit[anonymize]"        # add [dicom] for DICOM files
ecgdatakit anonymize run /path/to/source
```

## Source and datasets

By default the source folder is anonymized as a whole:

```text
source/
  PATIENT-001/V1/ecg.xml              raw file (read only)
  PATIENT-002/ecg.scp
  ANONYMIZED/                         created by the anonymizer
    PATIENT-001/V1/<new name>.xml
    PATIENT-002/<new name>.scp
  anonymization_catalog.csv           re-identification key
```

The first folder level under the source is the patient folder: all files
below it share one patient pseudonym.

With `--datasets`, each sub-folder of the source is a separate dataset
with its own `ANONYMIZED` folder and catalog, and new sub-folders are
picked up automatically:

```text
source/
  dataset-a/ ... ANONYMIZED/  anonymization_catalog.csv
  dataset-b/ ... ANONYMIZED/  anonymization_catalog.csv
```

With `--raw-dir NAME`, only files under folders with that name are read,
wherever they are in the dataset, and the first folder under it is the
patient folder. Other folders (reports, previous exports) are ignored:

```text
dataset-a/
  xml/RAW/PATIENT-001/V1/ecg.xml      read
  pdf/report.pdf                      ignored
  ANONYMIZED/xml/RAW/PATIENT-001/V1/<new name>.xml
```

The anonymized copy keeps the same sub-folders. Folder names are not
anonymized yet (`folder_anonymized` is `FALSE` in the catalog): a patient
folder named after the patient keeps that name in `ANONYMIZED`.

Only files that ecgdatakit reads are processed; every other file (PDF,
images, unsupported formats) is skipped and not copied. EDAN Holter files
are not supported.

## What is replaced

| Format | Patient name | Patient ID | ECG ID |
|--------|--------------|------------|--------|
| HL7 aECG | `subjectDemographicPerson/name` | `trialSubject/id@extension` | `AnnotatedECG/id` (UUID kept as a UUID) |
| GE MUSE XML | `PatientLastName`, `PatientFirstName` | `PatientID`, `SecondaryID` | `PharmaUniqueECGID` |
| Philips Sierra XML | `name/lastname`, `firstname`, `middlename` | `patientid`, `uniquepatientid`, `MRN` | `documentname` |
| Mortara ELI XML | `SUBJECT` name attributes, demographic fields 1 and 7 | `SUBJECT@ID`, demographic field 2 | none |
| ISHNE Holter | first and last name fields | ID field | none |
| EDF / EDF+ | EDF+ name subfield | EDF+ patient code, or the whole plain EDF patient field | EDF+ admin code |
| SCP-ECG | tags 0, 1, 3 | tag 2 | tag 31 |
| MFER | MWF_PNM | MWF_PID | MWF_UID |
| DICOM | PatientName, OtherPatientNames, PatientBirthName | PatientID, OtherPatientIDs | SOPInstanceUID |
| WFDB | `# name:` comment | `# id:` comment | none |
| AliveCor Kardia JSON | none | `patientID` | `id` |

The last name and the ID take the patient pseudonym and the first name is
emptied. The patient's name and IDs are also replaced inside comments and
other free-text fields. Birth date, sex, clinician and operator names,
acquisition dates and visit labels are kept.

Only the bytes of these values change: an XML file keeps its encoding,
indentation and line endings, binary headers keep their size, and a
checksum (ISHNE, SCP-ECG) is computed again when the original one was
valid. DICOM files are written back with pydicom.

Each anonymized file is read back with ecgdatakit before it is published.
It must give the same samples, bit for bit, and the same fields apart from
the replaced values; otherwise it is not published and the catalog marks it
`failed`.

## File names

Parts of the file name equal to the file's name, patient ID or ECG ID
(ignoring case and accents) are replaced; dates, visits and other parts are
kept:

```text
R^ECG^F^0^PID-0042^DOE_20240109083645_V1.xml
R^ECG^F^0^Hh3kQ9sLm2Xa^Hh3kQ9sLm2Xa_20240109083645_V1.xml
```

When a file has no name field, the words of its patient folder name are
used. Parts that may still identify the patient, such as initials or long
random identifiers, are kept and the catalog row is marked `risk` with the
reason. Review these rows.

## Catalog

`anonymization_catalog.csv` has one row per raw file:

- `status`: `anonymized`, `copied` (no identity value in the file),
  `changed` (anonymized again after the raw file changed), `deleted` (raw
  file removed, its copy deleted), `failed` (see `message`).
- `risk`, `risk_reason`.
- `raw_path`, `anonymized_path` (relative to the dataset folder),
  `patient_folder`, `format`.
- `patient_code`, `ecg_code` and the original values
  (`original_patient_id`, `original_last_name`, `original_first_name`,
  `original_ecg_id`, `original_file_name`), `anonymized_ecg_id`,
  `anonymized_file_name`.
- `raw_sha256`, `anonymized_sha256`, `raw_size`, `raw_mtime_ns`.
- `detected_at`, `anonymized_at`, `last_change_at`, `tool_version`.

The catalog links pseudonyms to patients: keep it where only authorized
people can read it.

Pseudonyms are random 12-character codes. Each run checks that every
patient folder has one code and every code one patient folder, that ECG
codes and anonymized paths are unique, and that the number of files
matches between raw and anonymized folders, in total and per patient.

## Runs

```bash
ecgdatakit anonymize run SOURCE                                   # the whole source
ecgdatakit anonymize run SOURCE --datasets --raw-dir RAW          # each sub-folder, RAW folders only
ecgdatakit anonymize run SOURCE --datasets --dataset dataset-a    # one dataset
ecgdatakit anonymize run SOURCE SOURCE/PATIENT-001                # one folder (or file)
ecgdatakit anonymize run SOURCE SOURCE/PATIENT-001 --no-recursive # without its sub-folders
ecgdatakit anonymize run SOURCE --dry-run                         # report only
```

A run only reads files that are new or whose size or modification time
changed since the catalog was written. A changed file is anonymized again
with the same pseudonyms; a deleted file loses its anonymized copy. A
dataset is locked while it is processed (`.anonymize.lock`).

From Python, see the {doc}`API reference <reference/anonymization>`.

## Daemon

```bash
ecgdatakit anonymize daemon SOURCE --datasets --raw-dir RAW --interval 21600 --settle 60
ecgdatakit anonymize shell
```

The daemon takes the same options as `run`. It runs a full pass at start
and then every `--interval` seconds. In between, file-system events start a
pass on the folders that changed, once no change arrived for `--settle`
seconds (files still being copied are not read). Events need `watchdog` and
a local disk: on a network mount, use the scheduled passes only
(`--no-watch`).

The shell talks to the daemon on `127.0.0.1:8765`:

| Command | Effect |
|---------|--------|
| `status` | current activity, last and next pass, last results |
| `datasets` | datasets with their catalog counts |
| `catalog [TEXT] [--dataset NAME] [--status S] [--risk] [--limit N]` | browse a catalog |
| `scan [DATASET]` | start a pass now |

A single command can be run directly: `ecgdatakit anonymize shell status`.

### Docker

```bash
docker build -f docker/anonymizer/Dockerfile -t ecgdatakit-anonymizer .
docker run -d --name ecg-anonymizer --restart unless-stopped \
  -v /path/to/source:/data -p 127.0.0.1:8765:8765 \
  ecgdatakit-anonymizer --datasets --raw-dir RAW
docker exec -it ecg-anonymizer ecgdatakit anonymize shell
```
