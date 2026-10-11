# Anonymization

ECGDataKit anonymizes ECG files in any supported format. Each file is
copied into an `ANONYMIZED` folder with the patient's name, the ECG ID
and, when it can be trusted, the patient ID replaced by pseudonyms, in
the file and in its name. A CSV catalog links every raw file to its
anonymized copy. Raw files are only read, never modified.

To run it, see {doc}`anonymization/usage`.

## How it works

1. **Find the files.** The source (a single file or a folder) is scanned
   for files that ecgdatakit reads. Every other file (PDF, images,
   unsupported formats) is skipped and not copied. EDAN Holter files are
   not supported.
2. **Read the identity values.** The patient's name and ID and the ECG ID
   are read from the fields of each format (see
   [What is replaced](#what-is-replaced)). The patient ID is replaced only
   with `--group-by-patient-folders` (see [Patients](#patients)). With
   `--anonymize-patient-folders`, the name of the patient folder is added
   to the values to replace.
3. **Choose the pseudonyms.** Each file gets its own ECG pseudonym. Each
   patient gets one patient pseudonym: a patient folder with
   `--group-by-patient-folders`, otherwise each file.
4. **Write the copy.** Only the bytes of the identity values change, then
   the copy gets its new file name.
5. **Check the copy.** It is read back with ecgdatakit and must give the
   same samples, bit for bit, and the same fields apart from the replaced
   values. A copy that fails is not kept and the catalog marks it `failed`.
6. **Record it** in the catalog, with the original and anonymized values.

(anonymization-patients)=
## Patients

Three options decide how patients are handled. They are independent and
all off by default:

| Option | What it does |
|--------|--------------|
| `--patients-dir-name NAME` | Only reads files under the folders called `NAME` (for example `RAW`), and says the patient folders are their sub-folders. It does not group anything by itself. |
| `--group-by-patient-folders` | Each patient folder is one patient: all its files share one patient pseudonym. The patient folders are the sub-folders of the `NAME` folders, or without `--patients-dir-name` the folders directly inside the dataset. |
| `--anonymize-patient-folders` | Renames each patient folder to its patient pseudonym. Needs `--group-by-patient-folders`. |

### Grouping

Files are only grouped by patient folder, never by the patient ID or name
read in them: some devices store something else in that field (a date, a
study code), and grouping by it would merge the recordings of different
patients.

| | Patient pseudonym |
|-|-------------------|
| Without `--group-by-patient-folders` | each file has its own, even files of the same folder |
| With it, file in a patient folder | shared by all the files of that folder |
| With it, file outside a patient folder | its own |

### The patient ID field

For the same reason, the patient ID field of a file is only replaced when
the patient is known from the patient folder:

| | Patient ID field |
|-|------------------|
| Without `--group-by-patient-folders` | never replaced: kept as is in the file, its free text and its name |
| With it, file in a patient folder | replaced by the patient pseudonym, whatever it holds: the folder says who the patient is |
| With it, file outside a patient folder | kept as is, and the catalog row is marked as risk |

When the ID does not match the patient folder name (a typing error such as
`PAT-002-003-AB` in folder `PAT-01-003-AB`, or the ID of another patient),
it is still replaced, and the catalog row is marked as risk with the ID
read in the file, for example `patient ID field (PAT-002-003-AB) does not
match the patient folder name, replaced`. These rows can be reviewed with
the clinicians: a file may be in the wrong patient folder.

The ID matches the folder name when both are equal ignoring case, accents
and separators (`SITE-001-0001` and `site 001 0001`), or when one appears
as whole words in the other and has at least 4 characters: the ID inside
the folder name (`001-0003` in folder `SITE_001_0003_AB_CD`), or the
folder name inside the ID (folder `PAT-01-0015-CD` in ID
`PAT 01 0015 CD V1PRE`). The patient's name and the ECG ID
are always replaced, whatever the options.

### Renaming patient folders

With `--anonymize-patient-folders`, the patient folder name is treated as
one more identifier of the patient, for folders named after the patient
or their hospital number:

- the patient folder is named after the patient pseudonym in `ANONYMIZED`;
- the folder name is replaced by the pseudonym wherever it appears in the
  files (any field or free text, all formats), in the file names, and in
  the names of the sub-folders of the patient folder;
- it is matched as a whole, ignoring case, accents and spacing: a part of
  it (such as a number alone) is not replaced;
- folders above the patient folders keep their names.

Without it, every folder keeps its name in `ANONYMIZED`, and a folder
named after the patient shows that name. The catalog column
`folder_anonymized` says which of the two was used for each file.

## What is replaced

| Format | Patient name | Patient ID | ECG ID |
|--------|--------------|------------|--------|
| HL7 aECG | `subjectDemographicPerson/name` | `trialSubject/id@extension` | `AnnotatedECG/id` |
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

- The last name takes the patient pseudonym and the first name is
  emptied. The patient ID takes the patient pseudonym only as described
  in [The patient ID field](#the-patient-id-field).
- The ECG ID takes the ECG pseudonym.
- The patient's name and ID (and, with `--anonymize-patient-folders`, the
  patient folder name) are also replaced inside comments and other
  free-text fields.
- Birth date, sex, clinician and operator names, acquisition dates and
  visit labels are kept.

**Dates in the patient ID field.** Some devices store a date in the
patient ID field (for example `28 11 23`). A value written as a date
(`28 11 23`, `28/11/2023`, `2023-11-28`) is never replaced, even with
`--group-by-patient-folders`: it is kept as is in the file and in the
file name, and the catalog row is marked as risk with the reason.

**Format kept.** Only the bytes of the identity values change: an XML
file keeps its encoding, indentation and line endings, binary headers keep
their size, and a checksum (ISHNE, SCP-ECG) is computed again when the
original one was valid. DICOM files are written back with pydicom.

## File names

Parts of the file name equal to the patient's name, the ECG ID or the
patient ID when it is replaced in the file (ignoring case and accents)
are replaced; dates, visits and other parts are kept:

```text
R^ECG^F^0^PID-0042^DOE_20240109083645_V1.xml
R^ECG^F^0^Hh3kQ9sLm2Xa^Hh3kQ9sLm2Xa_20240109083645_V1.xml
```

- With `--anonymize-patient-folders`, the patient folder name is replaced
  in file names too.
- With `--group-by-patient-folders`, a file that has no name field also
  has the words of its patient folder name replaced in its file name.
- Parts that may still identify the patient, such as initials or long
  random identifiers, are kept and the catalog row is marked `risk` with
  the reason. Review these rows.

## Pseudonyms

- **Patient pseudonyms** are random 12-character codes, for example
  `qifUPPHzDGKG`.
- **ECG pseudonyms** are random UUIDs, written as is in place of the ECG
  ID. DICOM only accepts digits and dots, so it gets the same UUID in its
  standard UID form (`2.25.` followed by the UUID as a number).

A code already used in the dataset is never given again. A file anonymized
again after a change keeps its pseudonyms.

## Catalog

`anonymization_catalog.csv` has one row per raw file:

| Column | Content |
|--------|---------|
| `status` | `anonymized`, `copied` (no identity value in the file), `changed` (anonymized again after the raw file changed), `deleted` (raw file removed, its copy deleted), `failed` (see `message`) |
| `risk`, `risk_reason` | `TRUE` when something should be reviewed, and why |
| `raw_path`, `anonymized_path` | Paths relative to the dataset folder |
| `patient_folder` | What groups the patient's files: the patient folder, or `file:` followed by the file's path when it is not in a patient folder |
| `format` | Parser that read the file |
| `original_patient_id` | With `--group-by-patient-folders`, the patient folder name, whatever the file holds. Without it, empty (the patient ID is not replaced) |
| `anonymized_patient_id` | The patient pseudonym, the same for every file of the patient |
| `file_patient_id` | The patient ID field as read in the raw file. With `--group-by-patient-folders` it becomes `anonymized_patient_id` in the anonymized file (except a date); `risk_reason` says when it did not match the folder name |
| `original_ecg_id` | The ECG ID read in the raw file |
| `anonymized_ecg_id` | The ECG pseudonym, one per file |
| `original_last_name`, `original_first_name` | The names replaced (several values are separated by a vertical bar) |
| `original_file_name`, `anonymized_file_name` | File name before and after |
| `folder_anonymized` | `TRUE` when the patient folder was renamed |
| `companions` | Other anonymized files of the same recording, for formats stored as several files: a WFDB record is a `.hea` header (the row) plus its `.dat` signal files (listed here). Empty for other formats |
| `duplicate` | Raw paths of the other files with exactly the same content (several separated by a vertical bar), empty when the file is unique. Each copy is anonymized on its own, with its own ECG pseudonym; a copy in another patient folder gets that folder's patient pseudonym |
| `raw_size`, `raw_mtime_ns`, `raw_sha256`, `anonymized_sha256` | Used to detect changes and check copies |
| `detected_at`, `anonymized_at`, `last_change_at` | Dates of the events |
| `message`, `tool_version` | Failure message, ecgdatakit version |

The catalog links pseudonyms to patients: keep it where only authorized
people can read it. {doc}`anonymization/stats` counts the raw and the
anonymized files, each read again from disk, after each run, and writes
them to `anonymization_stats.csv` next to the catalog.

## Checks

At the end of each run, for every dataset:

- each patient has one pseudonym and each pseudonym one patient;
- ECG pseudonyms and anonymized paths are unique;
- every raw file has its anonymized copy (or is marked `failed`), in total
  and per patient (when the whole dataset is processed, not only some of
  its paths).

A problem is reported and the run ends with exit code 1. The summary of
the run shows the counts side by side:

```text
  120 raw file(s) -> 120 anonymized, 120 ECG pseudonym(s); 40 patient(s) -> 40 patient pseudonym(s)
```

```{toctree}
:hidden:

anonymization/usage
anonymization/stats
```
