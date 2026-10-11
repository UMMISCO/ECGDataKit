# Statistics

Each run counts, for each dataset, what was there before anonymization
and what is in `ANONYMIZED`, side by side, so a mismatch shows at a
glance. The counts are:

- printed at the end of `ecgdatakit anonymize run`;
- written next to the catalog, in `anonymization_stats.csv` (columns
  `count`, `raw`, `anonymized`, `same`);
- shown again at any time by `ecgdatakit anonymize stats`, or by `stats`
  in the shell of a running daemon.

They are computed from the catalog of the whole dataset, not from the
files of one run only.

```bash
ecgdatakit anonymize stats /path/to/dataset
ecgdatakit anonymize stats /path/to/source --datasets
ecgdatakit anonymize stats /path/to/source --datasets --dataset dataset-a
ecgdatakit anonymize shell stats        # same counts, from a running daemon
```

```text
dataset-a
                                                        raw  anonymized
  folders with ECG files                                 52          52
  patient folders                                        40          40
  patients                                               40          40
  files with a patient ID                               120         120
  different patient IDs                                  41          41
  files whose patient ID is not the folder name           3           3
  ECG files                                             120         120
  ECG IDs                                               120         120
```

A line whose two counts differ ends with `differs`.

| Line | Raw | Anonymized |
|------|-----|------------|
| folders with ECG files | Every folder that directly holds at least one raw ECG file, at any depth: a patient folder, and also each of its sub-folders that holds files (`PATIENT-001/visit-1`) | Folders of `ANONYMIZED` holding the copies |
| patient folders | Patient folders holding ECG files: only the top folder of each patient, whatever its sub-folders (with `--group-by-patient-folders`, otherwise 0) | Patient pseudonyms of the files in these folders |
| patients | Patient folders, plus one per file outside a patient folder | Patient pseudonyms |
| files with a patient ID | Raw files whose patient ID field is not empty | Copies whose patient ID field is not empty |
| different patient IDs | Different values of the patient ID field in the raw files | Different values of that field in the copies: the patient pseudonyms, plus the values kept |
| files whose patient ID is not the folder name | Files in a patient folder whose patient ID field does not match the folder name, or holds a date | Copies whose patient ID was kept for that reason (rows marked as risk) |
| ECG files | Raw files (failed ones included, deleted ones not) | Anonymized copies |
| ECG IDs | Different ECG IDs in the raw files | ECG pseudonyms of the files that have an ECG ID |

## Reading the counts

- **Folders with ECG files against patient folders.** They are equal
  when every patient folder holds its files directly. More folders with
  ECG files means some patient folders have sub-folders (visits, exports).
- **Files whose patient ID is not the folder name.** With
  `--group-by-patient-folders`, 0 means every patient ID found in the files
  matches its folder and was replaced. Otherwise these files kept their
  patient ID (marked as risk in the catalog): compare `file_patient_id`
  with `original_patient_id` (the folder name) in those rows.
- **Different patient IDs against patient folders.** With one patient per
  folder and IDs that match, both are equal. More IDs than folders means
  some folders hold several IDs (another patient, or a date); fewer means
  some files have no ID or the same ID is in several folders.
- **Patient IDs before and after.** A replaced ID becomes the patient
  pseudonym and a kept ID stays as is, so the two counts differ when the
  same ID is replaced in two patient folders (each folder gets its own
  pseudonym).
- **ECG IDs before and after.** Each file gets its own ECG pseudonym, so
  fewer raw ECG IDs than pseudonyms means some files share an ECG ID
  (often the same recording exported twice).
- **ECG files.** Fewer copies than raw files means some files failed: see
  the rows with status `failed`.
