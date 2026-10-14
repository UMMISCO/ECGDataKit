# Statistics

Each run counts, for each dataset, the raw ECG files and the anonymized
ECG files side by side, and checks that anonymization did what it should.
Both columns are counted from disk:

- **raw**: the raw files of the dataset (only those under the
  `--patients-dir-name` folders when it is given), each file read again.
- **anonymized**: the files of the `ANONYMIZED` folder, each file read
  again. Nothing is taken from the raw side or from the catalog, so a copy
  that is missing, moved or still holds an identity value shows up.

The counts are:

- printed at the end of `ecgdatakit anonymize run`
- written next to the catalog, in `anonymization_stats.csv` (columns
  `count`, `raw`, `anonymized`, `ok`, `check`)
- shown again at any time by `ecgdatakit anonymize stats` (same options
  as `run`), or by `stats` in the shell of a running daemon

```bash
ecgdatakit anonymize stats /path/to/dataset --group-by-patient-folders
ecgdatakit anonymize stats /path/to/source --datasets --patients-dir-name RAW --group-by-patient-folders
ecgdatakit anonymize shell stats
```

```text
dataset-a
                                                                 raw  anonymized  check
  folders holding ECG files, sub-folders included                 52          52  OK     same
  patient folders                                                 42          40  OK     anonymized: those holding ECG files
  patient folders without ECG files (nothing to copy)              2           0  OK     anonymized: 0
  ECG files                                                      120         120  OK     same
  ECG files with a patient ID                                    120         120  OK     same
  ECG files whose patient ID is not their patient folder's         3           0  OK     anonymized: 0
  distinct patient IDs in the files, dates aside                 236          40  OK     anonymized: one per patient folder
  duplicate ECG files (same content as another)                    2           2  OK     anonymized: at most raw
  distinct ECGs (duplicates counted once)                        118         118  OK     same: one ECG pseudonym per distinct ECG
  distinct ECG IDs in the files                                  118         118  OK     anonymized: the ECG pseudonym of each file that has an ECG ID field
  patient folders without ECG files (2): RAW/SUBJ-0007, RAW/SUBJ-0019
```

Each line ends with `OK` when its check holds, `CHECK` when it does not. The patient folders that hold no ECG file are listed below the table
(in the printed summary and the `stats` command, not in the CSV file).

| Line | Raw | Anonymized | Check |
|------|-----|------------|-------|
| folders holding ECG files, sub-folders included | Every folder that directly holds an ECG file, at any depth: a patient folder and each of its sub-folders that holds files (`PATIENT-001/visit-1`) | The same, in `ANONYMIZED` | same |
| patient folders | Every patient folder on disk: the folders directly under the `--patients-dir-name` folders, or directly in the dataset (with `--group-by-patient-folders`, otherwise 0) | The same, in `ANONYMIZED` | anonymized: those holding ECG files |
| patient folders without ECG files (nothing to copy) | Patient folders where no file is read by ecgdatakit (empty, or only PDFs, images, unsupported formats): nothing is copied for them. Their names are printed below the table | The same, in `ANONYMIZED` | anonymized: 0 |
| ECG files | ECG files | ECG files | same |
| ECG files with a patient ID | Files whose patient ID field is not empty | The same | same |
| ECG files whose patient ID is not their patient folder's | Files whose patient ID does not match the patient folder name, or holds a date | Files whose patient ID is not the one of their patient folder: the folder name when folders are renamed, otherwise the ID most files of the folder hold. Dates are left aside | anonymized: 0 |
| distinct patient IDs in the files, dates aside | Different values of the patient ID field (the same patient often has several: visit codes, typing errors) | Different values in the anonymized files | with grouping: one per patient folder, never shared by two folders. Without grouping: same |
| duplicate ECG files (same content as another) | Files identical to another file, each extra copy counted | The same, in `ANONYMIZED`: copies in the same patient folder stay identical (same pseudonyms), copies in two patient folders differ by their patient pseudonym | anonymized: at most raw |
| distinct ECGs (duplicates counted once) | Different file contents | Different ECG pseudonyms: read from the anonymized file names with `--anonymize-file-names`, otherwise from the catalog row of each anonymized file | same: one ECG pseudonym per distinct ECG |
| distinct ECG IDs in the files | Different ECG IDs stored in the files (0 for formats without an ECG ID field: Mortara, ISHNE, WFDB) | The same, in the anonymized files | anonymized: one per ECG pseudonym of the files that have an ECG ID field |

## Reading the counts

- **A `CHECK` line** means the anonymized files are not what they should
  be: look at the files of `ANONYMIZED`, and at the catalog rows marked
  `failed` or `risk`.
- **ECG files whose patient ID is not their patient folder's**, raw
  side: typing errors, visit codes that hide the folder name, or a file in
  the wrong patient folder. Their rows are marked as risk in the catalog
  with the ID read in the file. Review them with the clinicians.
- **Distinct patient IDs**, raw side: devices often add a visit code to
  the ID (`SUBJ-0042 VISIT2`), so one patient gives several values.
  After anonymization each patient folder holds one pseudonym.
- **Duplicate files**: identical raw files (a `- Copie` next to its
  original, the same file in two folders) are the same ECG and share one
  ECG pseudonym. Each copy is still written in `ANONYMIZED`. The catalog
  column `duplicate` lists, for each file, the other files with the same
  content.
- **Distinct ECG IDs at 0**: the format has no ECG ID field (Mortara,
  ISHNE, WFDB) or it is empty. The ECG pseudonym is then only in the
  catalog (and in the file name with `--anonymize-file-names`), and
  "distinct ECGs" still counts them.
- **Dates in the patient ID field** are kept as they are, so they are left
  aside in the patient ID lines.
