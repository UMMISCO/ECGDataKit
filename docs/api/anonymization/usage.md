# Usage

## Examples

Each block shows alternatives, not steps to run one after the other. The
commands are explained in the sections that follow. Running the same
command again only processes new, changed and deleted files.

**One file**

```bash
ecgdatakit anonymize run /path/to/ecg.xml
```

**One dataset**

```bash
# Every ECG file of the folder
ecgdatakit anonymize run /path/to/dataset

# Only the files under the RAW folder, one patient per sub-folder of RAW
ecgdatakit anonymize run /path/to/dataset --patients-dir-name RAW

# Same, and the patient folders named after the patient pseudonyms
ecgdatakit anonymize run /path/to/dataset --patients-dir-name RAW --anonymize-patient-folders

# Only one patient folder of the dataset
ecgdatakit anonymize run /path/to/dataset /path/to/dataset/xml/RAW/PATIENT-001 --patients-dir-name RAW

# See what would be done, write nothing
ecgdatakit anonymize run /path/to/dataset --patients-dir-name RAW --dry-run
```

**Several datasets** (each sub-folder of the source is a dataset)

```bash
# All datasets, each with its own ANONYMIZED folder and catalog
ecgdatakit anonymize run /path/to/datasets --datasets --patients-dir-name RAW

# Only two of them
ecgdatakit anonymize run /path/to/datasets --datasets --patients-dir-name RAW --dataset dataset-a --dataset dataset-b

# One line per file anonymized, changed or at risk
ecgdatakit anonymize run /path/to/datasets --datasets --patients-dir-name RAW -v
```

**Daemon** (keeps the source anonymized as files arrive, stop it with Ctrl+C)

```bash
# Local disk: new files handled about a minute after they stop changing,
# plus a full pass every 6 hours
ecgdatakit anonymize daemon /path/to/datasets --datasets --patients-dir-name RAW

# Network mount: no live detection, a full pass every 10 minutes
ecgdatakit anonymize daemon /path/to/datasets --datasets --patients-dir-name RAW --no-watch --interval 600
```

**Shell** (from another terminal, while the daemon runs)

```bash
ecgdatakit anonymize shell                                         # interactive prompt
ecgdatakit anonymize shell status                                  # what the daemon is doing, next pass
ecgdatakit anonymize shell datasets                                # files per status, per dataset
ecgdatakit anonymize shell catalog --dataset dataset-a --risk      # rows to review
ecgdatakit anonymize shell catalog --dataset dataset-a --status failed
ecgdatakit anonymize shell catalog --dataset dataset-a PATIENT-001 # rows of one patient
ecgdatakit anonymize shell scan dataset-a                          # start a pass now
```

**Docker** (the daemon in a container, the source mounted on `/data`)

```bash
docker build -f docker/anonymizer/Dockerfile -t ecgdatakit-anonymizer .
docker run -d --name ecg-anonymizer --restart unless-stopped \
  -v /path/to/datasets:/data -p 127.0.0.1:8765:8765 \
  ecgdatakit-anonymizer --datasets --patients-dir-name RAW
docker exec -it ecg-anonymizer ecgdatakit anonymize shell status
```

## Single file

```bash
ecgdatakit anonymize run path/to/ecg.xml
```

The copy is written to `path/to/ANONYMIZED/` and the catalog to
`path/to/anonymization_catalog.csv`.

## Single folder

```bash
ecgdatakit anonymize run path/to/source
```

The whole folder is anonymized:

```text
source/
  PATIENT-001/V1/ecg.xml              raw file (read only)
  PATIENT-002/ecg.scp
  ANONYMIZED/                         created by the anonymizer
    PATIENT-001/V1/<new name>.xml
    PATIENT-002/<new name>.scp
  anonymization_catalog.csv           re-identification key
```

Every file of every folder is processed (except the output folder) and
the same tree is rebuilt in `ANONYMIZED`. Folder names are kept. Files
with the same patient ID (or, without ID, the same name) share one patient
pseudonym, wherever they are (see {ref}`how patients are identified
<anonymization-patients>`).

To process only part of the source, give the files or folders after it:

```bash
ecgdatakit anonymize run SOURCE SOURCE/PATIENT-001                # one folder
ecgdatakit anonymize run SOURCE SOURCE/PATIENT-001 --no-recursive # without its sub-folders
ecgdatakit anonymize run SOURCE --dry-run                         # report only, write nothing
```

## Several datasets

With `--datasets`, each sub-folder of the source is a separate dataset
with its own `ANONYMIZED` folder and catalog. New sub-folders are picked
up automatically:

```bash
ecgdatakit anonymize run SOURCE --datasets
ecgdatakit anonymize run SOURCE --datasets --dataset dataset-a    # one dataset
```

```text
source/
  dataset-a/ ... ANONYMIZED/  anonymization_catalog.csv
  dataset-b/ ... ANONYMIZED/  anonymization_catalog.csv
```

## Patient folders

When each patient has a folder, and these folders are inside a folder with
the same name in every dataset (for example `RAW`), give that name with
`--patients-dir-name`:

```bash
ecgdatakit anonymize run SOURCE --datasets --patients-dir-name RAW
```

```text
dataset-a/
  xml/RAW/PATIENT-001/V1/ecg.xml      read, patient PATIENT-001
  xml/RAW/PATIENT-001/V2/ecg.xml      read, same patient
  xml/RAW/PATIENT-002/ecg.xml         read, patient PATIENT-002
  pdf/report.pdf                      ignored (not under RAW)
  ANONYMIZED/xml/RAW/PATIENT-001/V1/<new name>.xml
```

- Only files under the `RAW` folders are read. Other folders (reports,
  previous exports) are ignored.
- Each sub-folder of `RAW` is one patient: all its files share one patient
  pseudonym, even when the ID is missing or different in some files.
- Folder names are copied as they are: `PATIENT-001` stays `PATIENT-001`
  in `ANONYMIZED`.

### Anonymizing patient folder names

When the patient folders are named after the patients (a name, initials,
a hospital number), add `--anonymize-patient-folders`. It is off by
default and needs `--patients-dir-name`:

```bash
ecgdatakit anonymize run SOURCE --datasets --patients-dir-name RAW --anonymize-patient-folders
```

```text
dataset-a/
  xml/RAW/DOE Jane/V1/DOE Jane_2023.xml
  ANONYMIZED/xml/RAW/qifUPPHzDGKG/V1/qifUPPHzDGKG_2023.xml
```

The folder name is then one more identifier of the patient:

| Where | Without the option | With the option |
|-------|--------------------|-----------------|
| Patient folder in `ANONYMIZED` | same name as the raw folder | patient pseudonym |
| Folder name inside the files (any field, free text) | kept | replaced by the patient pseudonym |
| Folder name in file names | kept (replaced only when the file has no name field) | replaced by the patient pseudonym |
| Sub-folders of the patient folder (`V1`, dates...) | kept | kept, except parts equal to the patient's name, ID or folder name |
| Folders above the patient folders (`xml`, `RAW`) | kept | kept |

The folder name is matched as a whole, ignoring case, accents and spacing:
a part of it, such as a number alone, is never replaced. Turning the
option on or off for a dataset already anonymized moves the existing
copies on the next run; the pseudonyms do not change.

## Running it again

Running the same command again updates the dataset, it does not start
over:

| Since the last run | What the run does |
|--------------------|-------------------|
| Nothing changed | Nothing: unchanged files are not even read |
| New file | Anonymized; a new file of a known patient gets that patient's pseudonym |
| File modified | Anonymized again with the same pseudonyms, the old copy replaced, status `changed` |
| File deleted | Its anonymized copy deleted, status `deleted` |
| Anonymized copy deleted by hand | Written again |
| `--anonymize-patient-folders` switched on or off | Copies moved to the right folders, same pseudonyms |

Files are compared by size and modification time, and by content when
those differ. The catalog is saved during the run, so an interrupted run
continues where it stopped. A dataset is locked while it is processed
(`.anonymize.lock`), so two runs never write the same catalog.

Keep the same `--patients-dir-name` setting for a dataset: patients are
grouped differently with and without it, and mixing the two would give a
patient two pseudonyms. A run with another setting stops without changing
anything. To change it, delete the dataset's `ANONYMIZED` folder and
catalog and run again: every pseudonym is then drawn again.

## From Python

```python
from ecgdatakit.anonymize import Anonymizer

Anonymizer("path/to/ecg.xml").run()                                  # a single file
Anonymizer("path/to/source").run()                                   # a folder
reports = Anonymizer("path/to/source", datasets=True, patients_dir_name="RAW").run()
for report in reports:
    print(report.dataset, report.anonymized, report.failed, report.errors)
```

Every class and option is listed in the {doc}`API reference <../reference/anonymization>`.

## Keeping a folder anonymized

The daemon keeps a source anonymized while files arrive, for example on a
shared drive where new recordings are uploaded every day:

```bash
ecgdatakit anonymize daemon SOURCE --datasets --patients-dir-name RAW
```

It runs a full pass at start, then:

- **When files appear, change or are deleted**, it processes the folders
  concerned once nothing has changed in them for `--settle` seconds (60
  by default), so a file still being copied is never read. This needs the
  `watchdog` package and a local disk: the operating system does not report
  changes made through a network mount.
- **Every `--interval` seconds** (6 hours by default), it runs a full pass
  again. This catches what the live detection missed, such as changes made
  while the daemon was stopped. With `--no-watch`, these passes are the only
  detection, so choose a shorter interval.

`ecgdatakit anonymize shell`, from another terminal, shows what the daemon
is doing and lets you browse the catalogs or start a pass (see
[shell](#ecgdatakit-anonymize-shell) below).

### Docker

The image runs the daemon on the folder mounted on `/data`. Options given
after the image name are passed to the daemon:

```bash
docker build -f docker/anonymizer/Dockerfile -t ecgdatakit-anonymizer .
docker run -d --name ecg-anonymizer --restart unless-stopped \
  -v /path/to/source:/data -p 127.0.0.1:8765:8765 \
  ecgdatakit-anonymizer --datasets --patients-dir-name RAW
docker exec -it ecg-anonymizer ecgdatakit anonymize shell
```

## Command line

Options written without a value are switches: giving them turns the behaviour on.

### `ecgdatakit anonymize run`

Anonymizes once and exits.

```text
ecgdatakit anonymize run SOURCE [PATH ...]
    [--datasets] [--dataset NAME ...] [--patients-dir-name NAME]
    [--anonymize-patient-folders]
    [--out-dir NAME] [--catalog NAME] [--threads N]
    [--no-recursive] [--dry-run] [-v]
```

| Argument | Description |
|----------|-------------|
| `SOURCE` | Folder or single file to anonymize. The output folder and the catalog are created in it (in its folder for a file). |
| `PATH ...` | Optional. Only these files or folders inside `SOURCE` are processed. If not given, all of `SOURCE` is. |

| Option | Default | Description |
|--------|---------|-------------|
| `--datasets` | | If set, each sub-folder of `SOURCE` is a separate dataset with its own output folder and catalog. If not set, `SOURCE` is anonymized as one whole. |
| `--dataset NAME` | all | With `--datasets` only. If set, only this dataset is processed; give it several times for several datasets. If not set, every dataset is processed. |
| `--patients-dir-name NAME` | none | If set, only files under the folder with this name (the same in every dataset) are read, and each of its sub-folders is one patient. If not set, every file is read and patients are told apart by their patient ID. |
| `--anonymize-patient-folders` | | With `--patients-dir-name` only. If set, each patient folder is named after the patient pseudonym in the output, and the patient folder name is replaced by the pseudonym wherever it appears in the files and their names. If not set, folder names are copied as they are and only the identity fields of each format are used. |
| `--out-dir NAME` | `ANONYMIZED` | Output folder created in each dataset. |
| `--catalog NAME` | `anonymization_catalog.csv` | Catalog file created in each dataset. |
| `--threads N` | 8 | Number of files processed in parallel. |
| `--no-recursive` | | If set, the sub-folders of the folders given as `PATH` are not looked into. If not set, they are. No effect without `PATH`. |
| `--dry-run` | | If set, reports what would be done and writes nothing. |
| `-v`, `--verbose` | | If set, prints one line per file anonymized, changed or at risk. If not set, only the summary and the failures are printed. |

At the end, each dataset gets two lines:

```text
dataset-a: 120 file(s), 118 anonymized, 2 copied, 0 changed, 0 unchanged, 0 deleted, 0 failed, 3 with risks, 15 not ECG files (skipped)
  120 raw file(s) -> 120 anonymized, 120 ECG pseudonym(s); 40 patient(s) -> 40 patient pseudonym(s)
```

The first line gives what the run did. The second compares the raw side
with the anonymized side: each raw file must have its anonymized copy and
its own ECG pseudonym, and each patient (patient folder, or patient ID
without `--patients-dir-name`) its own patient pseudonym.

### `ecgdatakit anonymize daemon`

Keeps `SOURCE` anonymized until stopped (Ctrl+C): a full pass at start,
then live detection of changes and a full pass every `--interval` seconds
(see [Keeping a folder anonymized](#keeping-a-folder-anonymized)).

```text
ecgdatakit anonymize daemon SOURCE
    [--datasets] [--patients-dir-name NAME] [--anonymize-patient-folders]
    [--out-dir NAME] [--catalog NAME] [--threads N]
    [--interval SECONDS] [--settle SECONDS] [--no-watch]
    [--host ADDRESS] [--port PORT] [-v]
```

| Argument | Description |
|----------|-------------|
| `SOURCE` | Folder to keep anonymized (a single file is refused). |

| Option | Default | Description |
|--------|---------|-------------|
| `--datasets` | | If set, each sub-folder of `SOURCE` is a separate dataset, and new sub-folders are picked up. If not set, `SOURCE` is one dataset. |
| `--patients-dir-name NAME` | none | If set, only files under the folder with this name (the same in every dataset) are read, and each of its sub-folders is one patient. If not set, every file is read and patients are told apart by their patient ID. |
| `--anonymize-patient-folders` | | With `--patients-dir-name` only. If set, each patient folder is named after the patient pseudonym in the output, and the patient folder name is replaced by the pseudonym wherever it appears in the files and their names. If not set, folder names are copied as they are and only the identity fields of each format are used. |
| `--out-dir NAME` | `ANONYMIZED` | Output folder created in each dataset. |
| `--catalog NAME` | `anonymization_catalog.csv` | Catalog file created in each dataset. |
| `--threads N` | 8 | Number of files processed in parallel. |
| `--interval SECONDS` | 21600 (6 h) | Time between the end of a full pass and the start of the next one. Without live detection (`--no-watch`, no `watchdog`, network mount), it is the delay before a new file is anonymized: choose a shorter one, for example 600. |
| `--settle SECONDS` | 60 | A file is read only once it has not changed for this long, so a file still being copied is not read. |
| `--no-watch` | | If set, changes are not detected live and only the full passes find them: use it on a network mount. If not set, changes are detected as they happen (needs `watchdog`). |
| `--host ADDRESS` | `127.0.0.1` | Address the shell connects to. |
| `--port PORT` | 8765 | Port the shell connects to. |
| `-v`, `--verbose` | | The daemon always prints one line per file anonymized, changed or at risk. |

The control port has no password: keep it on `127.0.0.1`, or publish it
only on the local address of the host when the daemon runs in Docker.

### `ecgdatakit anonymize shell`

Connects to a running daemon. Without a command, it opens an interactive
prompt; with one, it runs it and exits (`ecgdatakit anonymize shell status`).

```text
ecgdatakit anonymize shell [COMMAND ...] [--host ADDRESS] [--port PORT]
```

| Argument | Description |
|----------|-------------|
| `COMMAND ...` | Optional. One command to run, then exit. If not given, an interactive prompt opens. |

| Option | Default | Description |
|--------|---------|-------------|
| `--host ADDRESS` | `127.0.0.1` | Address of the daemon (its `--host`). |
| `--port PORT` | 8765 | Port of the daemon (its `--port`). |

| Command | Description |
|---------|-------------|
| `status` | What the daemon is doing, start and end of the last pass, next pass, results per dataset (including files waiting because they changed less than `--settle` seconds ago). |
| `datasets` | Datasets of the source with the number of files per status. |
| `catalog [TEXT] [--dataset NAME] [--status S] [--risk] [--limit N] [--all-columns]` | Catalog rows. `TEXT` filters on any column (a name, a path, a pseudonym). `--dataset` picks the dataset when the source has several, `--status` keeps one status, `--risk` only rows at risk, `--limit` sets the number of rows shown (50), `--all-columns` shows every column. |
| `scan [DATASET]` | Start a pass now, of one dataset or of the whole source. |
| `quit` | Leave the shell. The daemon keeps running. |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Done, no file failed and every check passed. |
| 1 | Done, but some files failed or a check found a problem (see the summary and the catalog). |
| 2 | Wrong arguments, or a path or dependency is missing. |
| 3 | The dataset is being anonymized by another process. |
