"""Anonymization of ECG files under a source path.

The source path is one dataset, or with ``datasets=True`` each of its
sub-folders is one. In a dataset, files (only those under folders named
``raw_dir`` when it is given) are detected with the ecgdatakit parsers,
anonymized by the format handler, checked (see ``_verify``), then moved
into the dataset's output folder. The dataset's CSV catalog records every
file. A file already in the catalog with the same size and modification
time is not read again; a changed file is anonymized again and a deleted
file loses its anonymized copy.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ecgdatakit import __version__
from ecgdatakit.anonymize._catalog import ACTIVE, SEPARATOR, Catalog
from ecgdatakit.anonymize._codes import new_code
from ecgdatakit.anonymize._filenames import new_file_name, split_name
from ecgdatakit.anonymize._identity import Identity, Replacer, norm
from ecgdatakit.anonymize._verify import VerificationError, verify
from ecgdatakit.anonymize.formats import HANDLERS, Codes, Handler, Spliced
from ecgdatakit.anonymize.formats._base import is_data
from ecgdatakit.anonymize.formats.wfdb import WFDBHandler
from ecgdatakit.parsing.parser import FileParser

log = logging.getLogger("ecgdatakit.anonymize")

_SNIFF = 4096
_SKIP_DIRS = {"#recycle", "@eaDir", "#snapshot", ".snapshot"}
_STAGING = ".anonymize-staging"
_LOCK = ".anonymize.lock"
_SAVE_EVERY = 200


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _hidden(name: str) -> bool:
    return name.startswith(".") or name in _SKIP_DIRS


@dataclass
class Entry:
    """A raw file (or WFDB record) found in a dataset."""

    path: Path
    rel: str
    patient_folder: str
    companions: list[Path] = field(default_factory=list)
    size: int = 0
    mtime_ns: int = 0
    handler: Handler | None = None


@dataclass
class Report:
    dataset: str
    found: int = 0
    anonymized: int = 0
    copied: int = 0
    changed: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: int = 0
    risks: int = 0
    skipped_unsupported: int = 0
    waiting: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.failed


class DatasetLocked(RuntimeError):
    """Another process is anonymizing this dataset."""


class _DatasetLock:
    def __init__(self, dataset: Path, stale_seconds: float) -> None:
        self.path = dataset / _LOCK
        self.stale = stale_seconds

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                age = time.time() - self.path.stat().st_mtime
                if age < self.stale:
                    raise DatasetLocked(f"{self.path.parent.name} is being anonymized "
                                      f"by another process ({self.path})") from None
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as f:
                f.write(f"{socket.gethostname()} {os.getpid()} {_now()}\n")
            return self
        raise DatasetLocked(f"cannot lock {self.path}")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


class Anonymizer:
    """Anonymize the ECG files under a source path.

    Parameters
    ----------
    source : path
        Folder to anonymize.
    datasets : bool
        ``False`` (default): *source* is one dataset. ``True``: every
        sub-folder of *source* is a separate dataset, with its own output
        folder and catalog.
    raw_dir : str, optional
        When given, only files under folders with this name are read and the
        first folder under it is the patient folder. When ``None``, every
        file of the dataset is read (except the output folder) and the first
        folder of the dataset is the patient folder.
    out_dir, catalog_name : str
        Output folder and CSV catalog created in each dataset.

    Raw files are only read, never modified.
    """

    def __init__(self, source: str | Path, datasets: bool = False, raw_dir: str | None = None,
                 out_dir: str = "ANONYMIZED", catalog_name: str = "anonymization_catalog.csv",
                 threads: int = 8, settle_seconds: float = 0.0,
                 lock_stale_seconds: float = 12 * 3600, dry_run: bool = False) -> None:
        self.source = Path(source).resolve()
        if not self.source.is_dir():
            raise FileNotFoundError(f"Source folder not found: {self.source}")
        self.multi = datasets
        self.raw_dir = raw_dir or None
        self.out_dir = out_dir
        self.catalog_name = catalog_name
        self.threads = max(1, int(threads))
        self.settle = settle_seconds
        self.lock_stale = lock_stale_seconds
        self.dry_run = dry_run
        self._parsers = [p for p in FileParser().parsers if p.__name__ in HANDLERS]

    # -- datasets ---------------------------------------------------------------

    def datasets(self) -> list[Path]:
        """The source itself, or its sub-folders with ``datasets=True``."""
        if not self.multi:
            return [self.source]
        return sorted(p for p in self.source.iterdir() if p.is_dir() and not _hidden(p.name))

    def dataset_of(self, path: Path) -> Path:
        path = Path(path).resolve()
        try:
            rel = path.relative_to(self.source)
        except ValueError:
            raise ValueError(f"{path} is not inside the source folder {self.source}") from None
        if not self.multi:
            return self.source
        if not rel.parts:
            raise ValueError("Give a path inside a dataset, or no path to run every dataset")
        return self.source / rel.parts[0]

    def catalog_path(self, dataset: Path) -> Path:
        return Path(dataset) / self.catalog_name

    # -- runs ------------------------------------------------------------------

    def run(self, paths: list[str | Path] | None = None, recursive: bool = True,
            datasets: list[str] | None = None) -> list[Report]:
        """Anonymize every dataset, the named *datasets*, or only *paths*."""
        if paths:
            by_dataset: dict[Path, list[Path]] = {}
            for p in paths:
                p = Path(p).resolve()
                if not p.exists():
                    raise FileNotFoundError(f"Not found: {p}")
                by_dataset.setdefault(self.dataset_of(p), []).append(p)
            return [self.run_dataset(d, scope=None if ps == [d] else ps, recursive=recursive)
                    for d, ps in by_dataset.items()]
        targets = self.datasets()
        if datasets:
            if not self.multi:
                raise ValueError("Dataset names need datasets=True (--datasets)")
            names = set(datasets)
            missing = names - {d.name for d in targets}
            if missing:
                raise FileNotFoundError(f"Datasets not found in {self.source}: {sorted(missing)}")
            targets = [d for d in targets if d.name in names]
        return [self.run_dataset(d) for d in targets]

    def run_dataset(self, dataset: Path, scope: list[Path] | None = None,
                    recursive: bool = True) -> Report:
        dataset = Path(dataset).resolve()
        report = Report(dataset=dataset.name)
        if self.dry_run:
            return self._run_locked(dataset, scope, recursive, report)
        with _DatasetLock(dataset, self.lock_stale):
            return self._run_locked(dataset, scope, recursive, report)

    # -- scan ------------------------------------------------------------------

    def _raw_roots(self, dataset: Path, scope: list[Path] | None, recursive: bool):
        """Yield (folder to list, recursive, root of the patient folders)."""
        out_root = dataset / self.out_dir
        if self.raw_dir is None:
            # Every file of the dataset is read
            for target in scope or [dataset]:
                if target == out_root or out_root in target.parents:
                    raise ValueError(f"{target} is inside the anonymized output folder")
                yield target, recursive if scope else True, dataset
            return
        if scope is None:
            for root, dirs, _ in os.walk(dataset):
                root_path = Path(root)
                if root_path == dataset:
                    dirs[:] = [d for d in dirs if d != self.out_dir]
                dirs[:] = sorted(d for d in dirs if not _hidden(d))
                if root_path.name == self.raw_dir and root_path != dataset:
                    dirs[:] = []
                    yield root_path, True, root_path
            return
        for target in scope:
            if target == out_root or out_root in target.parents:
                raise ValueError(f"{target} is inside the anonymized output folder")
            raw_root = next((p for p in [target, *target.parents]
                             if p.name == self.raw_dir and dataset in p.parents), None)
            if raw_root is not None:
                yield target, recursive, raw_root
                continue
            if target.is_file():
                raise ValueError(f"{target} is not inside a {self.raw_dir} folder")
            if not recursive:
                raise ValueError(f"{target} is not inside a {self.raw_dir} folder "
                                 "(use recursion to look for one below it)")
            for root, dirs, _ in os.walk(target):
                root_path = Path(root)
                dirs[:] = sorted(d for d in dirs if not _hidden(d)
                                 and not (root_path == dataset and d == self.out_dir))
                if root_path.name == self.raw_dir:
                    dirs[:] = []
                    yield root_path, True, root_path

    def _list(self, folder: Path, recursive: bool,
              skip: Path | None = None) -> list[tuple[Path, list[os.DirEntry]]]:
        """Files of *folder* grouped by directory (threaded over sub-folders).

        *skip* (the dataset's output folder) is never entered.
        """
        if folder.is_file():
            return [(folder.parent, [e for e in os.scandir(folder.parent) if e.is_file()
                                     and (e.name == folder.name or e.name.startswith(Path(folder.name).stem + "."))])]

        def walk(top: Path) -> list[tuple[Path, list[os.DirEntry]]]:
            out = []
            stack = [top]
            while stack:
                d = stack.pop()
                files, subdirs = [], []
                try:
                    with os.scandir(d) as it:
                        for e in it:
                            if e.is_dir(follow_symlinks=False):
                                if (recursive and not _hidden(e.name) and e.name != _STAGING
                                        and Path(e.path) != skip):
                                    subdirs.append(Path(e.path))
                            elif e.is_file() and not e.name.startswith("."):
                                files.append(e)
                except OSError as err:
                    log.warning("cannot list %s: %s", d, err)
                    continue
                out.append((d, files))
                stack.extend(sorted(subdirs, reverse=True))
            return out

        if not recursive:
            return walk(folder)
        subs = []
        files: list[os.DirEntry] = []
        try:
            with os.scandir(folder) as it:
                for e in it:
                    if (e.is_dir(follow_symlinks=False) and not _hidden(e.name)
                            and Path(e.path) != skip):
                        subs.append(Path(e.path))
                    elif e.is_file() and not e.name.startswith("."):
                        files.append(e)
        except OSError as err:
            log.warning("cannot list %s: %s", folder, err)
            return []
        result = [(folder, files)]
        with ThreadPoolExecutor(self.threads) as pool:
            for part in pool.map(walk, sorted(subs)):
                result.extend(part)
        return result

    def _detect(self, path: Path) -> Handler | None:
        try:
            with open(path, "rb") as f:
                header = f.read(_SNIFF)
        except OSError:
            return None
        for parser in self._parsers:
            try:
                if parser.can_parse(path, header):
                    return HANDLERS[parser.__name__]
            except Exception:  # noqa: BLE001 - a sniffer failing means "not this format"
                continue
        return None

    def _scan(self, dataset: Path, scope, recursive, catalog: Catalog,
              report: Report) -> tuple[dict[str, Entry], list[tuple[Path, bool]]]:
        entries: dict[str, Entry] = {}
        scanned: list[tuple[Path, bool]] = []
        now = time.time()
        for folder, rec, raw_root in self._raw_roots(dataset, scope, recursive):
            scanned.append((folder, rec, folder.is_file()))
            for directory, files in self._list(folder, rec, skip=dataset / self.out_dir):
                names = {e.name for e in files}
                for e in files:
                    path = Path(e.path)
                    rel = path.relative_to(dataset).as_posix()
                    if path.parent == dataset and path.name in (self.catalog_name, _LOCK):
                        continue
                    st = e.stat()
                    if self.settle and now - st.st_mtime < self.settle:
                        report.waiting += 1
                        continue
                    old = catalog.get(rel)
                    if (old and old["status"] in ACTIVE and old["raw_size"] == str(st.st_size)
                            and old["raw_mtime_ns"] == str(st.st_mtime_ns)
                            and not old["companions"]):
                        entries[rel] = Entry(path, rel, old["patient_folder"], [],
                                             st.st_size, st.st_mtime_ns, None)
                        continue
                    handler = self._detect(path)
                    if handler is None:
                        report.skipped_unsupported += 1
                        continue
                    if isinstance(handler, WFDBHandler) and path.suffix.lower() != ".hea":
                        if path.stem + ".hea" in names:
                            continue  # part of a record, handled with its header
                        report.skipped_unsupported += 1
                        continue
                    companions = handler.companions(path)
                    size = st.st_size + sum(c.stat().st_size for c in companions)
                    mtime = max([st.st_mtime_ns] + [c.stat().st_mtime_ns for c in companions])
                    entries[rel] = Entry(path, rel, self._patient_folder(dataset, raw_root, path),
                                         companions, size, mtime, handler)
        return entries, scanned

    @staticmethod
    def _patient_folder(dataset: Path, raw_root: Path, path: Path) -> str:
        rel = path.relative_to(raw_root)
        if len(rel.parts) > 1:
            return (raw_root / rel.parts[0]).relative_to(dataset).as_posix()
        return ""

    # -- processing --------------------------------------------------------------

    def _run_locked(self, dataset: Path, scope, recursive, report: Report) -> Report:
        catalog = Catalog(self.catalog_path(dataset))
        entries, scanned = self._scan(dataset, scope, recursive, catalog, report)
        report.found = len(entries)
        state = _State(catalog)
        work = []
        for rel, entry in entries.items():
            old = catalog.get(rel)
            if entry.handler is None and old is not None:
                # Unchanged on disk; re-done only when its output disappeared
                if all((dataset / p).exists() for p in _outputs(old)):
                    report.unchanged += 1
                    continue
                entry.handler = self._detect(entry.path)
                if entry.handler is None:
                    report.skipped_unsupported += 1
                    continue
                entry.companions = entry.handler.companions(entry.path)
                entry.patient_folder = old["patient_folder"] or entry.patient_folder
            work.append(entry)

        done = 0
        with ThreadPoolExecutor(self.threads) as pool:
            futures = {pool.submit(self._process, dataset, e, state): e for e in work}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    row = future.result()
                except Exception as e:  # noqa: BLE001 - one bad file never stops the dataset
                    log.exception("unexpected error on %s", entry.rel)
                    row = self._failed_row(dataset, entry, catalog.get(entry.rel),
                                           f"{type(e).__name__}: {e}")
                status = row["status"]
                if status == "failed":
                    report.failed += 1
                    log.warning("failed: %s (%s)", entry.rel, row["message"])
                elif status == "unchanged":
                    report.unchanged += 1
                    row["status"] = catalog.get(entry.rel)["status"]
                else:
                    setattr(report, status, getattr(report, status) + 1)
                if row.get("risk") == "TRUE":
                    report.risks += 1
                if not self.dry_run:
                    catalog.put(row)
                    done += 1
                    if done % _SAVE_EVERY == 0:
                        catalog.save()

        if not self.dry_run:
            report.deleted = self._deleted(dataset, entries, scanned, catalog)
            catalog.save()
            report.errors = self.check(dataset, catalog, entries if scope is None else None)
        for error in report.errors:
            log.error("%s: %s", dataset.name, error)
        return report

    def _process(self, dataset: Path, entry: Entry, state: _State) -> dict[str, str]:
        catalog = state.catalog
        old = catalog.get(entry.rel)
        handler = entry.handler
        files = [entry.path, *entry.companions]
        sha = _sha256(files)
        now = _now()
        if (old and old["status"] in ACTIVE and old["raw_sha256"] == sha
                and all((dataset / p).exists() for p in _outputs(old))):
            row = dict(old, raw_size=str(entry.size), raw_mtime_ns=str(entry.mtime_ns))
            row["status"] = "unchanged"
            return row

        identity = handler.identity(entry.path)
        key = entry.patient_folder or _identity_key(identity) or f"file:{entry.rel}"
        with state.lock:
            patient_code = state.patient_codes.get(key)
            if patient_code is None:
                patient_code = new_code(state.used)
                state.patient_codes[key] = patient_code
            ecg_ids = SEPARATOR.join(identity.ecg_ids)
            if old and old["ecg_code"] and old["original_ecg_id"] == ecg_ids:
                ecg_code = old["ecg_code"]
            else:
                ecg_code = new_code(state.used)

        folder_names = [Path(entry.patient_folder).name] if entry.patient_folder else []
        risks: list[str] = []
        if isinstance(handler, WFDBHandler) and handler.is_multi_segment(entry.path):
            new_name = entry.path.name
            if new_file_name(entry.path.name, identity, patient_code, ecg_code, folder_names)[0] != new_name:
                risks.append("multi-segment WFDB record name kept (it may hold identity)")
        else:
            new_name, name_risks = new_file_name(entry.path.name, identity, patient_code,
                                                 ecg_code, folder_names)
            risks += name_risks

        rel_dir = Path(entry.rel).parent
        out_dir = dataset / self.out_dir / rel_dir
        with state.lock:
            new_name = state.reserve((self.out_dir / rel_dir / new_name).as_posix(), entry.rel,
                                     new_name, ecg_code)
        out_path = out_dir / new_name

        replacer = Replacer(identity, patient_code, ecg_code)
        rewrite = handler.rewrite(entry.path, out_path, identity, Codes(patient_code, ecg_code), replacer)
        risks += rewrite.notes

        row = {
            "raw_path": entry.rel,
            "patient_folder": key,
            "format": handler.parser,
            "patient_code": patient_code,
            "ecg_code": ecg_code,
            "original_patient_id": SEPARATOR.join(identity.patient_ids),
            "original_last_name": SEPARATOR.join(identity.last_names),
            "original_first_name": SEPARATOR.join(identity.first_names),
            "original_ecg_id": ecg_ids,
            "anonymized_ecg_id": SEPARATOR.join(rewrite.ecg_written),
            "original_file_name": entry.path.name,
            "anonymized_file_name": new_name,
            "folder_anonymized": "FALSE",
            "companions": SEPARATOR.join(
                (Path(self.out_dir) / rel_dir / p.name).as_posix()
                for p in rewrite.outputs if p != out_path),
            "raw_size": str(entry.size),
            "raw_mtime_ns": str(entry.mtime_ns),
            "raw_sha256": sha,
            "detected_at": (old or {}).get("detected_at") or now,
            "tool_version": __version__,
            "anonymized_path": (Path(self.out_dir) / rel_dir / new_name).as_posix(),
        }
        if self.dry_run:
            row.update(status="anonymized" if not identity.is_empty else "copied",
                       risk="TRUE" if risks else "FALSE", risk_reason=SEPARATOR.join(risks))
            return row

        staging = out_dir / _STAGING / ecg_code
        try:
            staged = self._stage(rewrite.outputs, staging)
            verify(entry.path, staged[out_path], replacer, patient_code)
            values = handler.free_text(_head(rewrite.outputs[out_path]))
            left = sum(1 for v in values if replacer.replace_value(v, is_data(v)) != v)
            if left:
                risks.append(f"{left} field value(s) of the anonymized file still hold identity text")
            if old:
                for p in _outputs(old):
                    target = dataset / p
                    if target.exists() and target not in staged:
                        target.unlink()
            for final, temp in staged.items():
                os.replace(temp, final)
            row["anonymized_sha256"] = _sha256([out_path])
        except VerificationError as e:
            return self._failed_row(dataset, entry, old, f"verification failed: {e}", row)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            _remove_empty(staging.parent)

        if old and old["status"] in ACTIVE | {"deleted", "failed"}:
            status = "changed"
            row["last_change_at"] = now
        else:
            status = "anonymized" if not identity.is_empty else "copied"
            row["last_change_at"] = (old or {}).get("last_change_at", "")
        row.update(status=status, anonymized_at=now, risk="TRUE" if risks else "FALSE",
                   risk_reason=SEPARATOR.join(dict.fromkeys(risks)), message="")
        return row

    @staticmethod
    def _stage(outputs: dict, staging: Path) -> dict[Path, Path]:
        staging.mkdir(parents=True, exist_ok=True)
        staged: dict[Path, Path] = {}
        for final, content in outputs.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            temp = staging / final.name
            with open(temp, "wb") as f:
                if isinstance(content, Spliced):
                    f.write(content.head)
                    with open(content.source, "rb") as src:
                        src.seek(content.offset)
                        shutil.copyfileobj(src, f, 1024 * 1024)
                else:
                    f.write(content)
            staged[final] = temp
        return staged

    def _failed_row(self, dataset: Path, entry: Entry, old, message: str,
                    row: dict | None = None) -> dict:
        row = dict(row or old or {"raw_path": entry.rel, "patient_folder": entry.patient_folder})
        row.update(status="failed", message=message, raw_size=str(entry.size),
                   raw_mtime_ns=str(entry.mtime_ns), tool_version=__version__,
                   original_file_name=entry.path.name, anonymized_path="", anonymized_file_name="",
                   companions="")
        row.setdefault("detected_at", _now())
        if old:
            # A previous anonymized copy of an older version is removed
            for p in _outputs(old):
                (dataset / p).unlink(missing_ok=True)
        return row

    def _deleted(self, dataset: Path, entries: dict[str, Entry], scanned, catalog: Catalog) -> int:
        count = 0
        now = _now()
        for row in list(catalog.rows.values()):
            if row["status"] == "deleted" or row["raw_path"] in entries:
                continue
            raw = dataset / row["raw_path"]
            if not _in_scope(raw, scanned) or raw.exists():
                continue
            for p in _outputs(row):
                (dataset / p).unlink(missing_ok=True)
            row.update(status="deleted", last_change_at=now, anonymized_path="",
                       anonymized_sha256="", companions="")
            catalog.put(row)
            count += 1
        return count

    # -- consistency -------------------------------------------------------------

    def check(self, dataset: Path, catalog: Catalog,
              entries: dict[str, Entry] | None = None) -> list[str]:
        """Collision and count checks between raw and anonymized files."""
        errors: list[str] = []
        rows = catalog.active()
        by_folder: dict[str, set[str]] = {}
        by_code: dict[str, set[str]] = {}
        for r in rows:
            by_folder.setdefault(r["patient_folder"], set()).add(r["patient_code"])
            by_code.setdefault(r["patient_code"], set()).add(r["patient_folder"])
        for folder, codes in by_folder.items():
            if len(codes) > 1:
                errors.append(f"patient folder {folder!r} has {len(codes)} pseudonyms")
        for code, folders in by_code.items():
            if len(folders) > 1:
                errors.append(f"pseudonym {code} is shared by {len(folders)} patient folders")
        if len(by_folder) != len(by_code):
            errors.append(f"{len(by_folder)} patients but {len(by_code)} pseudonyms")
        ecg = [r["ecg_code"] for r in rows]
        if len(ecg) != len(set(ecg)):
            errors.append("an ECG pseudonym is used by more than one file")
        outputs = [r["anonymized_path"] for r in rows]
        if len(outputs) != len(set(outputs)):
            errors.append("two raw files map to the same anonymized file")
        missing = [p for r in rows for p in _outputs(r) if not (dataset / p).exists()]
        if missing:
            errors.append(f"{len(missing)} anonymized file(s) missing on disk")
        if entries is not None:
            raw_count = len(entries)
            active = {r["raw_path"] for r in rows}
            failed = sum(1 for r in catalog.rows.values() if r["status"] == "failed"
                         and r["raw_path"] in entries)
            if raw_count != len(active & set(entries)) + failed:
                errors.append(f"{raw_count} raw files but {len(active & set(entries))} "
                              f"anonymized and {failed} failed")
            per_raw: dict[str, int] = {}
            for e in entries.values():
                per_raw[e.patient_folder or ""] = per_raw.get(e.patient_folder or "", 0) + 1
            per_anon: dict[str, int] = {}
            for r in rows:
                if r["raw_path"] in entries and (dataset / r["anonymized_path"]).exists():
                    folder = entries[r["raw_path"]].patient_folder or ""
                    per_anon[folder] = per_anon.get(folder, 0) + 1
            for folder, n in per_raw.items():
                failed_here = sum(1 for r in catalog.rows.values() if r["status"] == "failed"
                                  and r["raw_path"] in entries
                                  and (entries[r["raw_path"]].patient_folder or "") == folder)
                if per_anon.get(folder, 0) + failed_here != n:
                    errors.append(f"patient folder {folder or '(none)'!r}: {n} raw files, "
                                  f"{per_anon.get(folder, 0)} anonymized")
        return errors


class _State:
    """Codes and output names shared by the worker threads of one dataset."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.lock = threading.Lock()
        self.used = catalog.used_codes()
        self.patient_codes = catalog.patient_codes()
        self.taken = {r["anonymized_path"]: r["raw_path"] for r in catalog.active()}

    def reserve(self, out_rel: str, raw_rel: str, name: str, ecg_code: str) -> str:
        owner = self.taken.get(out_rel)
        if owner is not None and owner != raw_rel:
            stem, ext = split_name(name)
            name = f"{stem}_{ecg_code}{ext}"
            out_rel = str(Path(out_rel).parent / name)
        self.taken[out_rel] = raw_rel
        return name


def _identity_key(identity: Identity) -> str:
    if identity.patient_ids:
        return "id:" + norm(identity.patient_ids[0])
    if identity.last_names or identity.first_names:
        return "name:" + norm(" ".join(identity.last_names + identity.first_names))
    return ""


def _outputs(row: dict) -> list[str]:
    paths = [row.get("anonymized_path", "")]
    paths += [p for p in (row.get("companions") or "").split(SEPARATOR) if p]
    return [p for p in paths if p]


def _in_scope(path: Path, scanned: list[tuple[Path, bool, bool]]) -> bool:
    """True when *path* was in a folder (or is the file) listed by this run."""
    for target, recursive, is_file in scanned:
        if is_file:
            if path == target:
                return True
        elif path.parent == target or recursive and target in path.parents:
            return True
    return False


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _head(content) -> bytes:
    return content.head if isinstance(content, Spliced) else content


def _remove_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass
