"""Anonymization daemon.

Runs a full pass at start and then every *interval* seconds. Between passes,
file-system events (when ``watchdog`` is installed and the folders are on a
local disk) queue the folders that changed; they are processed once no
event arrived for *settle* seconds, so files still being copied are not
read. A control server on a local TCP port answers the commands of
``ecgdatakit anonymize shell`` (one JSON object per line).
"""

from __future__ import annotations

import csv
import json
import logging
import socketserver
import threading
import time
from datetime import datetime
from pathlib import Path

from ecgdatakit.anonymize._catalog import ACTIVE
from ecgdatakit.anonymize._engine import Anonymizer, DatasetLocked, Report

log = logging.getLogger("ecgdatakit.anonymize")

DEFAULT_PORT = 8765


def _stamp(t: float | None) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds") if t else ""


class Daemon:
    def __init__(self, anonymizer: Anonymizer, interval: float = 6 * 3600, settle: float = 60.0,
                 host: str = "127.0.0.1", port: int = DEFAULT_PORT, watch: bool = True) -> None:
        self.anonymizer = anonymizer
        self.interval = interval
        self.settle = settle
        self.host, self.port = host, port
        self.watch = watch
        self._wake = threading.Condition()
        self._pending: dict[Path, float] = {}
        self._requested: list[str | None] = []
        self._stop = False
        self.running = ""
        self.last_full_start: float | None = None
        self.last_full_end: float | None = None
        self.next_full: float | None = None
        self.last_reports: list[Report] = []
        self.watching = False

    # -- main loop ----------------------------------------------------------------

    def serve_forever(self) -> None:
        server = _ControlServer((self.host, self.port), _ControlHandler, self)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log.info("control server on %s:%d", self.host, self.port)
        observer = self._start_watcher() if self.watch else None
        try:
            self._full_pass()
            while not self._stop:
                with self._wake:
                    self._wake.wait(timeout=self._timeout())
                if self._requested:
                    requested, self._requested = self._requested, []
                    self._full_pass([s for s in requested if s] if None not in requested else None)
                self._process_events()
                if self.next_full is not None and time.time() >= self.next_full:
                    self._full_pass()
        except KeyboardInterrupt:
            log.info("stopping")
        finally:
            if observer is not None:
                observer.stop()
            server.shutdown()

    def stop(self) -> None:
        self._stop = True
        with self._wake:
            self._wake.notify_all()

    def trigger(self, dataset: str | None = None) -> None:
        self._requested.append(dataset)
        with self._wake:
            self._wake.notify_all()

    def _timeout(self) -> float:
        now = time.time()
        waits = [max(0.0, (self.next_full or now) - now)]
        if self._pending:
            waits.append(max(0.5, min(self._pending.values()) + self.settle - now))
        return min(waits)

    def _full_pass(self, datasets: list[str] | None = None) -> None:
        self.running = "full pass" if not datasets else f"pass of {', '.join(datasets)}"
        self.last_full_start = time.time()
        log.info("%s started", self.running)
        try:
            self.last_reports = self.anonymizer.run(datasets=datasets)
        except (DatasetLocked, FileNotFoundError, OSError) as e:
            log.error("pass stopped: %s", e)
        finally:
            self.last_full_end = time.time()
            self.next_full = self.last_full_end + self.interval
            self.running = ""
        for r in self.last_reports:
            log.info("%s: %d found, %d anonymized, %d copied, %d changed, %d deleted, %d failed, "
                     "%d with risks", r.dataset, r.found, r.anonymized, r.copied, r.changed,
                     r.deleted, r.failed, r.risks)

    # -- file-system events -----------------------------------------------------------

    def _start_watcher(self):
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            log.warning("watchdog is not installed: changes are found by the scheduled passes "
                        "only (pip install 'ecgdatakit[anonymize]')")
            return None
        daemon = self

        class _Events(FileSystemEventHandler):
            def on_any_event(self, event):
                for attr in ("src_path", "dest_path"):
                    path = getattr(event, attr, None)
                    if path:
                        daemon._queue(Path(path))

        observer = Observer()
        observer.schedule(_Events(), str(self.anonymizer.source), recursive=True)
        observer.daemon = True
        observer.start()
        self.watching = True
        log.info("watching %s for changes", self.anonymizer.source)
        return observer

    def _queue(self, path: Path) -> None:
        a = self.anonymizer
        try:
            dataset = a.dataset_of(path)
            parts = path.relative_to(dataset).parts
        except ValueError:
            return
        # Files of the dataset only: not its output folder, catalog or lock,
        # and only inside a raw folder when one is configured
        if len(parts) < 2 or parts[0] == a.out_dir or any(p.startswith(".") for p in parts):
            return
        if a.patients_dir_name is not None and a.patients_dir_name not in parts:
            return
        with self._wake:
            self._pending[path] = time.time()
            self._wake.notify_all()

    def _process_events(self) -> None:
        now = time.time()
        with self._wake:
            ready = [p for p, t in self._pending.items() if now - t >= self.settle]
            for p in ready:
                del self._pending[p]
        if not ready:
            return
        folders = {p if p.is_dir() else p.parent for p in ready}
        # A folder already covered by another one in the list is dropped
        folders = {f for f in folders if not any(o in f.parents for o in folders)}
        self.running = f"{len(folders)} changed folder(s)"
        try:
            for folder in sorted(folders):
                if folder.exists():
                    self.anonymizer.run(paths=[folder], recursive=True)
        except (DatasetLocked, ValueError, OSError) as e:
            log.warning("changed folders not processed now: %s", e)
            with self._wake:
                for p in ready:
                    self._pending.setdefault(p, time.time())
        finally:
            self.running = ""

    # -- commands -----------------------------------------------------------------

    def command(self, request: dict) -> dict:
        cmd = request.get("cmd")
        a = self.anonymizer
        if cmd == "status":
            return {
                "source": str(a.source), "running": self.running or "idle",
                "last_pass_start": _stamp(self.last_full_start),
                "last_pass_end": _stamp(self.last_full_end),
                "next_pass": _stamp(self.next_full), "interval_seconds": self.interval,
                "watching": self.watching, "queued_changes": len(self._pending),
                "last_reports": [r.__dict__ for r in self.last_reports],
            }
        if cmd == "datasets":
            out = []
            for dataset in a.datasets():
                counts: dict[str, int] = {}
                catalog = a.catalog_path(dataset)
                if catalog.exists():
                    with open(catalog, newline="", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            counts[row["status"]] = counts.get(row["status"], 0) + 1
                            if row["risk"] == "TRUE" and row["status"] in ACTIVE:
                                counts["risk"] = counts.get("risk", 0) + 1
                out.append({"dataset": dataset.name, "catalog": catalog.exists(), **counts})
            return {"datasets": out}
        if cmd == "catalog":
            name = request.get("dataset")
            if name:
                dataset = a.source / str(name) if a.multi else a.source
            elif not a.multi:
                dataset = a.source
            else:
                return {"error": "give a dataset name (see the datasets command)"}
            path = a.catalog_path(dataset)
            if not path.exists():
                return {"error": f"no catalog for {dataset.name!r}"}
            text = str(request.get("filter") or "").lower()
            status = request.get("status")
            risk = request.get("risk")
            limit = int(request.get("limit") or 50)
            rows = []
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if status and row["status"] != status:
                        continue
                    if risk and row["risk"] != "TRUE":
                        continue
                    if text and text not in " ".join(row.values()).lower():
                        continue
                    rows.append(row)
            return {"total": len(rows), "rows": rows[:limit]}
        if cmd == "scan":
            dataset = request.get("dataset") or None
            if dataset and not a.multi:
                dataset = None  # a single dataset: the whole source
            if dataset and not (a.source / dataset).is_dir():
                return {"error": f"no dataset {dataset!r} in {a.source}"}
            self.trigger(dataset)
            return {"ok": f"pass of {dataset or 'every dataset'} requested"}
        return {"error": f"unknown command {cmd!r}"}


class _ControlServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, daemon: Daemon) -> None:
        self.daemon = daemon
        super().__init__(address, handler)


class _ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for line in self.rfile:
            try:
                reply = self.server.daemon.command(json.loads(line))
            except Exception as e:  # noqa: BLE001 - reported to the client
                reply = {"error": f"{type(e).__name__}: {e}"}
            self.wfile.write((json.dumps(reply, default=str) + "\n").encode())
            self.wfile.flush()


def request(command: dict, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
            timeout: float = 30.0) -> dict:
    """Send one command to a running daemon and return its reply."""
    import socket

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall((json.dumps(command) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)
