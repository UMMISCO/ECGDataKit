"""Interactive shell connected to a running anonymization daemon."""

from __future__ import annotations

import cmd
import shlex

from ecgdatakit.anonymize._daemon import DEFAULT_PORT, request

_BRIEF = ["status", "risk", "raw_path", "anonymized_path", "patient_code", "ecg_code",
          "last_change_at", "message"]


def _table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "(nothing)"
    widths = {c: min(60, max(len(c), *(len(str(r.get(c, ""))) for r in rows))) for c in columns}
    line = "  ".join(c.ljust(widths[c]) for c in columns)
    out = [line, "  ".join("-" * widths[c] for c in columns)]
    for r in rows:
        out.append("  ".join(str(r.get(c, ""))[: widths[c]].ljust(widths[c]) for c in columns))
    return "\n".join(out)


class Shell(cmd.Cmd):
    intro = "ECGDataKit anonymization shell. Type help or ? to list commands."
    prompt = "anonymize> "

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        super().__init__()
        self.host, self.port = host, port

    def _ask(self, command: dict) -> dict | None:
        try:
            reply = request(command, self.host, self.port)
        except OSError as e:
            print(f"No daemon on {self.host}:{self.port} ({e}). "
                  "Start one with: ecgdatakit anonymize daemon BASE")
            return None
        if "error" in reply:
            print(reply["error"])
            return None
        return reply

    def do_status(self, arg: str) -> None:
        """status: what the daemon is doing, last and next pass."""
        reply = self._ask({"cmd": "status"})
        if reply is None:
            return
        for key in ("source", "running", "last_pass_start", "last_pass_end", "next_pass",
                    "interval_seconds", "watching", "queued_changes"):
            print(f"{key:18} {reply[key]}")
        reports = reply["last_reports"]
        if reports:
            print()
            print(_table(reports, ["dataset", "found", "anonymized", "copied", "changed",
                                   "unchanged", "deleted", "failed", "risks"]))

    def do_datasets(self, arg: str) -> None:
        """datasets: datasets of the source with their catalog counts."""
        reply = self._ask({"cmd": "datasets"})
        if reply is not None:
            print(_table(reply["datasets"], ["dataset", "anonymized", "copied", "changed",
                                            "deleted", "failed", "risk"]))

    def do_catalog(self, arg: str) -> None:
        """catalog [TEXT] [--dataset NAME] [--status S] [--risk] [--limit N] [--all-columns]

        Rows of a catalog. TEXT filters on any column (a name, a path, a
        code). --dataset picks the dataset when the source has several,
        --status keeps one status (anonymized, copied, changed, deleted,
        failed), --risk only rows marked as risk."""
        try:
            args = shlex.split(arg)
        except ValueError as e:
            print(e)
            return
        command = {"cmd": "catalog", "limit": 50}
        columns = _BRIEF
        while args:
            a = args.pop(0)
            if a == "--dataset" and args:
                command["dataset"] = args.pop(0)
            elif a == "--status" and args:
                command["status"] = args.pop(0)
            elif a == "--limit" and args:
                command["limit"] = int(args.pop(0))
            elif a == "--risk":
                command["risk"] = True
            elif a == "--all-columns":
                columns = None
            else:
                command["filter"] = a
        reply = self._ask(command)
        if reply is None:
            return
        rows = reply["rows"]
        if columns is None:
            for row in rows:
                print("\n".join(f"{k:22} {v}" for k, v in row.items()))
                print()
        else:
            print(_table(rows, columns))
        print(f"{len(rows)} of {reply['total']} row(s)")

    def do_scan(self, arg: str) -> None:
        """scan [DATASET]: start a pass now, of one dataset or of every dataset."""
        reply = self._ask({"cmd": "scan", "dataset": arg.strip() or None})
        if reply is not None:
            print(reply["ok"])

    def do_quit(self, arg: str) -> bool:
        """quit: leave the shell (the daemon keeps running)."""
        return True

    do_exit = do_quit
    do_EOF = do_quit

    def emptyline(self) -> None:
        pass
