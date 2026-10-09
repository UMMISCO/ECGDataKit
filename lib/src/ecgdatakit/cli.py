"""``ecgdatakit`` command line."""

from __future__ import annotations

import argparse
import logging
import sys

from ecgdatakit import __version__


def _anonymizer_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", help="folder to anonymize")
    p.add_argument("--datasets", action="store_true",
                   help="each sub-folder of SOURCE is a separate dataset, with its own "
                        "output folder and catalog (default: SOURCE is one dataset)")
    p.add_argument("--raw-dir", default=None, metavar="NAME",
                   help="only read files under folders with this name (the first folder "
                        "under it is the patient folder); default: every file")
    p.add_argument("--out-dir", default="ANONYMIZED",
                   help="output folder created in each dataset (default: ANONYMIZED)")
    p.add_argument("--catalog", default="anonymization_catalog.csv",
                   help="CSV catalog file name in each dataset")
    p.add_argument("--threads", type=int, default=8, help="parallel workers (default: 8)")
    p.add_argument("-v", "--verbose", action="store_true", help="log every file")


def _anonymizer(args, **extra):
    from ecgdatakit.anonymize import Anonymizer

    return Anonymizer(args.source, datasets=args.datasets, raw_dir=args.raw_dir,
                      out_dir=args.out_dir, catalog_name=args.catalog, threads=args.threads,
                      **extra)


def _print_reports(reports) -> bool:
    ok = True
    for r in reports:
        print(f"{r.dataset}: {r.found} file(s), {r.anonymized} anonymized, {r.copied} copied, "
              f"{r.changed} changed, {r.unchanged} unchanged, {r.deleted} deleted, "
              f"{r.failed} failed, {r.risks} with risks, "
              f"{r.skipped_unsupported} not ECG files (skipped)")
        for error in r.errors:
            print(f"  ERROR {error}")
        ok &= r.ok
    return ok


def _run(args) -> int:
    anonymizer = _anonymizer(args, dry_run=args.dry_run)
    reports = anonymizer.run(paths=args.paths or None, recursive=not args.no_recursive,
                             datasets=args.dataset or None)
    return 0 if _print_reports(reports) else 1


def _daemon(args) -> int:
    from ecgdatakit.anonymize._daemon import Daemon

    anonymizer = _anonymizer(args, settle_seconds=args.settle)
    Daemon(anonymizer, interval=args.interval, settle=args.settle, host=args.host,
           port=args.port, watch=not args.no_watch).serve_forever()
    return 0


def _shell(args) -> int:
    from ecgdatakit.anonymize._shell import Shell

    shell = Shell(args.host, args.port)
    if args.command:
        shell.onecmd(" ".join(args.command))
    else:
        shell.cmdloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    from ecgdatakit.anonymize._daemon import DEFAULT_PORT

    parser = argparse.ArgumentParser(prog="ecgdatakit", description="ECGDataKit tools")
    parser.add_argument("--version", action="version", version=f"ecgdatakit {__version__}")
    tools = parser.add_subparsers(dest="tool", required=True)

    anon = tools.add_parser("anonymize", help="anonymize ECG files",
                            description="Copy the ECG files of SOURCE into SOURCE/ANONYMIZED "
                                        "with the identity values replaced by pseudonyms, and "
                                        "a CSV catalog linking raw and anonymized files.")
    actions = anon.add_subparsers(dest="action", required=True)

    run = actions.add_parser("run", help="anonymize SOURCE, some of its datasets, or some paths")
    _anonymizer_options(run)
    run.add_argument("paths", nargs="*",
                     help="files or folders inside SOURCE to process (default: all of it)")
    run.add_argument("--dataset", action="append", metavar="NAME",
                     help="with --datasets: only this dataset (repeatable)")
    run.add_argument("--no-recursive", action="store_true",
                     help="do not look into sub-folders of the given folders")
    run.add_argument("--dry-run", action="store_true",
                     help="report what would be done, write nothing")
    run.set_defaults(func=_run)

    daemon = actions.add_parser("daemon", help="keep SOURCE anonymized as files arrive")
    _anonymizer_options(daemon)
    daemon.add_argument("--interval", type=float, default=6 * 3600,
                        help="seconds between full passes (default: 21600)")
    daemon.add_argument("--settle", type=float, default=60.0,
                        help="seconds a file must stay unchanged before it is read (default: 60)")
    daemon.add_argument("--no-watch", action="store_true",
                        help="no file-system events, scheduled passes only")
    daemon.add_argument("--host", default="127.0.0.1", help="control server address")
    daemon.add_argument("--port", type=int, default=DEFAULT_PORT, help="control server port")
    daemon.set_defaults(func=_daemon)

    shell = actions.add_parser("shell", help="talk to a running daemon")
    shell.add_argument("command", nargs=argparse.REMAINDER,
                       help="run one command and exit (e.g. 'status', 'datasets')")
    shell.add_argument("--host", default="127.0.0.1")
    shell.add_argument("--port", type=int, default=DEFAULT_PORT)
    shell.set_defaults(func=_shell)

    args = parser.parse_args(argv)
    level = logging.INFO if getattr(args, "verbose", False) or args.action == "daemon" else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        from ecgdatakit.anonymize import DatasetLocked

        if isinstance(e, DatasetLocked):
            print(f"error: {e}", file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
