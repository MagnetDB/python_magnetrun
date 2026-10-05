#! /usr/bin/python3

"""
Download Cirrus run-log files from the Control/Monitoring site.

The "Voir Propre" page ``cirrus.php?file=A1/2026-07-27_cirrus_out.log`` is a
DataTables shell whose rows are fetched as JSON from
``cirrus.php?filedata=A1/2026-07-27_cirrus_out.log``.  This module calls that
endpoint directly and saves the rows as CSV (``Date,Time,Message,Type``)
under ``<datadir>/cirrus/<feed>/<date>_cirrus_out.csv``.
"""

import argparse
import csv
import getpass
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests  # type: ignore[import-untyped]

from ..log_utils import setup_logging
from .connect import createSession

logger = logging.getLogger(__name__)

PAGES = "site/sba/pages/"
COLUMNS = ["Date", "Time", "Message", "Type"]

_FILE_RE = re.compile(r"[?&](?:file|filedata)=([^&]+)")


def normalize_file(value: str) -> str:
    """Return the ``<feed>/<name>`` part of a Cirrus log reference.

    Parameters
    ----------
    value : str
        Either ``"A1/2026-07-27_cirrus_out.log"`` or a full
        ``cirrus.php?file=...`` / ``cirrus.php?filedata=...`` URL.

    Returns
    -------
    str
        The file reference, e.g. ``"A1/2026-07-27_cirrus_out.log"``.
    """
    m = _FILE_RE.search(value)
    return m.group(1) if m else value


def daily_files(feed: str, start: date, end: date) -> list[str]:
    """Build the daily Cirrus log references for a date range.

    Parameters
    ----------
    feed : str
        Power-supply feed, e.g. ``"A1"``.
    start, end : datetime.date
        First and last day (inclusive).

    Returns
    -------
    list of str
        References like ``"A1/2026-07-27_cirrus_out.log"``.
    """
    days = (end - start).days + 1
    return [
        f"{feed}/{(start + timedelta(days=i)).isoformat()}_cirrus_out.log"
        for i in range(days)
    ]


def load_cirrus_log(session: requests.Session, base_url: str, file: str) -> pd.DataFrame:
    """Fetch one Cirrus log from the ``cirrus.php?filedata=`` JSON endpoint.

    Parameters
    ----------
    session : requests.Session
        Authenticated session.
    base_url : str
        Server root, e.g. ``"https://srv-data-install.lncmi.cnrs.fr/"``.
    file : str
        Log reference (see :func:`normalize_file`).

    Returns
    -------
    :class:`~pandas.DataFrame`
        Columns ``Date``, ``Time``, ``Message``, ``Type``; empty if the server
        returned no rows.

    Raises
    ------
    RuntimeError
        On HTTP error, redirect to the login page, or a non-JSON response.
    """
    file = normalize_file(file)
    url = f"{base_url}{PAGES}cirrus.php?filedata={file}"
    r = session.get(url, verify=True)
    logger.debug(f"cirrus filedata: {r.url}, status={r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"download failed: HTTP {r.status_code} from {url}")
    if r.url.endswith("login.php"):
        raise RuntimeError("redirected to login page: session not authenticated")
    try:
        rows = r.json()["data"]
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"unexpected (non JSON) response from {url}") from e

    df = pd.DataFrame(rows, columns=["date", "time", "message", "type"])
    df.columns = COLUMNS
    return df


def find_duplicates(df: pd.DataFrame, seen: set[tuple]) -> pd.Series:
    """Flag rows already seen in this DataFrame or in previously checked ones.

    Rows are compared on all columns (``Date``, ``Time``, ``Message``,
    ``Type``); the first occurrence of a row is not flagged.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Rows returned by :func:`load_cirrus_log`.
    seen : set of tuple
        Rows from previously checked DataFrames; updated in place with the
        rows of ``df``.

    Returns
    -------
    :class:`~pandas.Series`
        Boolean mask aligned on ``df.index``, ``True`` for duplicate rows.
    """
    flags = []
    for row in df.itertuples(index=False, name=None):
        flags.append(row in seen)
        seen.add(row)
    return pd.Series(flags, index=df.index, dtype=bool)


def save_cirrus_log(
    df: pd.DataFrame, file: str, datadir: str | Path, overwrite: bool = False
) -> Path | None:
    """Save a Cirrus log as CSV under ``<datadir>/cirrus/<feed>/``.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Rows returned by :func:`load_cirrus_log`.
    file : str
        Log reference (see :func:`normalize_file`).
    datadir : str or :class:`~pathlib.Path`
        Root output directory.
    overwrite : bool, optional
        Replace an existing file (default: skip it).

    Returns
    -------
    :class:`~pathlib.Path` or None
        Path written, or ``None`` if the file already existed and
        ``overwrite`` is false.
    """
    feed, name = normalize_file(file).split("/", 1)
    path = Path(datadir) / "cirrus" / feed / Path(name).with_suffix(".csv")
    if path.exists() and not overwrite:
        logger.info(f"skip existing file: {path}")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"saved {len(df)} rows to {path}")
    return path


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    p = argparse.ArgumentParser(
        prog="python -m python_magnetrun.requests.cirrus_logs",
        description="Download Cirrus run-logs as CSV files",
    )
    p.add_argument(
        "files",
        nargs="*",
        help="log references (A1/2026-07-27_cirrus_out.log) or cirrus.php URLs",
    )
    p.add_argument("--feed", help="cirrus feed (A1, A2, …) for a date range")
    p.add_argument("--start", type=date.fromisoformat, help="first day (YYYY-MM-DD)")
    p.add_argument("--end", type=date.fromisoformat, help="last day (YYYY-MM-DD)")
    p.add_argument("--user", help="specify user")
    p.add_argument(
        "--server",
        help="specify server",
        default="https://srv-data-install.lncmi.cnrs.fr/",
    )
    p.add_argument("--datadir", help="specify data dir", type=str, default=".")
    p.add_argument("--overwrite", help="replace existing files", action="store_true")
    p.add_argument(
        "--drop-duplicates",
        help="remove rows already seen in this file or a previously loaded one",
        action="store_true",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p


def main() -> None:
    """Command-line entry point."""
    p = _build_parser()
    args = p.parse_args()
    setup_logging(level=getattr(logging, args.log_level))
    logger.setLevel(args.log_level)

    files = list(args.files)
    if args.feed or args.start or args.end:
        if not (args.feed and args.start):
            p.error("--feed and --start are required for a date range")
        files += daily_files(args.feed, args.start, args.end or args.start)
    if not files:
        p.error("give log files or --feed/--start/--end")

    if sys.stdin.isatty():
        password = getpass.getpass("Using getpass: ")
    else:
        password = sys.stdin.readline().rstrip()

    base_url = args.server
    url_logging = base_url + PAGES + "login.php"
    url_status = base_url + PAGES + "Etat.php"
    payload = {"email": args.user, "password": password}

    n_errors = 0
    seen: set[tuple] = set()
    with requests.Session() as s:
        createSession(s, url_logging, payload)
        r = s.get(url=url_status, verify=True)
        if r.url == url_logging:
            logger.error("check connection failed: Wrong credentials")
            sys.exit(1)

        for file in files:
            try:
                df = load_cirrus_log(s, base_url, file)
            except RuntimeError as e:
                logger.error(f"{file}: {e}")
                n_errors += 1
                continue
            if df.empty:
                logger.warning(f"{normalize_file(file)}: no data")
                continue
            dups = find_duplicates(df, seen)
            if dups.any():
                logger.warning(
                    f"{normalize_file(file)}: {dups.sum()} duplicate rows "
                    "(already seen in this file or a previous one)"
                )
                logger.debug(f"duplicate rows:\n{df[dups]}")
                if args.drop_duplicates:
                    df = df[~dups]
                    logger.info(f"{normalize_file(file)}: removed {dups.sum()} duplicate rows")
            save_cirrus_log(df, file, args.datadir, overwrite=args.overwrite)

    if n_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
