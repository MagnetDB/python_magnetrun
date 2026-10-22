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
_KEYS = ["Time", "Message", "Type"]

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
    return df.astype(str)


def _parse_day(dates: pd.Series) -> pd.Series:
    """Parse ``Date`` values (ISO or dd/mm/yyyy, optional time) to dates; NaT if unparseable."""
    iso = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce", exact=False)
    dmy = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce", exact=False)
    return iso.fillna(dmy).dt.date


def split_by_day(df: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Split log rows by the day of their ``Date`` column.

    ``Date`` may be ISO (``2026-07-27 15:12:29.575500``, ``2026-7-27``) or
    ``27/07/2026``, with or without a time part; rows with an unparseable
    date are dropped.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Rows returned by :func:`load_cirrus_log` (possibly several logs
        concatenated).

    Returns
    -------
    dict of datetime.date to :class:`~pandas.DataFrame`
        Rows of each day, in chronological day order; row order within a day
        is preserved.
    """
    days = _parse_day(df["Date"])
    return {
        day: group
        for day, group in df.groupby(days, sort=True, dropna=True)
    }


def _is_continuation(df: pd.DataFrame) -> pd.Series:
    """Mask of continuation rows (empty ``Time``), e.g. the KI/KP lines of a parameter dump."""
    return df["Time"] == ""


def find_duplicates(df: pd.DataFrame, tolerance_us: int = 1000) -> pd.Series:
    """Flag rows repeated in this DataFrame, e.g. the same log downloaded twice.

    The server recomputes ``Date`` on each download (a few µs jitter), so it
    is not used to compare rows: rows with a ``Time`` are compared on
    ``Time``, ``Message`` and ``Type``.  Continuation rows (empty ``Time``)
    are compared on ``Message`` and on ``Date`` within ``tolerance_us``.  The
    first occurrence of a row is not flagged.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Rows returned by :func:`load_cirrus_log` (possibly several logs
        concatenated).
    tolerance_us : int, optional
        Maximum ``Date`` difference [µs] for two continuation rows to be
        considered the same.

    Returns
    -------
    :class:`~pandas.Series`
        Boolean mask aligned on ``df.index``, ``True`` for duplicate rows.
    """
    cont = _is_continuation(df)
    dup = df[_KEYS].duplicated() & ~cont
    t = pd.to_datetime(df.loc[cont, "Date"], format="ISO8601", errors="coerce").sort_values()
    gaps = t.groupby(df.loc[cont, "Message"]).diff()
    dup.loc[gaps.index] = gaps <= pd.Timedelta(microseconds=tolerance_us)
    return dup


def save_day(
    df: pd.DataFrame, feed: str, day: date, datadir: str | Path, tolerance_us: int = 1000
) -> int:
    """Merge one day of log rows into ``<datadir>/cirrus/<feed>/<day>_cirrus_out.csv``.

    An existing file is never overwritten: its rows are kept and only rows it
    does not already contain are appended (compared as in
    :func:`find_duplicates`).  The result is sorted by ``Date`` (stable).

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        Rows of ``day`` (see :func:`split_by_day`).
    feed : str
        Power-supply feed, e.g. ``"A1"``.
    day : datetime.date
        Day the rows belong to.
    datadir : str or :class:`~pathlib.Path`
        Root output directory.
    tolerance_us : int, optional
        Maximum ``Date`` difference [µs] for a continuation row to match one
        already in the file.

    Returns
    -------
    int
        Number of rows added to the file.
    """
    path = Path(datadir) / "cirrus" / feed / f"{day.isoformat()}_cirrus_out.csv"
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        cont, old_cont = _is_continuation(df), _is_continuation(old)
        present = set(old.loc[~old_cont, _KEYS].itertuples(index=False, name=None))
        known = pd.Series(
            [row in present for row in df[_KEYS].itertuples(index=False, name=None)],
            index=df.index,
            dtype=bool,
        )
        # continuation rows: nearest row of the file with the same Message
        left = (
            df.loc[cont, ["Message"]]
            .assign(_t=pd.to_datetime(df.loc[cont, "Date"], format="ISO8601", errors="coerce"))
            .dropna(subset=["_t"])
            .rename_axis("_row")
            .reset_index()
            .sort_values("_t")
        )
        right = (
            old.loc[old_cont, ["Message"]]
            .assign(
                _t=pd.to_datetime(old.loc[old_cont, "Date"], format="ISO8601", errors="coerce"),
                _hit=True,
            )
            .dropna(subset=["_t"])
            .sort_values("_t")
        )
        m = pd.merge_asof(
            left, right, on="_t", by="Message", direction="nearest",
            tolerance=pd.Timedelta(microseconds=tolerance_us),
        )
        known.loc[m.loc[m["_hit"].notna(), "_row"]] = True
        new = df[~known]
        if new.empty:
            return 0
        out = pd.concat([old, new], ignore_index=True)
    else:
        new = out = df
        path.parent.mkdir(parents=True, exist_ok=True)
    out = out.sort_values("Date", kind="stable")
    out.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    return len(new)


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
    p.add_argument("--end", type=date.fromisoformat, help="last day (YYYY-MM-DD, default: today)")
    p.add_argument("--user", help="specify user")
    p.add_argument(
        "--server",
        help="specify server",
        default="https://srv-data-install.lncmi.cnrs.fr/",
    )
    p.add_argument("--datadir", help="specify data dir", type=str, default=".")
    p.add_argument(
        "--drop-duplicates",
        help="remove rows repeated within or across the loaded logs",
        action="store_true",
    )
    p.add_argument(
        "--tolerance-us",
        type=int,
        default=1000,
        help="max Date difference [µs] when matching continuation rows (empty Time)",
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
        files += daily_files(args.feed, args.start, args.end or date.today())
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

    logs: dict[str, list[pd.DataFrame]] = {}
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
                continue
            if df.empty:
                logger.warning(f"{normalize_file(file)}: no data")
                continue
            feed = normalize_file(file).split("/", 1)[0]
            logs.setdefault(feed, []).append(df)

    for feed, dfs in logs.items():
        df = pd.concat(dfs, ignore_index=True)
        dups = find_duplicates(df, args.tolerance_us)
        if dups.any():
            logger.warning(f"{feed}: {dups.sum()} duplicate rows within or across the loaded logs")
            logger.debug(f"duplicate rows:\n{df[dups]}")
            if args.drop_duplicates:
                df = df[~dups]
                logger.info(f"{feed}: removed {dups.sum()} duplicate rows")
        for day, rows in split_by_day(df).items():
            added = save_day(rows, feed, day, args.datadir, args.tolerance_us)
            logger.info(
                f"{feed}/{day}: {added} rows added ({len(rows) - added} already present)"
            )


if __name__ == "__main__":
    main()
