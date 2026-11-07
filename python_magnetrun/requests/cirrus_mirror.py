"""
Mirror the raw Cirrus run-logs and XML configurations from srv-data-install.

The web server publishes, per power-supply feed (``A1`` … ``A4``), Apache
directory listings of

- ``/cirrus/<feed>/YYYY-MM-DD_cirrus_out.log``: daily copies of the raw log.
  Each copy holds everything since the controller's last boot, so a later
  file of the same boot contains the earlier ones;
- ``/cirrus/<feed>/xml/*.xml``: the CIRRUS XML configurations.

Both are served without login.  This module copies them, unchanged, into
``<root>/cirrus/<feed>/`` and ``<root>/cirrus/<feed>/xml/``:

- a missing file is downloaded;
- a log dated before yesterday that is already present is skipped without
  a request (finished days do not change); yesterday's and today's logs are
  always downloaded again;
- an XML already present is never overwritten (a size different from the
  listing is only reported);
- the undated live file ``cirrus_out.log`` is not mirrored.

Files are written to ``<name>.part`` and renamed when complete; their mtime
is set to the server's ``Last-Modified``.
"""

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests  # type: ignore[import-untyped]

from ..data_dirs import PUPITRE_RUNLOG_DIR
from ..log_utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "https://srv-data-install.lncmi.cnrs.fr/"
FEEDS = ("A1", "A2", "A3", "A4")
TIMEZONE = ZoneInfo("Europe/Paris")
TIMEOUT = 60  # [s] per request

_ROW_RE = re.compile(
    r'<td><a href="([^"?/][^"]*)">[^<]*</a></td>'
    r'<td align="right">([^<]*)</td>'
    r'<td align="right">([^<]*)</td>'
)
_LOG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_cirrus_out\.log$")
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMG]?)$")
_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}


@dataclass(frozen=True)
class ListedFile:
    """One entry of an Apache directory listing."""

    name: str
    """File name (``"xml/"`` for a sub-directory)."""
    modified: str
    """Last-modified time as shown, e.g. ``"2026-09-19 22:16"`` (server local time)."""
    size: int | None
    """Size [bytes], rounded as shown in the listing; ``None`` for directories."""


@dataclass
class FeedSummary:
    """Counts for one feed."""

    logs_downloaded: int = 0
    logs_skipped: int = 0
    xml_downloaded: int = 0
    xml_skipped: int = 0
    failed: int = 0
    bytes: int = 0


def parse_size(text: str) -> int | None:
    """Convert an Apache listing size to bytes.

    Parameters
    ----------
    text : str
        Size column, e.g. ``"271K"``, ``"9.5K"``, ``"1.2M"``, ``"812"`` or
        ``"  - "`` (directory).

    Returns
    -------
    int or None
        Size [bytes], rounded like the listing; ``None`` if not a size.
    """
    m = _SIZE_RE.match(text.strip())
    if not m:
        return None
    return int(float(m.group(1)) * _UNITS[m.group(2)])


def parse_listing(html: str) -> list[ListedFile]:
    """Parse an Apache ``Index of`` page.

    Parameters
    ----------
    html : str
        Page content.

    Returns
    -------
    list of ListedFile
        Entries in page order; the parent-directory row and the column
        headers are not included.
    """
    return [
        ListedFile(name, modified.strip(), parse_size(size))
        for name, modified, size in _ROW_RE.findall(html)
    ]


def log_day(name: str) -> date | None:
    """Return the day of a daily log file name.

    Parameters
    ----------
    name : str
        File name, e.g. ``"2026-09-19_cirrus_out.log"``.

    Returns
    -------
    datetime.date or None
        The day, or ``None`` if *name* is not a daily log.
    """
    m = _LOG_RE.match(name)
    return date.fromisoformat(m.group(1)) if m else None


class _Client:
    """Send requests one at a time, at least *delay* seconds apart."""

    def __init__(self, session, delay: float, sleep=time.sleep, clock=time.monotonic):
        self.session, self.delay, self.sleep, self.clock = session, delay, sleep, clock
        self._last: float | None = None

    def get(self, url: str, stream: bool = False):
        """GET *url* after waiting for the delay since the previous request."""
        if self._last is not None and self.delay > 0:
            wait = self.delay - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        try:
            return self.session.get(url, timeout=TIMEOUT, stream=stream)
        finally:
            self._last = self.clock()


def _fetch_listing(client: _Client, url: str) -> list[ListedFile]:
    """Return the entries of the listing at *url*; raise RuntimeError on HTTP error."""
    r = client.get(url)
    try:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return parse_listing(r.text)
    finally:
        r.close()


def _download(client: _Client, url: str, path: Path) -> int:
    """Download *url* to *path* through ``<path>.part``; return the size [bytes]."""
    part = path.with_name(path.name + ".part")
    r = client.get(url, stream=True)
    try:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        path.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with part.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                size += len(chunk)
        last_modified = r.headers.get("Last-Modified")
        if last_modified:
            t = parsedate_to_datetime(last_modified).timestamp()
            os.utime(part, (t, t))
        part.replace(path)
        return size
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        r.close()


def _size_differs(local: int, listed: int | None) -> bool:
    """Tell whether *local* [bytes] is outside the rounding of the listed size."""
    if listed is None:
        return False
    return abs(local - listed) > 0.06 * listed + 1024


def mirror_feed(
    session,
    server: str,
    feed: str,
    root: str | Path,
    start: date | None = None,
    end: date | None = None,
    xml: bool = True,
    today: date | None = None,
    delay: float = 0.5,
    dry_run: bool = False,
) -> FeedSummary:
    """Mirror the daily logs and XMLs of one Cirrus feed.

    Parameters
    ----------
    session : requests.Session
        HTTP session (no login needed).
    server : str
        Server root, e.g. ``"https://srv-data-install.lncmi.cnrs.fr/"``.
    feed : str
        Power-supply feed, e.g. ``"A1"``.
    root : str or :class:`~pathlib.Path`
        Mirror root; files go to ``<root>/cirrus/<feed>/`` and
        ``<root>/cirrus/<feed>/xml/``.
    start, end : datetime.date, optional
        First and last log day to consider (inclusive); XMLs are not
        filtered.
    xml : bool, optional
        Also mirror the XML configurations.
    today : datetime.date, optional
        Current day (default: today in Europe/Paris); logs of this day and
        the day before are always downloaded again.
    delay : float, optional
        Minimum time between two requests [s].
    dry_run : bool, optional
        Only list what would be downloaded; nothing is written.

    Returns
    -------
    FeedSummary
        Counts of downloaded, skipped and failed files, and bytes written.
        In a dry run, ``*_downloaded`` counts the files that would be
        downloaded.
    """
    today = today or datetime.now(TIMEZONE).date()
    feed_url = f"{server}cirrus/{feed}/"
    feed_dir = Path(root) / "cirrus" / feed
    client = _Client(session, delay)
    summary = FeedSummary()

    jobs: list[tuple[str, Path, str]] = []  # (url, path, kind)
    try:
        listing = _fetch_listing(client, feed_url)
    except (RuntimeError, requests.RequestException) as e:
        logger.error(f"{feed}: cannot list {feed_url}: {e}")
        summary.failed += 1
        listing = []
    for entry in listing:
        day = log_day(entry.name)
        if day is None or (start and day < start) or (end and day > end):
            continue
        path = feed_dir / entry.name
        if path.exists() and day < today - timedelta(days=1):
            summary.logs_skipped += 1
            continue
        jobs.append((feed_url + entry.name, path, "log"))

    if xml:
        xml_url = feed_url + "xml/"
        try:
            xml_listing = _fetch_listing(client, xml_url)
        except (RuntimeError, requests.RequestException) as e:
            logger.error(f"{feed}: cannot list {xml_url}: {e}")
            summary.failed += 1
            xml_listing = []
        for entry in xml_listing:
            if not entry.name.endswith(".xml"):
                continue
            path = feed_dir / "xml" / entry.name
            if path.exists():
                summary.xml_skipped += 1
                local = path.stat().st_size
                if _size_differs(local, entry.size):
                    logger.warning(
                        f"{feed}: xml/{entry.name}: local size {local} B, server shows "
                        f"{entry.size} B; local file kept"
                    )
                continue
            jobs.append((xml_url + entry.name, path, "xml"))

    for url, path, kind in jobs:
        if dry_run:
            logger.info(f"would download {url}")
        else:
            try:
                summary.bytes += _download(client, url, path)
            except (RuntimeError, OSError, requests.RequestException) as e:
                logger.error(f"{feed}: {url}: {e}")
                summary.failed += 1
                continue
            logger.debug(f"downloaded {url} -> {path}")
        if kind == "log":
            summary.logs_downloaded += 1
        else:
            summary.xml_downloaded += 1
    return summary


def _build_parser(prog: str = "magnetrun cirrus-mirror") -> argparse.ArgumentParser:
    """Return the argument parser."""
    p = argparse.ArgumentParser(
        prog=prog,
        description="Mirror the raw Cirrus run-logs and XML configurations",
        add_help=False,
    )
    p.add_argument(
        "--root",
        help="mirror root (default: PUPITRE_RUNLOG_DIR / data_dirs.json['pupitre_runlog'])",
    )
    p.add_argument("--feeds", nargs="+", default=list(FEEDS), help="feeds (default: A1 A2 A3 A4)")
    p.add_argument("--start", type=date.fromisoformat, help="first log day (YYYY-MM-DD)")
    p.add_argument("--end", type=date.fromisoformat, help="last log day (YYYY-MM-DD)")
    p.add_argument("--no-xml", action="store_true", help="do not mirror the XML configurations")
    p.add_argument("--server", default=DEFAULT_SERVER, help="server root")
    p.add_argument("--delay", type=float, default=0.5, help="pause between requests [s]")
    p.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return p


def run(args: argparse.Namespace) -> int:
    """Run the mirror for the parsed command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from :func:`_build_parser`.

    Returns
    -------
    int
        Exit code: 0 if every request succeeded, 1 otherwise.

    Raises
    ------
    SystemExit
        If no mirror root is given or configured.
    """
    setup_logging(level=getattr(logging, args.log_level))
    root = args.root or PUPITRE_RUNLOG_DIR
    if not root:
        raise SystemExit(
            "no mirror root: give --root or set MAGNETRUN_PUPITRE_RUNLOG_DIR "
            "(data_dirs.json['pupitre_runlog'])"
        )
    failed = 0
    with requests.Session() as session:
        for feed in args.feeds:
            t0 = time.monotonic()
            s = mirror_feed(
                session,
                args.server,
                feed,
                root,
                start=args.start,
                end=args.end,
                xml=not args.no_xml,
                delay=args.delay,
                dry_run=args.dry_run,
            )
            failed += s.failed
            logger.info(
                f"{feed}: logs {s.logs_downloaded} downloaded, {s.logs_skipped} skipped; "
                f"xml {s.xml_downloaded} downloaded, {s.xml_skipped} skipped; "
                f"{s.failed} failed; {s.bytes / 1e6:.1f} MB in {time.monotonic() - t0:.0f} s"
                + (" (dry run)" if args.dry_run else "")
            )
    return 1 if failed else 0


def register(sub: "argparse._SubParsersAction") -> None:
    """Register the ``cirrus-mirror`` subcommand on *sub*.

    Parameters
    ----------
    sub : argparse._SubParsersAction
        Sub-parsers of the ``magnetrun`` entry point.
    """
    p = sub.add_parser(
        "cirrus-mirror",
        parents=[_build_parser()],
        help="mirror the raw Cirrus run-logs and XML configurations",
    )
    p.set_defaults(_handler=run)
