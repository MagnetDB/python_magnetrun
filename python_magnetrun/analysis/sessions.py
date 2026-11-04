"""
Session records from the SUPERVISION experiment log (EXPERIENCES_LOG).

Each EXPERIENCES_LOG row is one coil (``M9i``, ``M9e``, ...) used by one
proposal (``UserCode``) between ``HStart`` and ``HStop``. This module turns
those rows into sessions, then lists the files recorded during each session
window with :meth:`FileDiscovery.discover_window`:

1. :func:`load_sessions` cleans the rows of one housing and merges the rows
   whose windows overlap into one :class:`Session`.
2. :func:`discover_sessions` fills each session's files, anchor and data end.
3. :func:`find_unlogged` groups the Overview, Archive and pupitre files that
   no session claimed into unlogged sessions (runs missing from the log).
4. :func:`sessions_to_records` converts sessions to JSON-ready dicts.

All times are naive Europe/Paris wall-clock times, the convention of both
``HStart``/``HStop`` and the data filenames. The CSV ``timestamp`` column is
never used: it drifts from ``HStart`` by up to 7.5 h since 2020.

Usage::

    magnetrun sessions --log Data/EXPERIENCES_LOG.csv --housing M9 \\
        --from 2018-01-01 --output-dir /tmp/M9_sessions
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..utils.files import TIMESTAMP_FORMAT, extract_data
from .loaders import FileDiscovery, FileSet

logger = logging.getLogger("python_magnetrun.analysis.sessions")

OPERATOR_CODE = "EXPLOIT"
"""UserCode of the operator account (also logged as ``EXPLOIT.``)."""

SUPERSEDED_GAP = timedelta(minutes=10)
"""An unclosed row followed by a new row on the same coil within this gap is a false start."""

OPEN_SESSION_CAP = timedelta(hours=12)
"""Longest duration given to an unclosed row."""

UNLOGGED_GAP = timedelta(minutes=30)
"""Unclaimed files closer than this are grouped into one unlogged session."""

_HLOG_FORMAT = "%m/%d/%y  %H:%M:%S.%f"


@dataclass
class Session:
    """One session: a time window on a housing, and the files recorded during it.

    Attributes
    ----------
    housing : str
        Housing identifier (e.g. ``"M9"``).
    start : datetime
        Session start, naive Europe/Paris wall-clock time.
    stop : datetime
        Session stop, same convention.
    stop_capped : bool
        True when *stop* was inferred for an unclosed row (no ``HStop``).
    logged : bool
        False for an unlogged session built from unclaimed files.
    users : list[str]
        Distinct UserCodes, ``EXPLOIT.`` normalized to ``EXPLOIT``.
    operator : bool
        True when the operator account is the only user.
    coils : list[dict]
        One dict per EXPERIENCES_LOG row: ``coil``, ``elec_group``, ``start``,
        ``stop``, ``minute_count``, ``paimant``.
    files : FileSet
        Files overlapping ``[start, stop]``.
    anchor : str
        ``"overview"``, ``"archive"``, ``"pupitre"`` or ``"none"``: the most
        complete file type found.
    data_end : datetime or None
        End of the last Overview, Archive or pupitre file found.
    """

    housing: str
    start: datetime
    stop: datetime
    stop_capped: bool = False
    logged: bool = True
    users: list[str] = field(default_factory=list)
    operator: bool = False
    coils: list[dict] = field(default_factory=list)
    files: FileSet = field(default_factory=FileSet)
    anchor: str = "none"
    data_end: datetime | None = None

    @property
    def id(self) -> str:
        """Identifier ``<housing>_<YYYYmmdd-HHMMSS>`` of the session start."""
        return f"{self.housing}_{self.start:%Y%m%d-%H%M%S}"


def parse_hlog_timestamp(value: str | None) -> datetime | None:
    """Parse an EXPERIENCES_LOG ``HStart``/``HStop`` value.

    The values use a 24-hour clock followed by an AM/PM marker that has no
    effect, e.g. ``"8/31/06  14:50:56.078 PM"``.

    Parameters
    ----------
    value : str or None
        Raw CSV value.

    Returns
    -------
    datetime or None
        Naive Europe/Paris wall-clock time; ``None`` for a blank value.
    """
    if value is None or not str(value).strip():
        return None
    return datetime.strptime(re.sub(r"\s+(AM|PM)$", "", str(value).strip()), _HLOG_FORMAT)


def _to_int(value: str) -> int | None:
    """Parse an optional integer CSV value."""
    return int(value) if value.strip() else None


def _to_float(value: str) -> float | None:
    """Parse an optional float CSV value."""
    return float(value) if value.strip() else None


def load_sessions(
    log_file: str | Path,
    housing: str,
    start: datetime,
    end: datetime | None = None,
) -> list[Session]:
    """Load the sessions of one housing from an EXPERIENCES_LOG CSV file.

    Exact duplicate rows are removed. An unclosed row (no ``HStop``)
    followed by a new row on the same coil within :data:`SUPERSEDED_GAP` is
    dropped as a false start; other unclosed rows stop at the next ``HStart``
    on the same coil, or after :data:`OPEN_SESSION_CAP`. Rows whose windows
    overlap are merged into one session.

    Parameters
    ----------
    log_file : str or Path
        EXPERIENCES_LOG CSV file.
    housing : str
        Housing identifier (e.g. ``"M9"``); matches ``Magnet`` without its
        ``e``/``i`` suffix.
    start : datetime
        Keep rows with ``HStart >= start`` (naive Europe/Paris).
    end : datetime, optional
        Keep rows with ``HStart < end``; no upper bound when omitted.

    Returns
    -------
    list[Session]
        Sessions sorted by start, with no files yet (see
        :func:`discover_sessions`).
    """
    df = pd.read_csv(log_file, dtype=str, keep_default_na=False).drop_duplicates()
    df = df[df["Magnet"].str.replace(r"[ei]$", "", regex=True) == housing]

    rows = []
    for r in df.itertuples(index=False):
        rows.append({
            "coil": r.Magnet,
            "elec_group": _to_int(r.ElecGroup),
            "user": r.UserCode.strip().rstrip("."),
            "start": parse_hlog_timestamp(r.HStart),
            "stop": parse_hlog_timestamp(r.HStop),
            "minute_count": _to_int(r.MinuteCount),
            "paimant": _to_float(r.Paimant),
            "capped": False,
        })
    rows.sort(key=lambda row: row["start"])

    # Unclosed rows, judged against the next row on the same coil.
    kept = []
    for coil in sorted({row["coil"] for row in rows}):
        coil_rows = [row for row in rows if row["coil"] == coil]
        for i, row in enumerate(coil_rows):
            if row["stop"] is None:
                nxt = next(
                    (n["start"] for n in coil_rows[i + 1:] if n["start"] > row["start"]),
                    None,
                )
                if nxt is not None and nxt - row["start"] < SUPERSEDED_GAP:
                    continue
                cap = row["start"] + OPEN_SESSION_CAP
                row["stop"] = min(nxt, cap) if nxt is not None else cap
                row["capped"] = True
            kept.append(row)
    kept = [
        row for row in kept
        if row["start"] >= start and (end is None or row["start"] < end)
    ]
    kept.sort(key=lambda row: row["start"])

    sessions: list[Session] = []
    group: list[dict] = []
    for row in kept + [None]:
        if group and (row is None or row["start"] >= max(g["stop"] for g in group)):
            last = max(group, key=lambda g: g["stop"])
            users = sorted({g["user"] for g in group})
            sessions.append(Session(
                housing=housing,
                start=group[0]["start"],
                stop=last["stop"],
                stop_capped=last["capped"],
                users=users,
                operator=users == [OPERATOR_CODE],
                coils=[
                    {k: g[k] for k in ("coil", "elec_group", "start", "stop", "minute_count", "paimant")}
                    for g in group
                ],
            ))
            group = []
        if row is not None:
            group.append(row)
    logger.info(f"{housing}: {len(sessions)} sessions from {len(kept)} log rows")
    return sessions


@functools.cache
def _file_range(path: str, housing: str) -> tuple[datetime, datetime] | None:
    """Return (start, end) of a data file, or None when it cannot be read."""
    try:
        s, e, _ = extract_data(path, housing)
    except (OSError, ValueError, RuntimeError, UnicodeDecodeError) as exc:
        logger.warning(f"{path}: {exc}")
        return None
    if not s or not e:
        return None
    return datetime.strptime(s, TIMESTAMP_FORMAT), datetime.strptime(e, TIMESTAMP_FORMAT)


def discover_sessions(sessions: list[Session], discovery: FileDiscovery) -> None:
    """Fill the files, anchor and data end of each session in place.

    Parameters
    ----------
    sessions : list[Session]
        Sessions to complete.
    discovery : FileDiscovery
        Configured with the pupitre/pigbrother (and hybrid) data directories.
    """
    for i, s in enumerate(sessions, start=1):
        s.files = discovery.discover_window(s.housing, s.start, s.stop)
        s.anchor = next(
            (k for k in ("overview", "archive", "pupitre") if getattr(s.files, k)),
            "none",
        )
        ranges = [
            _file_range(f, s.housing)
            for f in s.files.overview + s.files.archive + s.files.pupitre
        ]
        ends = [r[1] for r in ranges if r is not None]
        s.data_end = max(ends) if ends else None
        if i % 50 == 0:
            logger.info(f"{s.housing}: {i}/{len(sessions)} sessions discovered")


def find_unlogged(
    sessions: list[Session],
    discovery: FileDiscovery,
    housing: str,
    start: datetime,
    end: datetime,
) -> list[Session]:
    """Group the files that no session claimed into unlogged sessions.

    Overview, Archive and pupitre files dated within ``[start, end)``,
    longer than 30 s and not listed by any session are chained when the gap
    between them is at most :data:`UNLOGGED_GAP`; each chain becomes a
    session with ``logged=False``, whose files are then discovered like
    those of a logged session.

    Parameters
    ----------
    sessions : list[Session]
        Logged sessions, already passed through :func:`discover_sessions`.
    discovery : FileDiscovery
        Same discovery object as for the logged sessions.
    housing : str
        Housing identifier.
    start, end : datetime
        Period covered (naive Europe/Paris).

    Returns
    -------
    list[Session]
        Unlogged sessions sorted by start.
    """
    claimed = {
        f for s in sessions for f in s.files.overview + s.files.archive + s.files.pupitre
    }
    candidates = discovery.candidate_files(housing, start, end)
    ranges = []
    for kind in ("overview", "archive", "pupitre"):
        for f in candidates[kind]:
            if f in claimed:
                continue
            r = _file_range(f, housing)
            if r is None or not start <= r[0] < end:
                continue
            if (r[1] - r[0]).total_seconds() <= 30.0:
                continue
            ranges.append(r)
    ranges.sort()

    chains: list[list[datetime]] = []
    for fs, fe in ranges:
        if chains and fs - chains[-1][1] <= UNLOGGED_GAP:
            chains[-1][1] = max(chains[-1][1], fe)
        else:
            chains.append([fs, fe])
    unlogged = [Session(housing=housing, start=a, stop=b, logged=False) for a, b in chains]
    discover_sessions(unlogged, discovery)
    logger.info(f"{housing}: {len(unlogged)} unlogged sessions from {len(ranges)} unclaimed files")
    return unlogged


def _fmt(dt: datetime | None) -> str | None:
    """Format a datetime as TIMESTAMP_FORMAT, keeping None."""
    return dt.strftime(TIMESTAMP_FORMAT) if dt is not None else None


def sessions_to_records(sessions: list[Session]) -> list[dict]:
    """Convert sessions to JSON-ready dicts.

    Parameters
    ----------
    sessions : list[Session]
        Sessions to convert.

    Returns
    -------
    list[dict]
        One dict per session: ``id``, ``housing``, ``logged``, ``start``,
        ``stop``, ``stop_capped``, ``data_end``, ``anchor``, ``users``,
        ``operator``, ``coils``, then one list of paths per
        :class:`FileSet` field. Times use ``TIMESTAMP_FORMAT``.
    """
    records = []
    for s in sessions:
        records.append({
            "id": s.id,
            "housing": s.housing,
            "logged": s.logged,
            "start": _fmt(s.start),
            "stop": _fmt(s.stop),
            "stop_capped": s.stop_capped,
            "data_end": _fmt(s.data_end),
            "anchor": s.anchor,
            "users": s.users,
            "operator": s.operator,
            "coils": [
                {**c, "start": _fmt(c["start"]), "stop": _fmt(c["stop"])} for c in s.coils
            ],
            **s.files.to_dict(),
        })
    return records


# =============================================================================
# CLI
# =============================================================================


def _run(args: argparse.Namespace) -> int:
    """Dispatcher entry for ``magnetrun sessions``."""
    from ..log_utils import setup_logging

    setup_logging(level=args.log_level)
    start = datetime.fromisoformat(args.date_from)
    end = datetime.fromisoformat(args.date_to) if args.date_to else datetime.now()

    discovery = FileDiscovery(
        pupitre_datadir=args.pupitre_datadir,
        pigbrother_datadir=args.pigbrother_datadir,
        pigbrother_runlog_dir=Path(args.pigbrother_datadir) / args.housing / "Overview",
        hybrid_datadir=args.hybrid_datadir if args.housing == "M8" else None,
    )
    sessions = load_sessions(args.log, args.housing, start, end)
    discover_sessions(sessions, discovery)
    unlogged = find_unlogged(sessions, discovery, args.housing, start, end)
    all_sessions = sorted(sessions + unlogged, key=lambda s: s.start)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sessions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sessions_to_records(all_sessions), f, indent=2, ensure_ascii=False)

    anchors = {k: sum(s.anchor == k for s in sessions) for k in ("overview", "archive", "pupitre", "none")}
    print(
        f"{args.housing} {start:%Y-%m-%d} → {end:%Y-%m-%d}: {len(sessions)} logged sessions "
        f"(anchor {anchors}), {len(unlogged)} unlogged; written to {json_path}"
    )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``sessions`` subcommand on *sub*."""
    from ..cli_args import create_datadir_parser
    from ..data_dirs import HYBRID_DATA_DIR

    p = sub.add_parser(
        "sessions",
        parents=[create_datadir_parser()],
        help="list the files of each EXPERIENCES_LOG session",
        description=(
            "Build one record per EXPERIENCES_LOG session of a housing, listing the "
            "overview, archive, pupitre, incident, stats, hybrid and runlog files "
            "recorded during it; files claimed by no session are grouped into "
            "unlogged records. Writes sessions.json."
        ),
    )
    p.add_argument("--log", required=True, help="EXPERIENCES_LOG CSV file")
    p.add_argument("--housing", required=True, help="housing (ex. M8, M9, ...)")
    p.add_argument(
        "--from", dest="date_from", required=True,
        help="first day, ISO format (ex. 2018-01-01); Europe/Paris local time",
    )
    p.add_argument(
        "--to", dest="date_to", default=None,
        help="end (exclusive), ISO format; default: now",
    )
    p.add_argument(
        "--hybrid_datadir", default=HYBRID_DATA_DIR or None,
        help="base directory for hybrid (kHz/rms/trigger) data; used for M8 only",
    )
    p.add_argument("--output-dir", default=".", help="directory for sessions.json")
    p.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
    )
    p.set_defaults(_handler=_run)
