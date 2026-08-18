"""pupitre — Cirrus power-supply run-log reader (stub).

Cirrus run-log files are written by the pupitre control system, one file per
power-supply unit (A1–A4) per calendar day::

    <runlog_root>/cirrus/A1/2026-04-16_cirrus_out.log
    <runlog_root>/cirrus/A2/2026-04-16_cirrus_out.log
    ...

Raw file format (``*_cirrus_out.log``)
--------------------------------------
Each line has three whitespace-separated fields::

    <Type>  <RelativeTime>  <Message>

``RelativeTime`` is seconds elapsed since an unspecified origin.  Absolute
timestamps must be inferred by anchoring on embedded messages of the form::

    Info  1234.5  Arret redresseur date  2026-4-15 16:0:9.111719

The CSV export (``Logs Cirrus.csv``) already has absolute timestamps computed
by the Cirrus application and can be used as a reference or alternative input.

This module is a **stub** — the discovery helper is functional; the
``CirrusRunlogLoader`` class exposes the intended interface but parsing is not
yet implemented.
"""

from __future__ import annotations

import glob
import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex for the absolute-timestamp anchor embedded in log messages.
# Matches: "Arret redresseur date  2026-4-15 16:0:9.111719"
_ANCHOR_RE = re.compile(
    r"Arret\s+redresseur\s+date\s+"
    r"(\d{4})-(\d{1,2})-(\d{1,2})\s+"
    r"(\d{1,2}):(\d{1,2}):(\d{1,2})"
)

# Power-supply unit names
PSU_NAMES: tuple[str, ...] = ("A1", "A2", "A3", "A4")


@dataclass
class CirrusLogEntry:
    """Single parsed line from a Cirrus ``*_cirrus_out.log`` file."""

    psu: str
    """Power-supply unit (e.g. ``'A1'``)."""
    log_type: str
    """Entry type as written in the file (e.g. ``'Info'``, ``'Erreur'``)."""
    relative_time: float
    """Relative time in seconds from the file origin."""
    message: str
    """Raw message string."""
    absolute_time: float | None = None
    """Unix timestamp (set after anchor-based calibration, else ``None``)."""


@dataclass
class CirrusRunlogLoader:
    """Reader for Cirrus power-supply run-log files.

    Parameters
    ----------
    runlog_root:
        Root directory that contains the ``cirrus/`` subtree.
        Expects ``<runlog_root>/cirrus/A[1-4]/YYYY-MM-DD_cirrus_out.log``.

    .. note::
        Parsing is not yet implemented.  The class exposes the intended public
        interface so that callers can be written now and the implementation
        filled in later.

    Usage::

        loader = CirrusRunlogLoader("/mnt/data/records/srv-data-install/M9")
        files = loader.find_files("2026-04-15", "2026-04-16")
        entries = loader.load(files)
    """

    runlog_root: Path
    psu_names: tuple[str, ...] = field(default_factory=lambda: PSU_NAMES)

    def __post_init__(self) -> None:
        self.runlog_root = Path(self.runlog_root)

    def find_files(self, start_date: str, end_date: str) -> list[Path]:
        """Return all Cirrus log files covering [start_date, end_date].

        Parameters
        ----------
        start_date, end_date:
            ISO-format date strings (``'YYYY-MM-DD'``).

        Returns
        -------
        list[Path]
            Sorted list of matching ``*_cirrus_out.log`` paths.
        """
        d0 = date.fromisoformat(start_date)
        d1 = date.fromisoformat(end_date)
        days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

        found: list[Path] = []
        for day in days:
            day_str = day.strftime("%Y-%m-%d")
            for psu in self.psu_names:
                pattern = str(
                    self.runlog_root / "cirrus" / psu / f"{day_str}_cirrus_out.log"
                )
                found.extend(Path(p) for p in glob.glob(pattern))

        found.sort()
        return found

    def load(self, files: list[Path]) -> list[CirrusLogEntry]:
        """Parse a list of Cirrus log files into :class:`CirrusLogEntry` records.

        .. warning::
            Not yet implemented — returns an empty list and logs a warning.

        Parameters
        ----------
        files:
            Paths returned by :meth:`find_files` (or any subset thereof).

        Returns
        -------
        list[CirrusLogEntry]
            Parsed entries with ``absolute_time`` set where anchoring is
            possible.
        """
        if files:
            logger.warning(
                "CirrusRunlogLoader.load() is not yet implemented; "
                "returning empty list for %d file(s).",
                len(files),
            )
        return []


def discover_pupitre_runlogs(
    runlog_root: str | Path,
    start_date: str,
    end_date: str,
    psu_names: tuple[str, ...] = PSU_NAMES,
) -> list[str]:
    """Convenience wrapper — discover Cirrus log files as a list of strings.

    Parameters
    ----------
    runlog_root:
        Root directory containing the ``cirrus/`` subtree.
    start_date, end_date:
        ISO-format date strings (``'YYYY-MM-DD'``).
    psu_names:
        Power-supply names to probe (default: ``('A1', 'A2', 'A3', 'A4')``).

    Returns
    -------
    list[str]
        Sorted list of matching file paths as strings.
    """
    loader = CirrusRunlogLoader(runlog_root=runlog_root, psu_names=psu_names)
    return [str(p) for p in loader.find_files(start_date, end_date)]
