"""Shared utilities for parsing timestamps from magnet data filenames.

Public API
----------
parse_txt_filename(filename)        → datetime | None
parse_tdms_filename(filename)       → datetime | None
parse_filename_timestamp(filename)  → datetime | None   (dispatches on extension)
parse_wf_start_time(groups)         → datetime | None   (from TDMS channel properties)
seconds_since_midnight(dt)          → float
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

# Pupitre .txt date/time formats, newest first.
# The date/time token is the last ``_``-separated component of the stem,
# so an optional ``housing_`` prefix is stripped automatically.
TXT_TIMESTAMP_FORMATS = (
    "%Y.%m.%d - %H:%M:%S",  # new standard:  YYYY.MM.DD - HH:MM:SS
    "%Y-%m-%d_%H-%M-%S",  # alternative:   YYYY-MM-DD_HH-MM-SS
    "%Y.%m.%d---%H:%M:%S",  # legacy:         YYYY.MM.DD---HH:MM:SS
)


def parse_txt_filename(filename: str) -> datetime | None:
    """Parse a :class:`~datetime.datetime` from a pupitre ``.txt`` filename.

    Handles three formats (newest first), with an optional ``housing_`` prefix:

    * new standard – ``[housing_]YYYY.MM.DD - HH:MM:SS.txt``
    * alternative  – ``[housing_]YYYY-MM-DD_HH-MM-SS.txt``
    * legacy       – ``[housing_]YYYY.MM.DD---HH:MM:SS.txt``

    Returns ``None`` for non-``.txt`` files or unrecognised date formats.
    """
    name, ext = os.path.splitext(os.path.basename(filename))
    if ext != ".txt":
        return None
    date_string = name.split("_")[-1]
    for fmt in TXT_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    logger.warning("parse_txt_filename: unrecognised date format in %r", filename)
    return None


def parse_tdms_filename(filename: str) -> datetime | None:
    """Parse a :class:`~datetime.datetime` from a pigbrother ``.tdms`` filename.

    Expected stem format: ``site_mode_YYMMDD-HHMM[SS][_dmode]``
    e.g. ``M8_Overview_251105-0949.tdms`` or ``M8_Default_251105-095300_raw.tdms``.

    Returns ``None`` for non-``.tdms`` files or unrecognised formats.
    """
    name, ext = os.path.splitext(os.path.basename(filename))
    if ext != ".tdms":
        return None
    parts = name.split("_")
    if len(parts) < 3:
        logger.warning(
            "parse_tdms_filename: cannot parse %r (expected site_mode_timestamp[_dmode])",
            filename,
        )
        return None
    timestamp_part = parts[2]
    try:
        date_str, time_str = timestamp_part.split("-", 1)
    except ValueError:
        logger.warning(
            "parse_tdms_filename: cannot split timestamp %r in %r",
            timestamp_part,
            filename,
        )
        return None
    # Try HHMMSS (6-char time), fall back to HHMM (4-char time).
    for fmt, t in (
        ("%y%m%d%H%M%S", time_str[:6]),
        ("%y%m%d%H%M", time_str[:4]),
    ):
        try:
            return datetime.strptime(date_str + t, fmt)
        except ValueError:
            continue
    logger.warning("parse_tdms_filename: unrecognised tdms date format in %r", filename)
    return None


def parse_filename_timestamp(filename: str) -> datetime | None:
    """Parse a start :class:`~datetime.datetime` from a magnet data filename.

    Dispatches to :func:`parse_txt_filename` for ``.txt`` files and
    :func:`parse_tdms_filename` for ``.tdms`` files.  Returns ``None`` for
    unsupported extensions or unparseable names.
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".txt":
        return parse_txt_filename(filename)
    if ext == ".tdms":
        return parse_tdms_filename(filename)
    logger.debug(
        "parse_filename_timestamp: unsupported extension %r for %r", ext, filename
    )
    return None


def parse_wf_start_time(groups: dict) -> datetime | None:
    """Extract a start :class:`~datetime.datetime` from TDMS channel properties.

    Iterates over *groups* (the ``Groups`` dict of a :class:`TdmsMagnetData`)
    and returns the ``wf_start_time`` property of the first channel that has
    one, skipping the ``"Infos"`` group.  Returns ``None`` when no
    ``wf_start_time`` is found.

    The ``wf_start_time`` value is a numpy datetime64 scalar stored by
    *nptdms*; it is converted to a plain :class:`~datetime.datetime` via
    ``.astype(datetime)``.
    """
    for gname, channels in groups.items():
        if gname == "Infos":
            continue
        if not isinstance(channels, dict):
            continue
        for props in channels.values():
            if not isinstance(props, dict):
                continue
            if "wf_start_time" not in props:
                continue
            try:
                return props["wf_start_time"].astype(datetime)
            except (TypeError, ValueError):
                logger.warning(
                    f"parse_wf_start_time: could not convert wf_start_time in group {gname}"
                )
                return None
    return None


def seconds_since_midnight(dt: datetime) -> float:
    """Return seconds elapsed since midnight for *dt*."""
    return float(dt.hour * 3600 + dt.minute * 60 + dt.second)
