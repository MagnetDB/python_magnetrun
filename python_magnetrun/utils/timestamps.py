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

import pytz

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
    logger.warning(f"parse_txt_filename: unrecognised date format in {filename}")
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
            f"parse_tdms_filename: cannot parse {filename} (expected site_mode_timestamp[_dmode])"
        )
        return None
    timestamp_part = parts[2]
    try:
        date_str, time_str = timestamp_part.split("-", 1)
    except ValueError:
        logger.warning(
            f"parse_tdms_filename: cannot split timestamp {timestamp_part} in {filename}",
            timestamp_part,
            filename,
        )
        return None
    # Try HHMMSS (6-char time), fall back to HHMM (4-char time).
    # Guard on slice length first: strptime is lenient about %M/%S digit count,
    # so "1506" (4 chars) fed to %H%M%S would mis-parse as H=15, M=0, S=6.
    for fmt, t in (
        ("%y%m%d%H%M%S", time_str[:6]),
        ("%y%m%d%H%M", time_str[:4]),
    ):
        if len(t) < (6 if "S" in fmt else 4):
            continue
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
        f"parse_filename_timestamp: unsupported extension {ext} for {filename}"
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


def convert_to_timestamp_aware(
    date_str: str,
    time_str: str,
    date_format: str = "%y%m%d",
    time_format: str = "%H%M",
    time_zone: str = "Europe/Paris",
) -> tuple:
    """Convert date and time strings to a UTC timestamp, handling the input time zone.

    :param date_str: Date string (e.g., '230718')
    :param time_str: Time string (e.g., '1506')
    :param date_format: Format string for the date part (e.g., '%y%m%d')
    :param time_format: Format string for the time part (e.g., '%H%M')
    :param time_zone: The time zone of the input date/time (e.g., 'Europe/Paris')
    :return: A tuple (UTC timestamp as float, UTC formatted datetime string).
    """
    date_time_str = date_str + time_str
    date_time_format = date_format + time_format
    naive_dt = datetime.strptime(date_time_str, date_time_format)

    tz = pytz.timezone(time_zone)
    aware_dt_local = tz.localize(naive_dt)
    aware_dt_utc = aware_dt_local.astimezone(pytz.utc)

    timestamp = aware_dt_utc.timestamp()
    formatted_date_time_utc = aware_dt_utc.strftime("%Y-%m-%dT%H:%M:%S")

    return (timestamp, formatted_date_time_utc)


def convert_to_timestamp(
    date_str: str,
    time_str: str,
    date_format: str = "%y%m%d",
    time_format: str = "%H%M",
) -> tuple:
    """Convert date and time strings to a naive local timestamp.

    Examples of format pairs:

    * TDMS files:    date_format="%y%m%d",  time_format="%H%M%S"
    * Pupitre files: date_format="%Y%m%d",  time_format="%H:%M:%S"

    :param date_str: Date string.
    :param time_str: Time string.
    :param date_format: strptime format for the date part.
    :param time_format: strptime format for the time part.
    :return: A tuple (local timestamp as float, formatted datetime string).
    """
    date_time_str = date_str + time_str
    date_time_format = date_format + time_format
    date_time_obj = datetime.strptime(date_time_str, date_time_format)

    timestamp = date_time_obj.timestamp()
    formatted_date_time = date_time_obj.strftime("%Y-%m-%d %H:%M:%S")

    return (timestamp, formatted_date_time)
