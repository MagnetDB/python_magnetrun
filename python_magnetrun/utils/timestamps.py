"""Shared utilities for parsing timestamps from magnet data filenames.

Public API
----------
parse_txt_filename(filename)                      → datetime | None
parse_tdms_filename(filename)                     → datetime | None
parse_filename_timestamp(filename)                → datetime | None   (dispatches on extension)
parse_wf_start_time(groups)                       → datetime | None   (from TDMS channel properties)
seconds_since_midnight(dt)                        → float
align_to_common_time(sources, reference, hours)   → dict[int, float]

Timezone conversion helpers have moved to :mod:`python_magnetrun.utils.timezone`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

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
    logger.debug(f"parsing {filename}")
    name, ext = os.path.splitext(os.path.basename(filename))
    logger.debug(f"name={name!r}, ext={ext!r}")
    if ext != ".txt":
        return None
    date_string = name.split("_")[-1]
    logger.debug(f"date_string={date_string!r}")
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
    logger.debug(f"name={name!r}, ext={ext!r}")
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
            f"parse_tdms_filename: cannot split timestamp {timestamp_part} in {filename}"
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
    logger.warning(
        f"parse_tdms_filename: unrecognised tdms date format in {filename!r}"
    )
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


@runtime_checkable
class _HasTimeRange(Protocol):
    def get_time_range(self) -> tuple[datetime, datetime]: ...


def align_to_common_time(
    sources: Sequence[_HasTimeRange],
    reference: datetime | None = None,
    hours: Sequence[int] | None = None,
) -> dict[int, float]:
    """Compute per-source time offsets [s] relative to a common UTC reference.

    Parameters
    ----------
    sources : Sequence
        Objects implementing ``get_time_range() -> (naive_utc, naive_utc)``.
        Typically :class:`~python_magnetrun.MagnetRun.MagnetRun` and
        :class:`~python_magnetrun.hybrid.hybrid_run.HybridRun` instances.
    reference : datetime, optional
        Explicit reference (naive UTC).  Defaults to the earliest
        ``get_time_range()[0]`` across all sources.
    hours : sequence of int, optional
        UTC hours to restrict/anchor.  When provided the reference is snapped
        to ``hours[0]:00:00`` on the same calendar day as *reference* (or the
        earliest source start when *reference* is ``None``).

    Returns
    -------
    dict[int, float]
        Maps ``id(source)`` to an offset in seconds.
        Add the offset to a source's time array (already in seconds from its
        own start) to place it on an axis where *t = 0* is *t_ref*:
        ``aligned_time = time_array + offsets[id(source)]``.

    Examples
    --------
    >>> offsets = align_to_common_time([hrun, pupitre_run], hours=[10, 11])
    >>> data, time = pupitre_run.getMData().getData(["t", field])
    >>> aligned_time = time["t"].to_numpy() + offsets[id(pupitre_run)]
    """
    t0s = {id(s): s.get_time_range()[0] for s in sources}
    for s in sources:
        logger.info(f"Source {type(s).__name__} (id={id(s)}): start time {t0s[id(s)]}")
    ref = pd.Timestamp(reference) if reference is not None else pd.Timestamp(min(t0s.values()))
    if hours is not None:
        ref = ref.replace(hour=hours[0], minute=0, second=0, microsecond=0)
    return {k: (pd.Timestamp(t0) - ref).total_seconds() for k, t0 in t0s.items()}


def add_time_columns(
    df: pd.DataFrame,
    t0: datetime,
    sampling_rate: float = 0.0,
    timestamp_col: str = "timestamp",
    time_col: str = "t",
) -> pd.DataFrame:
    """Return a copy of *df* with a float-seconds column *time_col*.

    Each value is ``(timestamp - t0).total_seconds() + half_period_offset``
    where ``half_period_offset = 1 / (2 * sampling_rate)`` when
    *sampling_rate* > 0 (TDMS downsampled data represents the centre of
    each sampling window).  When *sampling_rate* is 0 (default) no offset
    is applied.
    """
    t_offset = (1.0 / (2.0 * sampling_rate)) if sampling_rate else 0.0
    df = df.copy()
    if time_col in df.columns:
        df.drop(columns=[time_col], inplace=True)
    df[time_col] = df[timestamp_col].map(
        lambda ts: (ts - t0).total_seconds() + t_offset
    )
    return df
