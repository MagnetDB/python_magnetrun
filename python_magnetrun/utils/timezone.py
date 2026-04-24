"""Timezone conversion utilities for magnet data timestamps."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytz


def utc_naive_to_local(
    ts: pd.Timestamp | datetime,
    time_zone: str = "Europe/Paris",
) -> datetime:
    """Convert a naive UTC timestamp to a naive local datetime.

    :param ts: naive UTC (or tz-aware UTC) timestamp
    :param time_zone: IANA timezone for the output (default ``"Europe/Paris"``)
    :return: naive local :class:`~datetime.datetime`
    """
    pts = pd.Timestamp(ts)
    if pts.tzinfo is None:
        pts = pts.tz_localize(pytz.utc)
    tz = pytz.timezone(time_zone)
    return pts.tz_convert(tz).to_pydatetime().replace(tzinfo=None)


def local_to_utc_naive(
    ts: pd.Timestamp | datetime,
    time_zone: str = "Europe/Paris",
) -> datetime:
    """Convert a local (naive) timestamp to a naive UTC datetime.

    If *ts* is already timezone-aware its tzinfo is used as-is (no
    re-localizing).

    :param ts: local naive (or tz-aware) timestamp
    :param time_zone: IANA timezone of the local time (default ``"Europe/Paris"``)
    :return: naive UTC :class:`~datetime.datetime`
    """
    pts = pd.Timestamp(ts)
    if pts.tzinfo is None:
        pts = pts.tz_localize(pytz.timezone(time_zone))
    return pts.tz_convert(pytz.utc).to_pydatetime().replace(tzinfo=None)


def ensure_utc_naive(ts: pd.Timestamp | datetime) -> datetime:
    """Normalise an already-UTC timestamp to a naive UTC datetime.

    When *ts* is tz-aware it is converted to UTC; when naive it is
    assumed to be UTC already.

    :param ts: UTC (or naive-UTC) timestamp
    :return: naive UTC :class:`~datetime.datetime`
    """
    pts = pd.Timestamp(ts)
    if pts.tzinfo is None:
        pts = pts.tz_localize(pytz.utc)
    return pts.tz_convert(pytz.utc).to_pydatetime().replace(tzinfo=None)


def series_local_to_utc_naive(
    series: pd.Series,
    time_zone: str = "Europe/Paris",
) -> pd.Series:
    """Convert a Series of naive local datetimes to naive UTC.

    :param series: datetime Series in local time (naive)
    :param time_zone: IANA timezone name (default ``"Europe/Paris"``)
    :return: naive UTC datetime Series
    """
    tz = pytz.timezone(time_zone)
    return (
        series.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert(pytz.utc)
        .dt.tz_localize(None)
    )


def series_utc_to_local_naive(
    series: pd.Series,
    time_zone: str = "Europe/Paris",
) -> pd.Series:
    """Convert a Series of naive UTC datetimes to naive local time (for display).

    :param series: naive UTC datetime Series
    :param time_zone: IANA timezone name (default ``"Europe/Paris"``)
    :return: naive local datetime Series
    """
    tz = pytz.timezone(time_zone)
    return series.dt.tz_localize(pytz.utc).dt.tz_convert(tz).dt.tz_localize(None)


def timerange_to_utc(
    timerange: str,
    time_zone: str = "Europe/Paris",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse a ``"start;end"`` local-time string into a pair of naive UTC Timestamps.

    :param timerange: ``"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"`` in local time
    :param time_zone: IANA timezone (default ``"Europe/Paris"``)
    :return: ``(t_start, t_end)`` as naive UTC :class:`~pandas.Timestamp`
    """
    tz = pytz.timezone(time_zone)
    t0, t1 = timerange.split(";")
    t_start = pd.Timestamp(t0).tz_localize(tz).tz_convert(pytz.utc).tz_localize(None)
    t_end = pd.Timestamp(t1).tz_localize(tz).tz_convert(pytz.utc).tz_localize(None)
    return t_start, t_end
