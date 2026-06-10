"""
Utility functions for hybrid data processing

Includes:
- Date listing utilities

Re-exported for backward compatibility:
- Outlier detection: ``remove_outliers``, ``detect_outliers``, ``OutlierDetector``
  (canonical: :mod:`python_magnetrun.outliers`)
- Signal processing: ``normalize_signal``, ``binarize_signal``, ``_otsu_threshold``
  (canonical: :mod:`python_magnetrun.processing.signal`)
"""

from datetime import datetime
from pathlib import Path

from ..outliers import OutlierDetector, detect_outliers, remove_outliers  # noqa: F401
from ..processing.signal import (  # noqa: F401
    _otsu_threshold,
    binarize_signal,
    normalize_signal,
)

def local_hour_to_utc(local_h: int, date_str: str, tz: str = "Europe/Paris") -> int:
    """Convert a local integer hour to the equivalent UTC hour for the given date.

    Parameters
    ----------
    local_h : int
        Local hour (0–23) in *tz*.
    date_str : str
        ISO date string, e.g. ``"2025-01-27"``.
    tz : str
        IANA timezone name for the local zone.  Defaults to ``"Europe/Paris"``.

    Returns
    -------
    int
        UTC hour corresponding to *local_h* on *date_str*.
    """
    import datetime as _dt
    from zoneinfo import ZoneInfo

    d = _dt.date.fromisoformat(date_str)
    return (
        _dt.datetime(d.year, d.month, d.day, local_h, 0, 0, tzinfo=ZoneInfo(tz))
        .astimezone(ZoneInfo("UTC"))
        .hour
    )


def utc_hour_to_local(utc_h: int, date_str: str, tz: str = "Europe/Paris") -> int:
    """Convert a UTC integer hour to the equivalent local hour for the given date.

    Parameters
    ----------
    utc_h : int
        UTC hour (0–23).
    date_str : str
        ISO date string, e.g. ``"2025-01-27"``.
    tz : str
        IANA timezone name for the local zone.  Defaults to ``"Europe/Paris"``.

    Returns
    -------
    int
        Local hour in the given timezone.
    """
    import datetime as _dt
    from zoneinfo import ZoneInfo

    d = _dt.date.fromisoformat(date_str)
    return (
        _dt.datetime(d.year, d.month, d.day, utc_h, 0, 0, tzinfo=ZoneInfo("UTC"))
        .astimezone(ZoneInfo(tz))
        .hour
    )


def list_available_dates(base_dir: str | Path, data_type: str = "kHz") -> list[str]:
    """
    List available dates for a given data type

    Parameters
    ----------
    base_dir : str or Path
        Base directory
    data_type : str
        Data type: 'kHz', 'rms', or 'trigger'

    Returns
    -------
    list of str
        List of date strings in YYYY-MM-DD format
    """
    base_path = Path(base_dir) / data_type
    if not base_path.exists():
        return []

    dates = set()
    for item in base_path.iterdir():
        if item.is_dir():
            if data_type == "trigger":
                # Trigger directories are named TRIGGER__YYYY-MM-DD__HH-MM
                if item.name.startswith("TRIGGER__"):
                    parts = item.name.split("__")
                    if len(parts) >= 2:
                        date_str = parts[1]
                        try:
                            datetime.strptime(date_str, "%Y-%m-%d")
                            dates.add(date_str)
                        except ValueError:
                            pass
            else:
                # kHz and rms directories are named YYYY-MM-DD
                try:
                    datetime.strptime(item.name, "%Y-%m-%d")
                    dates.add(item.name)
                except ValueError:
                    pass

    return sorted(dates)


