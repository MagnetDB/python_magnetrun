"""
Utility functions for hybrid data processing

Includes:
- Date listing utilities
- Error logging utilities

Re-exported for backward compatibility:
- Outlier detection: ``remove_outliers``, ``detect_outliers``, ``OutlierDetector``
  (canonical: :mod:`python_magnetrun.outliers`)
- Signal processing: ``normalize_signal``, ``binarize_signal``, ``_otsu_threshold``
  (canonical: :mod:`python_magnetrun.processing.signal`)
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from ..outliers import OutlierDetector, detect_outliers, remove_outliers  # noqa: F401
from ..processing.signal import (  # noqa: F401
    _otsu_threshold,
    binarize_signal,
    normalize_signal,
)

# Setup logger
logger = logging.getLogger(__name__)


def log_exception(
    message: str,
    exception: Exception,
    logger_instance: logging.Logger | None = None,
    use_print: bool = False,
    include_traceback: bool = True,
) -> None:
    """
    Log exception with traceback information

    Parameters
    ----------
    message : str
        Custom error message to display
    exception : Exception
        The exception that was caught
    logger_instance : logging.Logger, optional
        Logger instance to use. If None, uses print or module logger
    use_print : bool
        If True and logger_instance is None, uses print instead of logger
    include_traceback : bool
        If True, includes full traceback. Otherwise just file, line, and function

    Examples
    --------
    >>> try:
    ...     risky_operation()
    ... except Exception as e:
    ...     log_exception("Failed to perform operation", e)
    """
    # Get exception information
    exc_type, exc_value, exc_tb = sys.exc_info()

    # Format the error message
    if include_traceback:
        # Full traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        error_msg = f"{message}: {exception}\n{''.join(tb_lines)}"
    else:
        # Just file, line, and function where error occurred
        if exc_tb is not None:
            tb = traceback.extract_tb(exc_tb)
            if tb:
                # Get the last frame (where the error actually occurred)
                frame = tb[-1]
                error_msg = (
                    f"{message}: {exception}\n"
                    f"  File: {frame.filename}\n"
                    f"  Line: {frame.lineno}\n"
                    f"  Function: {frame.name}"
                )
            else:
                error_msg = f"{message}: {exception}"
        else:
            error_msg = f"{message}: {exception}"

    # Log or print the error
    if logger_instance:
        logger_instance.error(error_msg)
    elif use_print:
        print(error_msg)
    else:
        logger.error(error_msg)


def format_exception_location(exception: Exception | None = None) -> str:
    """
    Get a concise string with file:line:function where exception occurred

    Parameters
    ----------
    exception : Exception, optional
        The exception (not used, but kept for API consistency)

    Returns
    -------
    str
        Formatted string like "file.py:123:function_name"

    Examples
    --------
    >>> try:
    ...     risky_operation()
    ... except Exception as e:
    ...     location = format_exception_location()
    ...     print(f"Error at {location}: {e}")
    """
    exc_type, exc_value, exc_tb = sys.exc_info()

    if exc_tb is not None:
        tb = traceback.extract_tb(exc_tb)
        if tb:
            frame = tb[-1]
            filename = Path(frame.filename).name
            return f"{filename}:{frame.lineno}:{frame.name}"

    return "unknown:?:?"


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


