"""
Utility functions for hybrid data processing

Includes:
- Outlier detection and removal
- Date listing utilities
- Error logging utilities
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

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


def remove_outliers(
    data: np.ndarray,
    time: np.ndarray,
    method: str = "iqr",
    threshold: float = 1.5,
    window_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Remove outliers from data

    Parameters
    ----------
    data : np.ndarray
        Input data array
    time : np.ndarray
        Time array corresponding to data
    method : str
        Outlier detection method:
        - 'iqr': Interquartile Range (default)
        - 'zscore': Z-score based
        - 'mad': Median Absolute Deviation
        - 'percentile': Percentile-based clipping
    threshold : float
        Threshold for outlier detection:
        - For 'iqr': IQR multiplier (default: 1.5, use 3.0 for extreme outliers)
        - For 'zscore': Number of standard deviations (default: 3.0)
        - For 'mad': MAD multiplier (default: 3.5)
        - For 'percentile': Percentile to clip (e.g., 1.0 clips 1% from each end)
    window_size : int, optional
        If provided, use rolling window for local outlier detection

    Returns
    -------
    clean_data : np.ndarray
        Data with outliers removed (replaced with interpolated values)
    clean_time : np.ndarray
        Corresponding time array
    n_outliers : int
        Number of outliers detected
    """
    data = data.copy().astype(np.float64)

    if window_size is not None and window_size > 0:
        # Rolling window outlier detection
        mask = _rolling_outlier_mask(data, window_size, method, threshold)
    else:
        # Global outlier detection
        mask = _global_outlier_mask(data, method, threshold)

    n_outliers = np.sum(mask)

    if n_outliers > 0:
        # Replace outliers with NaN, then interpolate
        data[mask] = np.nan

        # Interpolate NaN values
        valid_idx = ~np.isnan(data)
        if np.sum(valid_idx) > 2:
            data = np.interp(np.arange(len(data)), np.arange(len(data))[valid_idx], data[valid_idx])

    return data, time, n_outliers


def _global_outlier_mask(data: np.ndarray, method: str, threshold: float) -> np.ndarray:
    """
    Create mask for global outliers

    Returns boolean array where True indicates outlier
    """
    if method == "iqr":
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        mean = np.nanmean(data)
        std = np.nanstd(data)
        if std > 0:
            z_scores = np.abs((data - mean) / std)
            mask = z_scores > threshold
        else:
            mask = np.zeros(len(data), dtype=bool)

    elif method == "mad":
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median))
        if mad > 0:
            modified_z = 0.6745 * (data - median) / mad
            mask = np.abs(modified_z) > threshold
        else:
            mask = np.zeros(len(data), dtype=bool)

    elif method == "percentile":
        lower_bound = np.nanpercentile(data, threshold)
        upper_bound = np.nanpercentile(data, 100 - threshold)
        mask = (data < lower_bound) | (data > upper_bound)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr', 'zscore', 'mad', or 'percentile'")

    return mask


def _rolling_outlier_mask(
    data: np.ndarray, window_size: int, method: str, threshold: float
) -> np.ndarray:
    """
    Create mask for rolling window outliers

    Returns boolean array where True indicates outlier
    """
    n = len(data)
    mask = np.zeros(n, dtype=bool)
    half_window = window_size // 2

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window_data = data[start:end]

        if method == "iqr":
            q1 = np.nanpercentile(window_data, 25)
            q3 = np.nanpercentile(window_data, 75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                mask[i] = data[i] < lower or data[i] > upper

        elif method == "zscore":
            mean = np.nanmean(window_data)
            std = np.nanstd(window_data)
            if std > 0:
                z = abs((data[i] - mean) / std)
                mask[i] = z > threshold

        elif method == "mad":
            median = np.nanmedian(window_data)
            mad = np.nanmedian(np.abs(window_data - median))
            if mad > 0:
                modified_z = 0.6745 * abs(data[i] - median) / mad
                mask[i] = modified_z > threshold

    return mask
