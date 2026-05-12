"""
Utility functions for hybrid data processing

Includes:
- Date listing utilities
- Error logging utilities
- Signal normalization and binarization

Outlier detection is provided by :mod:`.outliers` and re-exported here for
backward compatibility (``remove_outliers``, ``detect_outliers``,
``OutlierDetector``).
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from ..outliers import OutlierDetector, detect_outliers, remove_outliers  # noqa: F401

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


def normalize_signal(data: np.ndarray) -> np.ndarray:
    """
    Normalize a signal by its maximum absolute value.

    Parameters
    ----------
    data : np.ndarray
        Input signal data.

    Returns
    -------
    np.ndarray
        Signal divided by its maximum absolute value, or unchanged if max is 0.
    """
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        return data
    return data / max_abs


def _otsu_threshold(abs_data: np.ndarray, n_bins: int = 256) -> float:
    """Compute Otsu's optimal threshold on absolute-value data.

    Uses log-scale binning so the narrow near-zero off-state population
    gets adequate resolution when the signal dynamic range is large.
    """
    eps = 1e-9
    positive = abs_data[abs_data > eps]
    if positive.size == 0:
        return 0.0

    log_data = np.log10(positive)
    counts, bin_edges = np.histogram(log_data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = counts.sum()
    if total == 0:
        return 0.0

    weights = counts / total
    cum_w = np.cumsum(weights)
    cum_mean = np.cumsum(weights * bin_centers)
    global_mean = cum_mean[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        between_var = np.where(
            (cum_w > 0) & (cum_w < 1),
            (global_mean * cum_w - cum_mean) ** 2 / (cum_w * (1.0 - cum_w)),
            0.0,
        )

    log_threshold = float(bin_centers[np.argmax(between_var)])
    return float(10**log_threshold)


def binarize_signal(
    data: np.ndarray,
    tolerance: float = 0.005,
    method: str = "otsu",
    n_bins: int = 256,
    normalize: bool = True,
    noise_percentile: float = 40.0,
) -> np.ndarray:
    """
    Return a binary array: 0 (signal off) or 1 (signal on).

    Parameters
    ----------
    data : array-like
        Input signal data (any scale).
    tolerance : float, optional
        Threshold value used only when method='fixed' (default: 0.005).
    method : str, optional
        'fixed'  — use the tolerance value directly.
        'otsu'   — compute the threshold automatically via Otsu's method
                   on the absolute values using log-scale bins (default).
        'noise'  — estimate the noise floor from the bottom
                   `noise_percentile` % of absolute values and use
                   3*sigma of that population as threshold.
    n_bins : int, optional
        Number of histogram bins for Otsu's method (default: 256).
    normalize : bool, optional
        If True (default), normalize the signal by its maximum absolute value
        before applying the threshold, so tolerance is scale-independent.
    noise_percentile : float, optional
        Percentile (0–100) defining the noise population when
        method='noise' (default: 40.0).

    Returns
    -------
    np.ndarray
        Integer array of 0s and 1s.
    """
    data = np.asarray(data, dtype=float)
    if normalize:
        data = normalize_signal(data)
        logger.debug(
            "data stats: min=%.4g, max=%.4g, mean=%.4g, std=%.4g",
            data.min(),
            data.max(),
            data.mean(),
            data.std(),
        )
    abs_data = np.abs(data)

    if method == "fixed":
        threshold = tolerance
    elif method == "otsu":
        threshold = _otsu_threshold(abs_data, n_bins)
    elif method == "noise":
        noise_ceil = float(np.percentile(abs_data, noise_percentile))
        quiet = abs_data[abs_data <= noise_ceil]
        sigma = float(quiet.std())
        logger.debug(
            "Noise floor (p%s=%.4g), 3*sigma=%.4g as threshold.",
            noise_percentile,
            noise_ceil,
            3.0 * sigma,
        )
        threshold = 3.0 * sigma
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose 'fixed', 'otsu', or 'noise'."
        )
    logger.debug("Using threshold: %.4g (method=%s)", threshold, method)
    return np.where(abs_data <= threshold, 0, 1)
