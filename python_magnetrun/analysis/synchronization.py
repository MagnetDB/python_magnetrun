"""
Time synchronization logic for magnetrun analysis.

This module handles synchronization between different data sources:
- Overview TDMS files (1 Hz)
- Archive TDMS files (120 Hz)
- Pupitre text files (variable rate)
- Incident files (4800 Hz)

The synchronization process involves:
1. Extracting timestamps from each source
2. Computing time shifts between sources
3. Aligning all data to a common time base
4. Computing and applying lag corrections using cross-correlation

The primary challenge is that different data sources may have:
- Different sampling rates
- Clock drift or offset
- Different start times

This module provides tools to detect and correct these differences.

Example usage::

    from python_magnetrun.analysis.synchronization import (
        synchronize_data,
        compute_lag,
        apply_lag_correction,
    )

    # Synchronize pupitre data to overview timestamp
    timeshift, df_synced = synchronize_data(df_pupitre, overview_t0)

    # Compute fine-grained lag using cross-correlation
    lag = compute_lag("timestamp", df1_data, df2_data)

    # Apply the lag correction
    df_corrected = apply_lag_correction(df_pupitre, lag)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import LAG_THRESHOLD_RATIO

# Module logger
logger = logging.getLogger("magnetrun.analysis.synchronization")


# =============================================================================
# Result dataclasses
# =============================================================================
@dataclass
class SyncResult:
    """
    Result of a synchronization operation.

    Attributes
    ----------
    timeshift : timedelta
        Time shift applied to align data
    df : pd.DataFrame
        Synchronized DataFrame
    source_t0 : datetime
        Original start time of source data
    target_t0 : datetime
        Target reference time
    """

    timeshift: timedelta
    df: pd.DataFrame
    source_t0: datetime
    target_t0: datetime

    @property
    def shift_seconds(self) -> float:
        """Get timeshift in seconds."""
        return self.timeshift.total_seconds()


@dataclass
class LagResult:
    """
    Result of a lag computation.

    Attributes
    ----------
    lag : timedelta
        Computed lag between series
    correlation : float
        Maximum correlation value
    confidence : float
        Confidence measure (0-1) based on correlation strength
    method : str
        Method used for lag computation
    """

    lag: timedelta
    correlation: float = 0.0
    confidence: float = 0.0
    method: str = "cross_correlation"

    @property
    def seconds(self) -> float:
        """Get lag in seconds."""
        return self.lag.total_seconds()


@dataclass
class RegimeMatch:
    """
    Result of matching regimes between two signatures.

    Attributes
    ----------
    regime : str
        Regime type ('U', 'P', 'D')
    match_regime : str
        Matched regime in reference
    score : float
        Match score (lower is better)
    lags : Tuple[float, float]
        (start_lag, end_lag) in seconds
    indices : Tuple[int, int]
        (source_index, reference_index)
    """

    regime: str
    match_regime: str
    score: float
    lags: Tuple[float, float]
    indices: Tuple[int, int]

    @property
    def is_reliable(self) -> bool:
        """Check if match is reliable based on lag threshold."""
        return abs(self.lags[0]) < float("inf") and abs(self.lags[1]) < float("inf")


# =============================================================================
# Basic synchronization
# =============================================================================
def synchronize_data(
    df: pd.DataFrame,
    target_t0: datetime,
    timestamp_col: str = "timestamp",
    time_col: str = "t",
) -> Tuple[timedelta, pd.DataFrame]:
    """
    Synchronize a DataFrame to a reference timestamp.

    Shifts the timestamps in the DataFrame so that the data aligns
    with a target reference time. Also recalculates the 't' column
    as seconds from the new start time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column
    target_t0 : datetime
        Target reference timestamp to align to
    timestamp_col : str, optional
        Name of timestamp column (default: "timestamp")
    time_col : str, optional
        Name of time-in-seconds column (default: "t")

    Returns
    -------
    Tuple[timedelta, pd.DataFrame]
        (timeshift_applied, synchronized_dataframe)

    Examples
    --------
    >>> timeshift, df_synced = synchronize_data(df_pupitre, overview_t0)
    >>> print(f"Applied shift: {timeshift.total_seconds()} seconds")
    """
    df = df.copy()

    # Get source start time
    source_t0 = df[timestamp_col].iloc[0]

    # Calculate timeshift
    timeshift = target_t0 - source_t0

    # Apply timeshift to timestamps
    df[timestamp_col] = df[timestamp_col] + pd.to_timedelta(timeshift)

    # Recalculate time column
    new_t0 = df[timestamp_col].iloc[0]
    if time_col in df.columns:
        df.drop([time_col], axis=1, inplace=True)
    df[time_col] = df.apply(
        lambda row: (row[timestamp_col] - new_t0).total_seconds(), axis=1
    )

    logger.debug(
        "Synchronized data: shift=%.3f s, source_t0=%s, target_t0=%s",
        timeshift.total_seconds(),
        source_t0,
        target_t0,
    )

    return timeshift, df


def apply_lag_correction(
    df: pd.DataFrame,
    lag: Union[timedelta, float],
    timestamp_col: str = "timestamp",
    reference_t0: Optional[datetime] = None,
    time_col: str = "t",
) -> pd.DataFrame:
    """
    Apply a lag correction to a DataFrame.

    Shifts timestamps by the specified lag amount and optionally
    recalculates the time column relative to a reference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to correct
    lag : timedelta or float
        Lag to apply (positive = shift forward, negative = shift backward)
        If float, interpreted as seconds
    timestamp_col : str, optional
        Name of timestamp column
    reference_t0 : datetime, optional
        Reference time for recalculating 't' column
    time_col : str, optional
        Name of time column to recalculate

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected timestamps
    """
    df = df.copy()

    # Convert lag to timedelta if needed
    if isinstance(lag, (int, float)):
        lag = timedelta(seconds=lag)

    # Apply lag to timestamps
    df[timestamp_col] = df[timestamp_col] - lag

    # Recalculate time column if reference provided
    if reference_t0 is not None and time_col in df.columns:
        df.drop([time_col], axis=1, inplace=True)
        df[time_col] = df.apply(
            lambda row: (row[timestamp_col] - reference_t0).total_seconds(), axis=1
        )

    return df


# =============================================================================
# Lag computation using cross-correlation
# =============================================================================
def compute_lag(
    tkey: str,
    df1_data: Dict[str, Any],
    df2_data: Dict[str, Any],
    show: bool = False,
    save: bool = False,
    debug: bool = False,
) -> timedelta:
    """
    Compute lag between two time series using cross-correlation.

    Uses scipy's cross-correlation to find the time shift that
    maximizes correlation between two signals.

    Parameters
    ----------
    tkey : str
        Name of time/index column (e.g., "timestamp")
    df1_data : dict
        First series data with keys:
        - "df": DataFrame with data
        - "field": Column name for values
        - "range": Tuple of (start_index, end_index) or None
    df2_data : dict
        Second series data (same format as df1_data)
    show : bool, optional
        Display diagnostic plots
    save : bool, optional
        Save diagnostic plots
    debug : bool, optional
        Print debug information

    Returns
    -------
    timedelta
        Computed lag (positive means df1 leads df2)

    Examples
    --------
    >>> df1_data = {
    ...     "df": df_overview[["timestamp", "Courant_GR1"]],
    ...     "field": "Courant_GR1",
    ...     "range": (0, 100)
    ... }
    >>> lag = compute_lag("timestamp", df1_data, df2_data)
    """
    from scipy.signal import correlate, correlation_lags

    logger.debug("compute_lag: %s - %s", df1_data["field"], df2_data["field"])

    # Extract series 1
    ts1 = df1_data["df"].copy()
    key1 = df1_data["field"]
    istart1, iend1 = df1_data["range"]
    ts1.set_index(tkey, inplace=True)
    ts1 = ts1.iloc[:, 0]  # Get first data column

    # Get time range for ts1
    ts1_index = ts1.index.to_list()
    otstart = ts1_index[istart1] if istart1 else ts1_index[0]
    otend = ts1_index[iend1] if iend1 is not None else ts1_index[-1]

    logger.debug("ts1 range: [%s, %s] -> [%s, %s]", istart1, iend1, otstart, otend)

    # Extract series 2
    ts2 = df2_data["df"].copy()
    key2 = df2_data["field"]
    istart2, iend2 = df2_data["range"]
    ts2.set_index(tkey, inplace=True)
    ts2 = ts2.iloc[:, 0]

    # Resample ts2 to 1s intervals for alignment
    try:
        ts2_index = ts2.index.to_list()
        ts2_resampled = ts2.resample("1s", origin=ts2_index[0]).asfreq()
        ts2_resampled = ts2_resampled.interpolate(method="linear")
    except Exception as e:
        logger.error("Resample error: %s", e)
        raise

    # Determine range for ts2
    if istart2 is None and iend2 is None:
        pstart = ts2_resampled.index.get_indexer(
            [pd.Timestamp(otstart)], method="nearest"
        )
        pend = ts2_resampled.index.get_indexer([pd.Timestamp(otend)], method="nearest")
        ptstart = ts2_resampled.index[pstart[0]]
        ptend = ts2_resampled.index[pend[0]]
    else:
        ts2_idx = ts2.index.to_list()
        ptstart = ts2_idx[istart2] if istart2 else ts2_idx[0]
        ptend = ts2_idx[iend2] if iend2 is not None else ts2_idx[-1]
        pstart = ts2_resampled.index.get_indexer(
            [pd.Timestamp(ptstart)], method="nearest"
        )
        pend = ts2_resampled.index.get_indexer([pd.Timestamp(ptend)], method="nearest")
        ptstart = ts2_resampled.index[pstart[0]]
        ptend = ts2_resampled.index[pend[0]]

    logger.debug("ts2 range: [%s, %s] -> [%s, %s]", istart2, iend2, ptstart, ptend)

    # Call the core lag correlation function
    ts2_data = {
        "field": key2,
        "df": ts2_resampled,
        "range": {"start": ptstart, "end": ptend},
    }
    ts1_data = {
        "field": key1,
        "df": ts1,
        "range": {"start": otstart, "end": otend},
    }

    lag = lag_correlation(ts2_data, ts1_data, show=show, save=save, debug=debug)

    return lag


def lag_correlation(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    show: bool = False,
    save: bool = False,
    debug: bool = False,
) -> timedelta:
    """
    Compute lag using scipy cross-correlation.

    This is the core correlation function that computes the time shift
    between two time series by finding the lag that maximizes their
    cross-correlation.

    Parameters
    ----------
    data1 : dict
        First series with keys: "field", "df", "range" (dict with "start", "end")
    data2 : dict
        Second series (same format)
    show : bool
        Display plots
    save : bool
        Save plots
    debug : bool
        Debug output

    Returns
    -------
    timedelta
        Lag as timedelta
    """
    from scipy.signal import correlate, correlation_lags

    logger.debug("lag_correlation: %s - %s", data1["field"], data2["field"])

    series = data1["df"]
    name_series = data1["field"]
    start_index1 = data1["range"]["start"]
    end_index1 = data1["range"]["end"]

    trend = data2["df"]
    name_trend = data2["field"]
    start_index2 = data2["range"]["start"]
    end_index2 = data2["range"]["end"]

    # Select slice of first series
    if start_index1 != 0 and start_index1 is not None:
        if end_index1 is not None:
            time_series_slice = series[start_index1:end_index1]
        else:
            time_series_slice = series[start_index1:]
    else:
        if end_index1 is not None:
            time_series_slice = series[:end_index1]
        else:
            time_series_slice = series

    # Select slice of second series
    if start_index2 != 0 and start_index2 is not None:
        if end_index2 is not None:
            trend_slice = trend[start_index2:end_index2]
        else:
            trend_slice = trend[start_index2:]
    else:
        if end_index2 is not None:
            trend_slice = trend[:end_index2]
        else:
            trend_slice = trend

    if debug:
        time_series_slice.to_csv("time_series_slice.csv")
        trend_slice.to_csv("trend_slice.csv")

    # Compute cross-correlation
    correlation = correlate(
        time_series_slice - time_series_slice.mean(), trend_slice - trend_slice.mean()
    )
    lags = correlation_lags(time_series_slice.size, trend_slice.size, mode="full")

    # Find lag with maximum correlation
    lag = lags[np.argmax(correlation)]
    time_shift = pd.to_timedelta(f"{lag}s")

    logger.debug(
        'lag_correlation: "%s" vs "%s", lag=%d s', name_series, name_trend, lag
    )

    # Visualization
    if show or save:
        _plot_lag_correlation(
            time_series_slice,
            trend_slice,
            name_series,
            name_trend,
            correlation,
            lags,
            lag,
            show=show,
            save=save,
        )

    return time_shift


def _plot_lag_correlation(
    time_series_slice: pd.Series,
    trend_slice: pd.Series,
    name_series: str,
    name_trend: str,
    correlation: np.ndarray,
    lags: np.ndarray,
    lag: int,
    show: bool = False,
    save: bool = False,
) -> None:
    """Plot lag correlation results."""
    import matplotlib.pyplot as plt

    time_trend_slice_lag = time_series_slice.copy()
    time_shift = pd.to_timedelta(f"{lag}s")
    time_trend_slice_lag.index = time_trend_slice_lag.index - time_shift

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_series_slice, label=name_series, marker="+")
    plt.plot(trend_slice, label=name_trend, color="red", marker="*")
    plt.plot(
        time_trend_slice_lag,
        label=f"{name_series} with lag {lag}s",
        color="green",
        marker="o",
        linestyle="None",
        alpha=0.2,
    )
    plt.title(f"{name_series} and {name_trend}")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(lags, correlation, label="Cross-Correlation")
    plt.axvline(lag, color="red", linestyle="--", label=f"Lag = {lag}")
    plt.title(f"Cross-Correlation between {name_series} and {name_trend}")
    plt.xlabel("Lag")
    plt.ylabel("Cross-Correlation")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig("lag_correlation.png", dpi=300)
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Regime-based matching
# =============================================================================
def compute_regime_score(
    regime: str,
    value: Tuple[float, float],
    time: Tuple[float, float],
    reference_regime: str,
    reference_value: Tuple[float, float],
    reference_time: Tuple[float, float],
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute similarity score between two regimes.

    Compares regime segments based on:
    - Value range similarity
    - Duration similarity
    - Time alignment (lag)

    Parameters
    ----------
    regime : str
        Source regime type ('U', 'P', 'D')
    value : tuple
        (start_value, end_value) for source
    time : tuple
        (start_time, end_time) for source
    reference_regime : str
        Reference regime type
    reference_value : tuple
        (start_value, end_value) for reference
    reference_time : tuple
        (start_time, end_time) for reference

    Returns
    -------
    tuple
        (score, time_score, (start_lag, end_lag))
        Lower scores indicate better matches.
    """
    score = float("inf")
    tscore = float("inf")
    lags = (float("inf"), float("inf"))

    if reference_regime == regime:
        # Compare value ranges
        score = abs(
            abs(value[1] - value[0]) - abs(reference_value[1] - reference_value[0])
        )

        # Compare durations
        tscore = abs(
            abs(time[1] - time[0]) - abs(reference_time[1] - reference_time[0])
        )

        # Compute lags
        start_lag = time[0] - reference_time[0]
        end_lag = time[1] - reference_time[1]
        lags = (start_lag, end_lag)

    return (score, tscore, lags)


def find_best_matching_regime(
    signature: Any,
    reference_signature: Any,
) -> List[RegimeMatch]:
    """
    Find best matching regimes between two signatures.

    For each 'U' (up) or 'D' (down) regime in the source signature,
    finds the best matching regime in the reference signature.

    Parameters
    ----------
    signature : Signature
        Source signature object with regimes, times, values attributes
    reference_signature : Signature
        Reference signature object

    Returns
    -------
    List[RegimeMatch]
        List of best matches for each U/D regime

    Notes
    -----
    This is used to verify synchronization quality by checking
    that regime transitions align between data sources.
    """
    best_matches = []

    for i, regime in enumerate(signature.regimes):
        best_score = float("inf")
        best_lags = (float("inf"), float("inf"))
        best_index = (0, 0)
        best_match = None

        # Only process U (up) and D (down) regimes
        if regime in ["U", "D"] and i <= len(signature.times) - 2:
            values = (signature.values[i], signature.values[i + 1])
            times = (signature.times[i], signature.times[i + 1])

            for j, ref_regime in enumerate(reference_signature.regimes):
                if ref_regime in ["U", "D"] and j <= len(reference_signature.times) - 2:
                    ref_values = (
                        reference_signature.values[j],
                        reference_signature.values[j + 1],
                    )
                    ref_times = (
                        reference_signature.times[j],
                        reference_signature.times[j + 1],
                    )

                    score, tscore, lags = compute_regime_score(
                        regime, values, times, ref_regime, ref_values, ref_times
                    )

                    if score < best_score:
                        best_score = score
                        best_match = ref_regime
                        best_lags = lags
                        best_index = (i, j)

        if best_match is not None:
            best_matches.append(
                RegimeMatch(
                    regime=regime,
                    match_regime=best_match,
                    score=best_score,
                    lags=best_lags,
                    indices=best_index,
                )
            )

    return best_matches


def check_lag_reliability(
    lags: Tuple[float, float],
    duration: float,
    threshold_ratio: float = LAG_THRESHOLD_RATIO,
) -> bool:
    """
    Check if computed lag is reliable.

    A lag is considered unreliable if it's a large fraction of
    the total duration, which may indicate misalignment.

    Parameters
    ----------
    lags : tuple
        (start_lag, end_lag) in seconds
    duration : float
        Total duration in seconds
    threshold_ratio : float, optional
        Maximum acceptable lag/duration ratio (default: 0.2)

    Returns
    -------
    bool
        True if lag is reliable (within threshold)
    """
    if duration <= 0:
        return False

    start_ratio = abs(lags[0]) / duration
    end_ratio = abs(lags[1]) / duration

    is_reliable = start_ratio < threshold_ratio and end_ratio < threshold_ratio

    if not is_reliable:
        logger.warning(
            "Lag may be unreliable: start_ratio=%.2f, end_ratio=%.2f (threshold=%.2f)",
            start_ratio,
            end_ratio,
            threshold_ratio,
        )

    return is_reliable


# =============================================================================
# Convenience functions
# =============================================================================
def compute_simple_timeshift(
    source_t0: datetime,
    target_t0: datetime,
) -> timedelta:
    """
    Compute simple timeshift between two timestamps.

    Parameters
    ----------
    source_t0 : datetime
        Source start time
    target_t0 : datetime
        Target start time

    Returns
    -------
    timedelta
        Timeshift to apply (target - source)
    """
    return target_t0 - source_t0


def add_time_column(
    df: pd.DataFrame,
    reference_t0: datetime,
    timestamp_col: str = "timestamp",
    time_col: str = "t",
) -> pd.DataFrame:
    """
    Add or update time column as seconds from reference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column
    reference_t0 : datetime
        Reference time (t=0)
    timestamp_col : str
        Name of timestamp column
    time_col : str
        Name of time column to create/update

    Returns
    -------
    pd.DataFrame
        DataFrame with time column
    """
    df = df.copy()

    if time_col in df.columns:
        df.drop([time_col], axis=1, inplace=True)

    df[time_col] = df.apply(
        lambda row: (row[timestamp_col] - reference_t0).total_seconds(), axis=1
    )

    return df


def get_timestamp_info(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> Dict[str, Any]:
    """
    Get timestamp information from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column
    timestamp_col : str
        Name of timestamp column

    Returns
    -------
    dict
        Dictionary with t0, t_end, duration, n_samples
    """
    t0 = df[timestamp_col].iloc[0]
    t_end = df[timestamp_col].iloc[-1]
    duration = (t_end - t0).total_seconds()

    return {
        "t0": t0,
        "t_end": t_end,
        "duration": duration,
        "n_samples": len(df),
    }
