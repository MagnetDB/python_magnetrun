"""
Distance metrics and correlation functions for magnetrun analysis.

This module provides tools for comparing time series data:
- Basic distance metrics (Euclidean, MAE, MAPE, Pearson)
- Cross-correlation functions (TLCC, windowed, rolling)
- Dynamic Time Warping (DTW) using dtaidistance

These metrics are used to:
- Compare signals from different data sources
- Validate synchronization quality
- Measure similarity between reference and actual values

Example usage::

    from python_magnetrun.analysis.metrics import (
        calc_euclidean,
        calc_correlation,
        compute_dtw_distance,
        crosscorr,
    )

    # Basic distance metrics
    distance = calc_euclidean(actual, predicted)
    correlation = calc_correlation(actual, predicted)

    # DTW distance
    dtw_result = compute_dtw_distance(series1, series2, window=25)
    print(f"DTW distance: {dtw_result.distance}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Module logger
logger = logging.getLogger("magnetrun.analysis.metrics")


# =============================================================================
# Result dataclasses
# =============================================================================
@dataclass
class DistanceResult:
    """
    Result of distance/similarity computation between two series.

    Attributes
    ----------
    euclidean : float
        Euclidean distance
    mae : float
        Mean Absolute Error
    mape : float
        Mean Absolute Percentage Error
    correlation : float
        Pearson correlation coefficient
    """

    euclidean: float
    mae: float
    mape: float
    correlation: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "euclidean": self.euclidean,
            "mae": self.mae,
            "mape": self.mape,
            "correlation": self.correlation,
        }


@dataclass
class DTWResult:
    """
    Result of Dynamic Time Warping computation.

    Attributes
    ----------
    distance : float
        DTW distance between series
    path : List[Tuple[int, int]]
        Optimal warping path as list of (i, j) index pairs
    similarity_score : float
        Normalized similarity score (distance / path_length)
    normalized_distance : float
        Distance normalized by series length
    """

    distance: float
    path: list[tuple[int, int]] = field(default_factory=list)
    similarity_score: float = 0.0
    normalized_distance: float = 0.0

    @property
    def path_length(self) -> int:
        """Length of the warping path."""
        return len(self.path)


@dataclass
class CorrelationResult:
    """
    Result of correlation analysis.

    Attributes
    ----------
    pearson_r : float
        Pearson correlation coefficient
    p_value : float
        P-value for the correlation test
    lag : int
        Optimal lag (for cross-correlation)
    max_correlation : float
        Maximum correlation value at optimal lag
    """

    pearson_r: float
    p_value: float = 0.0
    lag: int = 0
    max_correlation: float = 0.0


@dataclass
class TLCCResult:
    """
    Result of Time-Lagged Cross Correlation.

    Attributes
    ----------
    correlations : np.ndarray
        Correlation values for each lag
    lags : np.ndarray
        Lag values tested
    optimal_lag : int
        Lag with maximum correlation
    optimal_correlation : float
        Maximum correlation value
    offset : float
        Offset in frames
    """

    correlations: np.ndarray
    lags: np.ndarray
    optimal_lag: int
    optimal_correlation: float
    offset: float


# =============================================================================
# Basic distance metrics
# =============================================================================
def calc_euclidean(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two arrays.

    Parameters
    ----------
    actual : np.ndarray
        Actual/reference values
    predicted : np.ndarray
        Predicted/comparison values

    Returns
    -------
    float
        Euclidean distance (sqrt of sum of squared differences)

    Examples
    --------
    >>> actual = np.array([1.0, 2.0, 3.0])
    >>> predicted = np.array([1.1, 2.1, 3.1])
    >>> calc_euclidean(actual, predicted)
    0.1732...
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.sqrt(np.sum((actual - predicted) ** 2)))


def calc_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error between two arrays.

    Parameters
    ----------
    actual : np.ndarray
        Actual/reference values
    predicted : np.ndarray
        Predicted/comparison values

    Returns
    -------
    float
        Mean Absolute Error

    Examples
    --------
    >>> actual = np.array([1.0, 2.0, 3.0])
    >>> predicted = np.array([1.1, 2.2, 2.9])
    >>> calc_mae(actual, predicted)
    0.133...
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def calc_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error between two arrays.

    Parameters
    ----------
    actual : np.ndarray
        Actual/reference values (must not contain zeros)
    predicted : np.ndarray
        Predicted/comparison values

    Returns
    -------
    float
        Mean Absolute Percentage Error

    Warnings
    --------
    Will produce inf/nan if actual contains zeros.

    Examples
    --------
    >>> actual = np.array([100.0, 200.0, 300.0])
    >>> predicted = np.array([110.0, 190.0, 310.0])
    >>> calc_mape(actual, predicted)
    0.05...
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((actual - predicted) / actual))

    return float(mape)


def calc_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two arrays.

    Parameters
    ----------
    actual : np.ndarray
        Actual/reference values
    predicted : np.ndarray
        Predicted/comparison values

    Returns
    -------
    float
        Pearson correlation coefficient (-1 to 1)

    Examples
    --------
    >>> actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> predicted = np.array([1.1, 2.0, 3.1, 3.9, 5.0])
    >>> calc_correlation(actual, predicted)
    0.998...
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    a_diff = actual - np.mean(actual)
    p_diff = predicted - np.mean(predicted)

    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff**2)) * np.sqrt(np.sum(p_diff**2))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def compute_all_distances(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> DistanceResult:
    """
    Compute all basic distance metrics between two arrays.

    Parameters
    ----------
    actual : np.ndarray
        Actual/reference values
    predicted : np.ndarray
        Predicted/comparison values

    Returns
    -------
    DistanceResult
        Dataclass with all distance metrics

    Examples
    --------
    >>> result = compute_all_distances(actual, predicted)
    >>> print(f"Euclidean: {result.euclidean}, Correlation: {result.correlation}")
    """
    return DistanceResult(
        euclidean=calc_euclidean(actual, predicted),
        mae=calc_mae(actual, predicted),
        mape=calc_mape(actual, predicted),
        correlation=calc_correlation(actual, predicted),
    )


# =============================================================================
# Cross-correlation functions
# =============================================================================
def crosscorr(
    datax: pd.Series,
    datay: pd.Series,
    lag: int = 0,
    wrap: bool = False,
) -> float:
    """
    Compute lag-N cross correlation between two series.

    Shifted data is filled with NaNs (or wrapped if wrap=True).

    Parameters
    ----------
    datax : pd.Series
        First time series
    datay : pd.Series
        Second time series (will be shifted)
    lag : int, optional
        Number of periods to shift datay (default: 0)
    wrap : bool, optional
        If True, wrap shifted values instead of filling with NaN

    Returns
    -------
    float
        Correlation coefficient at the given lag

    Examples
    --------
    >>> x = pd.Series([1, 2, 3, 4, 5])
    >>> y = pd.Series([1.1, 2.0, 3.1, 3.9, 5.0])
    >>> crosscorr(x, y, lag=0)
    0.998...
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def compute_tlcc(
    datax: pd.Series,
    datay: pd.Series,
    seconds: float = 5.0,
    fps: float = 30.0,
) -> TLCCResult:
    """
    Compute Time-Lagged Cross Correlation (TLCC).

    Calculates correlation at multiple lag values to find
    the optimal time alignment between two series.

    Parameters
    ----------
    datax : pd.Series
        First time series
    datay : pd.Series
        Second time series
    seconds : float, optional
        Maximum lag in seconds (default: 5.0)
    fps : float, optional
        Frames/samples per second (default: 30.0)

    Returns
    -------
    TLCCResult
        Result with correlations, lags, and optimal values

    Examples
    --------
    >>> result = compute_tlcc(series1, series2, seconds=5, fps=30)
    >>> print(f"Optimal lag: {result.optimal_lag}, correlation: {result.optimal_correlation}")
    """
    lag_range = int(seconds * fps)
    lags = np.arange(-lag_range, lag_range + 1)

    correlations = np.array([crosscorr(datax, datay, lag) for lag in lags])

    # Find optimal lag
    optimal_idx = np.argmax(correlations)
    optimal_lag = lags[optimal_idx]
    optimal_correlation = correlations[optimal_idx]

    # Calculate offset
    offset = np.floor(len(correlations) / 2) - optimal_idx

    return TLCCResult(
        correlations=correlations,
        lags=lags,
        optimal_lag=int(optimal_lag),
        optimal_correlation=float(optimal_correlation),
        offset=float(offset),
    )


def compute_windowed_tlcc(
    datax: pd.Series,
    datay: pd.Series,
    seconds: float = 5.0,
    fps: float = 30.0,
    n_windows: int = 20,
) -> np.ndarray:
    """
    Compute Windowed Time-Lagged Cross Correlation.

    Splits the data into windows and computes TLCC for each,
    useful for detecting time-varying correlations.

    Parameters
    ----------
    datax : pd.Series
        First time series
    datay : pd.Series
        Second time series
    seconds : float, optional
        Maximum lag in seconds per window
    fps : float, optional
        Frames/samples per second
    n_windows : int, optional
        Number of windows to split data into

    Returns
    -------
    np.ndarray
        2D array of correlations (n_windows x n_lags)
    """
    lag_range = int(seconds * fps)
    samples_per_window = len(datax) // n_windows

    results = []
    for t in range(n_windows):
        start_idx = t * samples_per_window
        end_idx = (t + 1) * samples_per_window

        d1 = datax.iloc[start_idx:end_idx]
        d2 = datay.iloc[start_idx:end_idx]

        rs = [crosscorr(d1, d2, lag) for lag in range(-lag_range, lag_range + 1)]
        results.append(rs)

    return np.array(results)


def compute_rolling_tlcc(
    datax: pd.Series,
    datay: pd.Series,
    seconds: float = 5.0,
    fps: float = 30.0,
    window_size: int = 300,
    step_size: int = 30,
    max_samples: int | None = None,
) -> np.ndarray:
    """
    Compute Rolling Windowed Time-Lagged Cross Correlation.

    Uses a sliding window approach for continuous monitoring
    of correlation over time.

    Parameters
    ----------
    datax : pd.Series
        First time series
    datay : pd.Series
        Second time series
    seconds : float, optional
        Maximum lag in seconds
    fps : float, optional
        Frames/samples per second
    window_size : int, optional
        Window size in samples
    step_size : int, optional
        Step size between windows
    max_samples : int, optional
        Maximum samples to process (default: all)

    Returns
    -------
    np.ndarray
        2D array of correlations (n_windows x n_lags)
    """
    lag_range = int(seconds * fps)

    if max_samples is None:
        max_samples = len(datax)

    results = []
    t_start = 0
    t_end = window_size

    while t_end < max_samples:
        d1 = datax.iloc[t_start:t_end]
        d2 = datay.iloc[t_start:t_end]

        rs = [crosscorr(d1, d2, lag, wrap=False) for lag in range(-lag_range, lag_range + 1)]
        results.append(rs)

        t_start += step_size
        t_end += step_size

    return np.array(results)


def compute_pearson_correlation(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> CorrelationResult:
    """
    Compute Pearson correlation with statistical significance.

    Parameters
    ----------
    x : array-like
        First variable
    y : array-like
        Second variable

    Returns
    -------
    CorrelationResult
        Result with correlation coefficient and p-value
    """
    from scipy import stats

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return CorrelationResult(pearson_r=np.nan, p_value=np.nan)

    r, p = stats.pearsonr(x_clean, y_clean)

    return CorrelationResult(
        pearson_r=float(r),
        p_value=float(p),
    )


# =============================================================================
# Dynamic Time Warping (DTW)
# =============================================================================
def compute_dtw_distance(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int | None = None,
    psi: int | None = None,
    use_c: bool = False,
) -> DTWResult:
    """
    Compute Dynamic Time Warping distance between two series.

    DTW finds the optimal alignment between two time series that
    may vary in speed or timing.

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    window : int, optional
        Sakoe-Chiba band window size (None for no constraint)
    psi : int, optional
        Relaxation parameter for endpoints
    use_c : bool, optional
        Use C library for faster computation (requires compilation)

    Returns
    -------
    DTWResult
        Result with distance, path, and similarity metrics

    Raises
    ------
    ImportError
        If dtaidistance is not installed

    Examples
    --------
    >>> result = compute_dtw_distance(series1, series2, window=25)
    >>> print(f"DTW distance: {result.distance}")
    >>> print(f"Similarity score: {result.similarity_score}")
    """
    try:
        from dtaidistance import dtw
    except ImportError:
        raise ImportError(
            "dtaidistance is required for DTW. Install with: pip install dtaidistance"
        ) from None

    series1 = np.asarray(series1, dtype=np.float64)
    series2 = np.asarray(series2, dtype=np.float64)

    # Compute warping paths
    kwargs = {"use_c": use_c}
    if window is not None:
        kwargs["window"] = window
    if psi is not None:
        kwargs["psi"] = psi

    try:
        distance, paths = dtw.warping_paths(series1, series2, **kwargs)
        best_path = dtw.best_path(paths)
    except MemoryError:
        logger.warning("DTW memory error, trying without full path computation")
        distance = dtw.distance(series1, series2, **kwargs)
        best_path = []

    # Calculate similarity metrics
    path_length = len(best_path) if best_path else max(len(series1), len(series2))
    similarity_score = distance / path_length if path_length > 0 else float("inf")

    max_len = max(len(series1), len(series2))
    normalized_distance = distance / max_len if max_len > 0 else 0.0

    return DTWResult(
        distance=float(distance),
        path=list(best_path) if best_path else [],
        similarity_score=float(similarity_score),
        normalized_distance=float(normalized_distance),
    )


def compute_dtw_distance_fast(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int | None = None,
) -> float:
    """
    Compute DTW distance without path (faster, less memory).

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    window : int, optional
        Sakoe-Chiba band window size

    Returns
    -------
    float
        DTW distance
    """
    try:
        from dtaidistance import dtw
    except ImportError:
        raise ImportError(
            "dtaidistance is required for DTW. Install with: pip install dtaidistance"
        ) from None

    series1 = np.asarray(series1, dtype=np.float64)
    series2 = np.asarray(series2, dtype=np.float64)

    kwargs = {}
    if window is not None:
        kwargs["window"] = window

    return float(dtw.distance(series1, series2, **kwargs))


def dtw_with_paa(
    series1: np.ndarray,
    series2: np.ndarray,
    window_size: int = 125,
    dtw_window: int | None = None,
    psi: int = 2,
) -> DTWResult:
    """
    Compute DTW using Piecewise Aggregate Approximation for large series.

    PAA reduces the dimensionality of time series by averaging segments,
    making DTW feasible for very long series.

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    window_size : int, optional
        PAA window size (default: 125)
    dtw_window : int, optional
        DTW Sakoe-Chiba window (defaults to window_size)
    psi : int, optional
        DTW relaxation parameter

    Returns
    -------
    DTWResult
        Result with distance computed on PAA-reduced series

    Raises
    ------
    ImportError
        If pyts or dtaidistance is not installed
    """
    try:
        from dtaidistance import dtw
        from pyts.approximation import PiecewiseAggregateApproximation
    except ImportError as e:
        raise ImportError(f"Both pyts and dtaidistance are required. Missing: {e.name}") from e

    series1 = np.asarray(series1).reshape(1, -1)
    series2 = np.asarray(series2).reshape(1, -1)

    # Apply PAA transformation
    paa = PiecewiseAggregateApproximation(window_size=window_size, overlapping=False)

    series1_paa = paa.transform(series1)[0]
    series2_paa = paa.transform(series2)[0]

    # Compute DTW on reduced series
    if dtw_window is None:
        dtw_window = window_size

    distance, paths = dtw.warping_paths(series1_paa, series2_paa, window=dtw_window, psi=psi)
    best_path = dtw.best_path(paths)

    path_length = len(best_path) if best_path else 1
    similarity_score = distance / path_length

    return DTWResult(
        distance=float(distance),
        path=list(best_path),
        similarity_score=float(similarity_score),
        normalized_distance=float(distance / len(series1_paa)),
    )


# =============================================================================
# Plotting utilities for metrics
# =============================================================================
def plot_tlcc(
    result: TLCCResult,
    xfield: str,
    yfield: str,
    show: bool = False,
    save: bool = False,
    filename: str | None = None,
) -> None:
    """
    Plot Time-Lagged Cross Correlation results.

    Parameters
    ----------
    result : TLCCResult
        TLCC computation result
    xfield : str
        Name of first field (for labels)
    yfield : str
        Name of second field (for labels)
    show : bool
        Display plot
    save : bool
        Save plot to file
    filename : str, optional
        Output filename (default: "{xfield}-{yfield}-TLCC.png")
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(result.correlations)
    ax.axvline(np.ceil(len(result.correlations) / 2), color="k", linestyle="--", label="Center")
    ax.axvline(np.argmax(result.correlations), color="r", linestyle="--", label="Peak synchrony")
    ax.set(
        title=f"Offset = {result.offset} frames\n{xfield} leads <> {yfield} leads",
        xlabel="Offset",
        ylabel="Pearson r",
    )
    plt.legend()

    if save:
        fname = filename or f"{xfield}-{yfield}-TLCC.png"
        plt.savefig(fname, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_dtw_alignment(
    series1: np.ndarray,
    series2: np.ndarray,
    result: DTWResult,
    label1: str = "Series 1",
    label2: str = "Series 2",
    show: bool = False,
    save: bool = False,
    filename: str | None = None,
) -> None:
    """
    Plot DTW alignment between two series.

    Creates a multi-panel figure showing:
    - Original time series
    - Optimal warping path
    - Point-to-point alignment

    Parameters
    ----------
    series1 : np.ndarray
        First time series
    series2 : np.ndarray
        Second time series
    result : DTWResult
        DTW computation result
    label1 : str
        Label for first series
    label2 : str
        Label for second series
    show : bool
        Display plot
    save : bool
        Save plot to file
    filename : str, optional
        Output filename
    """
    import matplotlib.pyplot as plt

    if not result.path:
        logger.warning("No path available for DTW visualization")
        return

    path = np.array(result.path)

    plt.figure(figsize=(12, 8))

    # Original Time Series Plot
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.plot(series1, label=label1, color="blue")
    ax1.plot(series2, label=label2, linestyle="--", color="orange")
    ax1.set_title("Original Time Series")
    ax1.legend()
    ax1.grid(True)

    # Warping Path Plot
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.plot(path[:, 0], path[:, 1], "green", marker="o", linestyle="-")
    ax2.set_title("Shortest Path (Best Path)")
    ax2.set_xlabel(label1)
    ax2.set_ylabel(label2)
    ax2.grid(True)

    # Point-to-Point Comparison
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax3.plot(series1, label=label1, color="blue", marker="o", markersize=3)
    ax3.plot(series2, label=label2, color="orange", marker="x", linestyle="--", markersize=3)

    # Draw alignment lines (subsample for clarity)
    step = max(1, len(path) // 50)
    for i in range(0, len(path), step):
        a, b = path[i]
        if a < len(series1) and b < len(series2):
            ax3.plot(
                [a, b],
                [series1[a], series2[b]],
                color="grey",
                linestyle="-",
                linewidth=0.5,
                alpha=0.5,
            )

    ax3.set_title(f"Point-to-Point Comparison (DTW distance: {result.distance:.4f})")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()

    if save:
        fname = filename or "dtw_alignment.png"
        plt.savefig(fname, dpi=300)
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Convenience functions
# =============================================================================
def compare_series(
    actual: np.ndarray,
    predicted: np.ndarray,
    compute_dtw: bool = False,
    dtw_window: int | None = 25,
) -> dict[str, Any]:
    """
    Comprehensive comparison of two time series.

    Computes all available metrics including distances,
    correlation, and optionally DTW.

    Parameters
    ----------
    actual : np.ndarray
        Actual/reference series
    predicted : np.ndarray
        Predicted/comparison series
    compute_dtw : bool, optional
        Whether to compute DTW (slower)
    dtw_window : int, optional
        DTW window size if computing

    Returns
    -------
    dict
        Dictionary with all computed metrics
    """
    results = {
        "distances": compute_all_distances(actual, predicted),
        "pearson": compute_pearson_correlation(actual, predicted),
    }

    if compute_dtw:
        try:
            results["dtw"] = compute_dtw_distance(actual, predicted, window=dtw_window)
        except ImportError:
            logger.warning("DTW not available (dtaidistance not installed)")
        except (ValueError, RuntimeError) as e:
            logger.warning("DTW computation failed: %s", e)

    return results
