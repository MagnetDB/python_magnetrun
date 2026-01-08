"""
Outlier Detection Module for Hybrid Data

This module provides comprehensive outlier detection functionality,
separated from plotting for flexibility and performance.

Design principles:
- Detection is separate from removal/handling
- Support for both global and local (rolling) detection
- Memory-efficient chunked processing for large datasets
- Returns masks/indices rather than modifying data directly
- Multiple detection algorithms with configurable thresholds

Example usage:
    from hybrid.outliers import OutlierDetector, detect_outliers, get_outlier_summary

    # Simple API
    mask = detect_outliers(data, method='iqr', threshold=1.5)
    clean_data = data[~mask]

    # Object-oriented API for more control
    detector = OutlierDetector(method='mad', threshold=3.5)
    result = detector.detect(data, time)
    print(result.summary())

    # Chunked processing for large datasets
    detector = OutlierDetector(method='zscore', chunk_size=1_000_000)
    result = detector.detect_chunked(data, time)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from enum import Enum, auto

import numpy as np

# Setup logger
logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Available outlier detection methods"""
    IQR = auto()          # Interquartile Range
    ZSCORE = auto()       # Z-score (standard deviations from mean)
    MAD = auto()          # Median Absolute Deviation
    PERCENTILE = auto()   # Percentile-based clipping
    MODIFIED_ZSCORE = auto()  # Modified Z-score using median
    GRUBBS = auto()       # Grubbs' test (single outlier)
    DBSCAN = auto()       # Density-based (requires scipy)


@dataclass
class OutlierResult:
    """
    Result of outlier detection.

    Contains the outlier mask and metadata about the detection.
    """
    mask: np.ndarray  # Boolean mask where True = outlier
    method: str
    threshold: float
    n_outliers: int
    n_total: int
    indices: np.ndarray = field(default_factory=lambda: np.array([]))
    statistics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.indices) == 0:
            self.indices = np.where(self.mask)[0]

    @property
    def outlier_ratio(self) -> float:
        """Ratio of outliers to total points"""
        return self.n_outliers / self.n_total if self.n_total > 0 else 0.0

    @property
    def outlier_percentage(self) -> float:
        """Percentage of outliers"""
        return self.outlier_ratio * 100

    def summary(self) -> str:
        """Return human-readable summary"""
        return (
            f"OutlierResult: {self.n_outliers:,} outliers detected "
            f"({self.outlier_percentage:.2f}%) using {self.method} "
            f"(threshold={self.threshold})"
        )

    def get_clean_mask(self) -> np.ndarray:
        """Return mask for clean (non-outlier) data"""
        return ~self.mask

    def apply_to_data(
        self,
        data: np.ndarray,
        time: Optional[np.ndarray] = None,
        strategy: str = "remove",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply outlier handling to data.

        Parameters
        ----------
        data : np.ndarray
            Input data
        time : np.ndarray, optional
            Time array
        strategy : str
            How to handle outliers:
            - 'remove': Remove outlier points
            - 'nan': Replace with NaN
            - 'interpolate': Replace with interpolated values
            - 'clip': Clip to threshold bounds
            - 'median': Replace with rolling median

        Returns
        -------
        Cleaned data (and time if provided)
        """
        clean_data = data.copy()

        if strategy == "remove":
            clean_data = data[~self.mask]
            if time is not None:
                return clean_data, time[~self.mask]
            return clean_data

        elif strategy == "nan":
            clean_data = clean_data.astype(float)
            clean_data[self.mask] = np.nan

        elif strategy == "interpolate":
            clean_data = clean_data.astype(float)
            clean_data[self.mask] = np.nan
            valid_idx = ~np.isnan(clean_data)
            if np.sum(valid_idx) > 2:
                clean_data = np.interp(
                    np.arange(len(clean_data)),
                    np.arange(len(clean_data))[valid_idx],
                    clean_data[valid_idx]
                )

        elif strategy == "clip":
            if "lower_bound" in self.statistics and "upper_bound" in self.statistics:
                clean_data = np.clip(
                    clean_data,
                    self.statistics["lower_bound"],
                    self.statistics["upper_bound"]
                )

        elif strategy == "median":
            median = np.nanmedian(data[~self.mask])
            clean_data[self.mask] = median

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if time is not None:
            return clean_data, time
        return clean_data


class OutlierDetector:
    """
    Outlier detector with configurable methods and parameters.

    Parameters
    ----------
    method : str or OutlierMethod
        Detection method: 'iqr', 'zscore', 'mad', 'percentile', 'modified_zscore'
    threshold : float
        Detection threshold (meaning depends on method)
    window_size : int, optional
        Rolling window size for local detection (None = global)
    chunk_size : int, optional
        Process data in chunks for memory efficiency

    Examples
    --------
    >>> detector = OutlierDetector(method='iqr', threshold=1.5)
    >>> result = detector.detect(data)
    >>> clean_data = result.apply_to_data(data, strategy='interpolate')
    """

    # Default thresholds for each method
    DEFAULT_THRESHOLDS = {
        "iqr": 1.5,
        "zscore": 3.0,
        "mad": 3.5,
        "percentile": 1.0,  # 1% from each end
        "modified_zscore": 3.5,
        "grubbs": 0.05,  # significance level
    }

    def __init__(
        self,
        method: str = "iqr",
        threshold: Optional[float] = None,
        window_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ):
        self.method = method.lower()
        self.threshold = threshold or self.DEFAULT_THRESHOLDS.get(self.method, 1.5)
        self.window_size = window_size
        self.chunk_size = chunk_size

    def detect(
        self,
        data: np.ndarray,
        time: Optional[np.ndarray] = None,
    ) -> OutlierResult:
        """
        Detect outliers in data.

        Parameters
        ----------
        data : np.ndarray
            Input data array
        time : np.ndarray, optional
            Time array (not used in detection, but returned in result)

        Returns
        -------
        OutlierResult
            Detection result with mask and statistics
        """
        data = np.asarray(data, dtype=np.float64)

        if self.chunk_size and len(data) > self.chunk_size:
            return self._detect_chunked(data, time)

        if self.window_size:
            mask, stats = self._detect_rolling(data)
        else:
            mask, stats = self._detect_global(data)

        return OutlierResult(
            mask=mask,
            method=self.method,
            threshold=self.threshold,
            n_outliers=int(np.sum(mask)),
            n_total=len(data),
            statistics=stats,
        )

    def _detect_global(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Global outlier detection"""
        stats = {}

        if self.method == "iqr":
            q1 = np.nanpercentile(data, 25)
            q3 = np.nanpercentile(data, 75)
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
            mask = (data < lower) | (data > upper)
            stats = {
                "q1": q1, "q3": q3, "iqr": iqr,
                "lower_bound": lower, "upper_bound": upper
            }

        elif self.method == "zscore":
            mean = np.nanmean(data)
            std = np.nanstd(data)
            if std > 0:
                z_scores = np.abs((data - mean) / std)
                mask = z_scores > self.threshold
            else:
                mask = np.zeros(len(data), dtype=bool)
            stats = {
                "mean": mean, "std": std,
                "lower_bound": mean - self.threshold * std,
                "upper_bound": mean + self.threshold * std
            }

        elif self.method == "mad":
            median = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - median))
            if mad > 0:
                # Modified z-score using MAD
                modified_z = 0.6745 * (data - median) / mad
                mask = np.abs(modified_z) > self.threshold
            else:
                mask = np.zeros(len(data), dtype=bool)
            stats = {
                "median": median, "mad": mad,
                "lower_bound": median - self.threshold * mad / 0.6745,
                "upper_bound": median + self.threshold * mad / 0.6745
            }

        elif self.method == "percentile":
            lower = np.nanpercentile(data, self.threshold)
            upper = np.nanpercentile(data, 100 - self.threshold)
            mask = (data < lower) | (data > upper)
            stats = {"lower_bound": lower, "upper_bound": upper}

        elif self.method == "modified_zscore":
            median = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - median))
            if mad > 0:
                modified_z = 0.6745 * np.abs(data - median) / mad
                mask = modified_z > self.threshold
            else:
                mask = np.zeros(len(data), dtype=bool)
            stats = {"median": median, "mad": mad}

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return mask, stats

    def _detect_rolling(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Rolling window outlier detection"""
        n = len(data)
        mask = np.zeros(n, dtype=bool)
        half_window = self.window_size // 2

        # Use vectorized operations where possible
        if self.method in ["zscore", "mad"]:
            # Pre-compute rolling statistics using convolution
            from scipy.ndimage import uniform_filter1d

            if self.method == "zscore":
                rolling_mean = uniform_filter1d(data, self.window_size, mode='nearest')
                rolling_var = uniform_filter1d(data**2, self.window_size, mode='nearest') - rolling_mean**2
                rolling_std = np.sqrt(np.maximum(rolling_var, 0))

                with np.errstate(divide='ignore', invalid='ignore'):
                    z_scores = np.abs((data - rolling_mean) / rolling_std)
                    z_scores[~np.isfinite(z_scores)] = 0
                mask = z_scores > self.threshold

            elif self.method == "mad":
                # For MAD, need to iterate (no simple vectorization)
                for i in range(n):
                    start = max(0, i - half_window)
                    end = min(n, i + half_window + 1)
                    window = data[start:end]
                    median = np.nanmedian(window)
                    mad = np.nanmedian(np.abs(window - median))
                    if mad > 0:
                        modified_z = 0.6745 * abs(data[i] - median) / mad
                        mask[i] = modified_z > self.threshold
        else:
            # IQR and percentile - iterate
            for i in range(n):
                start = max(0, i - half_window)
                end = min(n, i + half_window + 1)
                window = data[start:end]

                if self.method == "iqr":
                    q1 = np.nanpercentile(window, 25)
                    q3 = np.nanpercentile(window, 75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower = q1 - self.threshold * iqr
                        upper = q3 + self.threshold * iqr
                        mask[i] = data[i] < lower or data[i] > upper

                elif self.method == "percentile":
                    lower = np.nanpercentile(window, self.threshold)
                    upper = np.nanpercentile(window, 100 - self.threshold)
                    mask[i] = data[i] < lower or data[i] > upper

        return mask, {"window_size": self.window_size}

    def _detect_chunked(
        self,
        data: np.ndarray,
        time: Optional[np.ndarray] = None,
    ) -> OutlierResult:
        """
        Detect outliers in chunks for memory efficiency.

        For very large datasets, processes data in chunks while
        maintaining consistency at chunk boundaries.
        """
        n = len(data)
        mask = np.zeros(n, dtype=bool)
        all_stats = []

        # Process in chunks with overlap for boundary consistency
        overlap = self.window_size if self.window_size else 1000

        for start in range(0, n, self.chunk_size - overlap):
            end = min(start + self.chunk_size, n)
            chunk_data = data[start:end]

            if self.window_size:
                chunk_mask, stats = self._detect_rolling(chunk_data)
            else:
                chunk_mask, stats = self._detect_global(chunk_data)

            # Handle overlap - don't double-count
            if start > 0:
                chunk_mask[:overlap] = False  # Will be handled by previous chunk

            mask[start:end] |= chunk_mask
            all_stats.append(stats)

            logger.debug(f"Processed chunk {start}-{end}, found {np.sum(chunk_mask)} outliers")

        # Combine statistics
        combined_stats = {"chunks": len(all_stats)}
        if all_stats and "lower_bound" in all_stats[0]:
            combined_stats["lower_bound"] = np.mean([s.get("lower_bound", 0) for s in all_stats])
            combined_stats["upper_bound"] = np.mean([s.get("upper_bound", 0) for s in all_stats])

        return OutlierResult(
            mask=mask,
            method=self.method,
            threshold=self.threshold,
            n_outliers=int(np.sum(mask)),
            n_total=n,
            statistics=combined_stats,
        )


# Convenience functions for simple usage

def detect_outliers(
    data: np.ndarray,
    method: str = "iqr",
    threshold: Optional[float] = None,
    window_size: Optional[int] = None,
) -> np.ndarray:
    """
    Detect outliers and return boolean mask.

    Simple functional API for outlier detection.

    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str
        Detection method: 'iqr', 'zscore', 'mad', 'percentile'
    threshold : float, optional
        Detection threshold (uses default for method if None)
    window_size : int, optional
        Rolling window size (None = global detection)

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates outlier

    Examples
    --------
    >>> mask = detect_outliers(data, method='iqr', threshold=1.5)
    >>> clean_data = data[~mask]
    """
    detector = OutlierDetector(method, threshold, window_size)
    result = detector.detect(data)
    return result.mask


def remove_outliers(
    data: np.ndarray,
    time: np.ndarray,
    method: str = "iqr",
    threshold: float = 1.5,
    window_size: Optional[int] = None,
    strategy: str = "interpolate",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Remove outliers from data (backward compatible with original API).

    Parameters
    ----------
    data : np.ndarray
        Input data
    time : np.ndarray
        Time array
    method : str
        Detection method
    threshold : float
        Detection threshold
    window_size : int, optional
        Rolling window size
    strategy : str
        Handling strategy: 'remove', 'interpolate', 'nan', 'clip', 'median'

    Returns
    -------
    tuple
        (clean_data, clean_time, n_outliers)
    """
    detector = OutlierDetector(method, threshold, window_size)
    result = detector.detect(data, time)

    clean_data, clean_time = result.apply_to_data(data, time, strategy=strategy)

    return clean_data, clean_time, result.n_outliers


def get_outlier_summary(
    data: np.ndarray,
    methods: Optional[List[str]] = None,
) -> Dict[str, OutlierResult]:
    """
    Get outlier summary using multiple methods.

    Useful for comparing different detection approaches.

    Parameters
    ----------
    data : np.ndarray
        Input data
    methods : list of str, optional
        Methods to use (default: ['iqr', 'zscore', 'mad'])

    Returns
    -------
    dict
        Method name -> OutlierResult

    Examples
    --------
    >>> summary = get_outlier_summary(data)
    >>> for method, result in summary.items():
    ...     print(f"{method}: {result.n_outliers} outliers")
    """
    if methods is None:
        methods = ["iqr", "zscore", "mad"]

    results = {}
    for method in methods:
        detector = OutlierDetector(method)
        results[method] = detector.detect(data)

    return results


def find_outlier_segments(
    mask: np.ndarray,
    min_gap: int = 10,
) -> List[Tuple[int, int]]:
    """
    Find contiguous segments of outliers.

    Useful for identifying problematic regions in data.

    Parameters
    ----------
    mask : np.ndarray
        Boolean outlier mask
    min_gap : int
        Minimum gap between segments to consider them separate

    Returns
    -------
    list of tuple
        List of (start, end) indices for each outlier segment
    """
    if not np.any(mask):
        return []

    # Find transitions
    padded = np.concatenate([[False], mask, [False]])
    transitions = np.diff(padded.astype(int))

    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    # Merge close segments
    segments = []
    current_start = starts[0]
    current_end = ends[0]

    for i in range(1, len(starts)):
        if starts[i] - current_end <= min_gap:
            # Merge with current segment
            current_end = ends[i]
        else:
            # Save current segment and start new one
            segments.append((int(current_start), int(current_end)))
            current_start = starts[i]
            current_end = ends[i]

    segments.append((int(current_start), int(current_end)))

    return segments


def analyze_outliers(
    data: np.ndarray,
    time: Optional[np.ndarray] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    Comprehensive outlier analysis.

    Parameters
    ----------
    data : np.ndarray
        Input data
    time : np.ndarray, optional
        Time array
    method : str
        Detection method
    threshold : float
        Detection threshold

    Returns
    -------
    dict
        Comprehensive analysis results
    """
    detector = OutlierDetector(method, threshold)
    result = detector.detect(data, time)

    segments = find_outlier_segments(result.mask)

    analysis = {
        "method": method,
        "threshold": threshold,
        "n_total": result.n_total,
        "n_outliers": result.n_outliers,
        "outlier_percentage": result.outlier_percentage,
        "statistics": result.statistics,
        "n_segments": len(segments),
        "segments": segments,
    }

    if result.n_outliers > 0:
        outlier_values = data[result.mask]
        analysis["outlier_stats"] = {
            "min": float(np.min(outlier_values)),
            "max": float(np.max(outlier_values)),
            "mean": float(np.mean(outlier_values)),
            "std": float(np.std(outlier_values)),
        }

        if time is not None:
            outlier_times = time[result.mask]
            analysis["outlier_time_stats"] = {
                "first": float(outlier_times[0]),
                "last": float(outlier_times[-1]),
            }

    return analysis
