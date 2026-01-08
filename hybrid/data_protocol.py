"""
Data Loader Protocol - Common interface for different data sources

This module defines a common interface (Protocol) that can be implemented by:
- MagnetRun (pupitre .txt and .tdms files)
- HybridRun (FEPC kHz, RMS, trigger data)

This allows writing code that works with any data source without coupling
to a specific implementation.

Example:
    from hybrid.data_protocol import DataLoader, load_and_compare

    # Both MagnetRun and HybridRun implement DataLoader protocol
    mrun: DataLoader = MagnetRun.fromtdms(site, insert, file)
    hrun: DataLoader = HybridRun.fromdir(base_dir, date_str)

    # Generic function that works with any DataLoader
    def analyze_current(loader: DataLoader, key: str) -> dict:
        data = loader.getData(key)
        return {"mean": data.mean(), "max": data.max()}
"""

from typing import (
    Protocol,
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    runtime_checkable,
)
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd


class DataSourceType(Enum):
    """Enumeration of data source types"""

    PUPITRE = auto()  # .txt files from pupitre
    TDMS = auto()  # .tdms files from pigbrother
    KHZ = auto()  # kHz binary files from FEPC
    RMS = auto()  # RMS files from FEPC
    TRIGGER = auto()  # Trigger data from FEPC
    CSV = auto()  # Generic CSV files
    UNKNOWN = auto()


@dataclass
class DataInfo:
    """
    Metadata about a data source.

    Provides information about the data without loading it.
    """

    source_type: DataSourceType
    filename: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    sample_rate_hz: Optional[float] = None
    num_samples: Optional[int] = None
    num_channels: Optional[int] = None
    keys: List[str] = field(default_factory=list)

    @property
    def estimated_size_mb(self) -> Optional[float]:
        """Estimate data size in MB"""
        if self.num_samples and self.num_channels:
            # Assume float32 (4 bytes per value)
            return (self.num_samples * self.num_channels * 4) / (1024 * 1024)
        return None


@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol defining common interface for data loaders.

    Any class that implements these methods can be used interchangeably
    where a DataLoader is expected.

    Implementations:
    - MagnetRun (python_magnetrun.MagnetRun)
    - HybridRun (hybrid.hybrid_run.HybridRun)
    """

    def getData(self, key: Optional[str] = None) -> Any:
        """
        Get data for a specific key.

        Parameters
        ----------
        key : str, optional
            Data key (e.g., 'IH', 'kHz/FEPC-LNCMI/I_H1')

        Returns
        -------
        Data (format depends on implementation)
        """
        ...

    def getKeys(self) -> List[str]:
        """
        Get list of available data keys.

        Returns
        -------
        List[str]
            Available keys
        """
        ...

    def getType(self) -> int:
        """
        Get data type identifier.

        Returns
        -------
        int
            Type code (0=pandas, 1=tdms, 2=ensight, 3=hybrid)
        """
        ...

    def getSite(self) -> str:
        """Get site identifier"""
        ...

    def getHousing(self) -> str:
        """Get housing identifier"""
        ...


@runtime_checkable
class DownsamplingLoader(Protocol):
    """
    Extended protocol for loaders that support downsampling.

    This is useful for large datasets where visualization doesn't
    need all data points.
    """

    def getData(
        self,
        key: Optional[str] = None,
        downsample: Optional[int] = None,
    ) -> Any:
        """
        Get data with optional downsampling.

        Parameters
        ----------
        key : str, optional
            Data key
        downsample : int, optional
            Target number of points (None = no downsampling)

        Returns
        -------
        Data
        """
        ...


@runtime_checkable
class TimeSeriesLoader(Protocol):
    """
    Protocol for loaders that provide time series data.

    Provides additional methods for time-based queries.
    """

    def get_time_range(self) -> Tuple[datetime, datetime]:
        """Get start and end time of data"""
        ...

    def get_data_at_time(
        self,
        key: str,
        target_time: datetime,
        window_seconds: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get data around a specific time"""
        ...


# Utility functions for working with any DataLoader


def get_data_info(loader: DataLoader) -> DataInfo:
    """
    Extract DataInfo from any DataLoader implementation.

    Parameters
    ----------
    loader : DataLoader
        Any object implementing DataLoader protocol

    Returns
    -------
    DataInfo
        Metadata about the data source
    """
    # Determine source type
    loader_type = type(loader).__name__

    if loader_type == "MagnetRun":
        mdata = loader.getMData() if hasattr(loader, "getMData") else None
        if mdata:
            source_type = (
                DataSourceType.TDMS if mdata.Type == 1 else DataSourceType.PUPITRE
            )
            filename = mdata.FileName
        else:
            source_type = DataSourceType.UNKNOWN
            filename = ""
    elif loader_type == "HybridRun":
        source_type = DataSourceType.KHZ  # Primary type
        hdata = loader.getMData() if hasattr(loader, "getMData") else None
        filename = hdata.FileName if hdata else ""
    else:
        source_type = DataSourceType.UNKNOWN
        filename = ""

    return DataInfo(
        source_type=source_type,
        filename=filename,
        keys=loader.getKeys() if hasattr(loader, "getKeys") else [],
    )


def load_comparable_data(
    loader: DataLoader,
    key: str,
    target_points: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from any loader in a comparable format.

    Normalizes data to (data_array, time_array) format with optional
    downsampling for efficient comparison.

    Parameters
    ----------
    loader : DataLoader
        Any DataLoader implementation
    key : str
        Data key
    target_points : int
        Target number of points (for large datasets)

    Returns
    -------
    tuple
        (data_array, time_array)
    """
    # Check if loader supports downsampling
    if isinstance(loader, DownsamplingLoader):
        result = loader.getData(key, downsample=target_points)
        if isinstance(result, tuple):
            return result

    # Standard getData
    result = loader.getData(key)

    # Handle different return types
    if isinstance(result, tuple) and len(result) == 2:
        data, time = result
    elif isinstance(result, pd.DataFrame):
        # Assume first column is time or index is time
        if "t" in result.columns:
            time = result["t"].values
            data = result.drop(columns=["t"]).values.flatten()
        elif "timestamp" in result.columns:
            time = result["timestamp"].values
            data = result.drop(columns=["timestamp"]).values.flatten()
        else:
            time = np.arange(len(result))
            data = result.values.flatten()
    elif isinstance(result, pd.Series):
        time = np.arange(len(result))
        data = result.values
    elif isinstance(result, np.ndarray):
        time = np.arange(len(result))
        data = result
    else:
        raise TypeError(f"Unexpected data type: {type(result)}")

    # Ensure numpy arrays
    data = np.asarray(data)
    time = np.asarray(time)

    # Apply simple downsampling if needed and loader doesn't support it
    if len(data) > target_points:
        stride = len(data) // target_points
        indices = np.arange(0, len(data), stride)[:target_points]
        data = data[indices]
        time = time[indices]

    return data, time


def align_time_series(
    data1: Tuple[np.ndarray, np.ndarray],
    data2: Tuple[np.ndarray, np.ndarray],
    method: str = "interpolate",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two time series for comparison.

    Parameters
    ----------
    data1 : tuple
        (data_array, time_array) for first series
    data2 : tuple
        (data_array, time_array) for second series
    method : str
        Alignment method: 'interpolate', 'nearest', 'common'

    Returns
    -------
    tuple
        (aligned_data1, aligned_data2, common_time)
    """
    d1, t1 = data1
    d2, t2 = data2

    # Find common time range
    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())

    # Filter to common range
    mask1 = (t1 >= t_start) & (t1 <= t_end)
    mask2 = (t2 >= t_start) & (t2 <= t_end)

    t1_filtered = t1[mask1]
    d1_filtered = d1[mask1]
    t2_filtered = t2[mask2]
    d2_filtered = d2[mask2]

    if method == "interpolate":
        # Use the denser time grid as reference
        if len(t1_filtered) >= len(t2_filtered):
            common_time = t1_filtered
            aligned_d1 = d1_filtered
            aligned_d2 = np.interp(common_time, t2_filtered, d2_filtered)
        else:
            common_time = t2_filtered
            aligned_d1 = np.interp(common_time, t1_filtered, d1_filtered)
            aligned_d2 = d2_filtered

    elif method == "nearest":
        # Use nearest neighbor matching
        common_time = t1_filtered
        aligned_d1 = d1_filtered
        indices = np.searchsorted(t2_filtered, common_time)
        indices = np.clip(indices, 0, len(d2_filtered) - 1)
        aligned_d2 = d2_filtered[indices]

    elif method == "common":
        # Only keep points that exist in both (within tolerance)
        # This is more complex and may result in fewer points
        common_time = t1_filtered
        aligned_d1 = d1_filtered
        aligned_d2 = np.interp(common_time, t2_filtered, d2_filtered)

    else:
        raise ValueError(f"Unknown alignment method: {method}")

    return aligned_d1, aligned_d2, common_time


def compare_loaders(
    loader1: DataLoader,
    loader2: DataLoader,
    key1: str,
    key2: str,
    target_points: int = 10000,
) -> Dict[str, Any]:
    """
    Compare data from two loaders.

    Parameters
    ----------
    loader1, loader2 : DataLoader
        Data loaders to compare
    key1, key2 : str
        Keys for each loader
    target_points : int
        Number of points for comparison

    Returns
    -------
    dict
        Comparison results including correlation, RMSE, etc.
    """
    # Load data
    d1, t1 = load_comparable_data(loader1, key1, target_points)
    d2, t2 = load_comparable_data(loader2, key2, target_points)

    # Align time series
    aligned_d1, aligned_d2, common_time = align_time_series((d1, t1), (d2, t2))

    # Calculate statistics
    diff = aligned_d1 - aligned_d2

    return {
        "loader1_type": type(loader1).__name__,
        "loader2_type": type(loader2).__name__,
        "key1": key1,
        "key2": key2,
        "n_points": len(common_time),
        "time_range": (float(common_time.min()), float(common_time.max())),
        "correlation": float(np.corrcoef(aligned_d1, aligned_d2)[0, 1]),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_diff": float(np.max(np.abs(diff))),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "loader1_stats": {
            "min": float(d1.min()),
            "max": float(d1.max()),
            "mean": float(d1.mean()),
        },
        "loader2_stats": {
            "min": float(d2.min()),
            "max": float(d2.max()),
            "mean": float(d2.mean()),
        },
    }
