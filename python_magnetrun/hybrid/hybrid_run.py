"""
HybridRun - MagnetRun-compatible interface for hybrid magnet data

This module provides a HybridRun class that mirrors the MagnetRun interface,
allowing seamless comparison between hybrid data (kHz, RMS, Trigger) and
traditional pupitre/tdms data.

Performance features:
- Lazy loading: Data is only loaded when accessed
- Memory mapping: Large binary files are memory-mapped to avoid loading into RAM
- Downsampling: Optional automatic downsampling for visualization using tsdownsample
- Caching: Loaded data is cached to avoid repeated disk access

Example usage:
    # Load hybrid data (similar to MagnetRun.fromtdms)
    hrun = HybridRun.fromdir(base_dir, "2025-01-06", fepc_system="FEPC-LNCMI")

    # Access data (similar to MagnetRun.getMData().getData())
    data = hrun.getData("kHz/FEPC-LNCMI/I_H1")

    # Get downsampled data for plotting
    data_ds = hrun.getData("kHz/FEPC-LNCMI/I_H1", downsample=10000)

    # Compare with MagnetRun
    mrun = MagnetRun.fromtdms(site, insert, tdms_file)
    # Both have similar interfaces for getData(), getKeys(), etc.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

# Local imports
from .hybrid_data import HybridData

# Setup logger
logger = logging.getLogger(__name__)

# Try to import tsdownsample for efficient downsampling
try:
    from tsdownsample import MinMaxLTTBDownsampler

    HAS_TSDOWNSAMPLE = True
except ImportError:
    HAS_TSDOWNSAMPLE = False
    logger.debug("tsdownsample not available - downsampling will use simple stride")


class DataProvider(Protocol):
    """Protocol defining common interface for data providers (MagnetRun, HybridRun)"""

    def getData(self, key: str | None = None) -> Any:
        """Get data for a specific key"""
        ...

    def getKeys(self) -> list[str]:
        """Get list of available data keys"""
        ...

    def getType(self) -> int:
        """Get data type identifier"""
        ...


@dataclass
class LoadOptions:
    """Options for data loading and processing"""

    # Loading options
    lazy: bool = True  # Use lazy loading (memory mapping)
    cache: bool = True  # Cache loaded data

    # Downsampling options !!! TODO make option dependant on method !!!
    downsample: int | None = None  # Target number of points (None = no downsampling)
    downsample_method: str = "stride"  # 'minmax_lttb', 'lttb', 'stride'

    # Time range selection
    start_time: datetime | None = None
    end_time: datetime | None = None
    hours: range | list[int] | None = None

    # Calibration
    apply_calib: bool = True
    cnv_dir: str | None = None


@dataclass
class CacheEntry:
    """Cache entry for loaded data"""

    data: np.ndarray
    time: np.ndarray
    loaded_at: datetime
    options: LoadOptions
    size_bytes: int


class LazyArrayLoader:
    """
    Lazy loader for binary files using memory mapping.

    Provides array-like access without loading entire file into memory.
    """

    def __init__(
        self,
        filepath: Path,
        dtype: np.dtype,
        shape: tuple[int, ...],
        offset: int = 0,
        block_size: int = 1614,  # Default for analog blocks
        samples_per_block: int = 50,
        header_size: int = 14,
    ):
        self.filepath = filepath
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        self.block_size = block_size
        self.samples_per_block = samples_per_block
        self.header_size = header_size

        self._mmap: np.memmap | None = None
        self._data: np.ndarray | None = None

    def _create_mmap(self) -> None:
        """Create memory-mapped file"""
        if self._mmap is None:
            # For complex block structures, we need custom reading
            # For simple continuous data, use direct memmap
            file_size = os.path.getsize(self.filepath)
            logger.debug(f"Creating memmap for {self.filepath}, size={file_size}")

    def __getitem__(self, key) -> np.ndarray:  # type: ignore[no-untyped-def]
        """Array-like access with lazy loading"""
        if self._data is None:
            self._load_data()
        return self._data[key]  # type: ignore[index]

    def _load_data(self) -> None:
        """Load data from file (called on first access)"""
        # This will be implemented based on file type
        raise NotImplementedError("Subclasses must implement _load_data")

    @property
    def data(self) -> np.ndarray:
        """Get full data array (loads if needed)"""
        if self._data is None:
            self._load_data()
        return self._data


class LazyKHzLoader(LazyArrayLoader):
    """
    Lazy loader specifically for kHz binary files.

    Supports reading specific time ranges without loading entire file.
    """

    ANALOG_BLOCK_SIZE = 1614
    DIGITAL_BLOCK_SIZE = 212
    SAMPLES_PER_BLOCK = 50
    ANALOG_HEADER_SIZE = 14
    DIGITAL_HEADER_SIZE = 12

    def __init__(
        self,
        filepath: Path,
        card_type: str,  # 'ANA' or 'DIG'
        endian: str = "big",
        channel: int | None = None,  # Load specific channel only
    ):
        self.card_type = card_type
        self.endian = endian
        self.channel = channel

        if card_type == "ANA":
            block_size = self.ANALOG_BLOCK_SIZE
            header_size = self.ANALOG_HEADER_SIZE
            num_channels = 16
            dtype = np.uint16
        else:
            block_size = self.DIGITAL_BLOCK_SIZE
            header_size = self.DIGITAL_HEADER_SIZE
            num_channels = 32
            dtype = np.bool_

        # Calculate shape based on file size
        file_size = os.path.getsize(filepath)
        num_blocks = file_size // block_size
        total_samples = num_blocks * self.SAMPLES_PER_BLOCK

        if channel is not None:
            shape: tuple[int, ...] = (total_samples,)
        else:
            shape = (total_samples, num_channels)

        super().__init__(
            filepath=filepath,
            dtype=np.dtype(dtype),
            shape=shape,
            block_size=block_size,
            samples_per_block=self.SAMPLES_PER_BLOCK,
            header_size=header_size,
        )

        self.num_blocks = num_blocks
        self.total_samples = total_samples
        self.num_channels = num_channels

    def _load_data(self) -> None:
        """Load kHz data using memory-efficient block reading"""
        from .kHz.fepc_reader import read_hour_file

        logger.debug(f"Loading kHz data from {self.filepath}")

        # Use existing reader for full data
        full_data, _timestamps = read_hour_file(
            str(self.filepath),
            self.card_type,
            num_blocks=self.num_blocks,
            endian=self.endian,
        )

        if self.channel is not None:
            self._data = full_data[:, self.channel]
        else:
            self._data = full_data

    def read_range(
        self,
        start_sample: int,
        end_sample: int,
        channel: int | None = None,
    ) -> np.ndarray:
        """
        Read a specific range of samples efficiently.

        This avoids loading the entire file when only a portion is needed.
        """
        # Calculate which blocks we need
        start_block = start_sample // self.SAMPLES_PER_BLOCK
        end_block = (end_sample + self.SAMPLES_PER_BLOCK - 1) // self.SAMPLES_PER_BLOCK

        num_blocks_needed = end_block - start_block
        data_offset = start_sample - (start_block * self.SAMPLES_PER_BLOCK)

        logger.debug(
            f"Reading range [{start_sample}:{end_sample}], "
            f"blocks {start_block}-{end_block}, offset={data_offset}"
        )

        # Allocate output
        num_samples = end_sample - start_sample
        ch = channel if channel is not None else self.channel

        if ch is not None:
            output = np.zeros(num_samples, dtype=self.dtype)
        else:
            output = np.zeros((num_samples, self.num_channels), dtype=self.dtype)

        # Read blocks
        with open(self.filepath, "rb") as f:
            for i, block_idx in enumerate(range(start_block, end_block)):
                f.seek(block_idx * self.block_size)

                # Skip header
                f.read(self.header_size)

                # Read data
                if self.card_type == "ANA":
                    data_size = self.SAMPLES_PER_BLOCK * self.num_channels * 2
                    endian_char = ">" if self.endian == "big" else "<"
                    dtype = np.dtype(np.uint16).newbyteorder(endian_char)
                    block_data = np.frombuffer(f.read(data_size), dtype=dtype).reshape(
                        self.SAMPLES_PER_BLOCK, self.num_channels
                    )
                else:
                    # Digital data - more complex unpacking
                    data_size = self.SAMPLES_PER_BLOCK * 4  # 32 bits per sample
                    block_data = self._read_digital_block_data(f)

                # Copy relevant portion to output
                block_start = i * self.SAMPLES_PER_BLOCK
                block_end = min(
                    block_start + self.SAMPLES_PER_BLOCK, num_samples + data_offset
                )

                src_start = data_offset if i == 0 else 0
                src_end = src_start + (block_end - block_start)

                dst_start = block_start - (data_offset if i == 0 else 0)
                if dst_start < 0:
                    dst_start = 0
                dst_end = dst_start + (src_end - src_start)

                if dst_end > num_samples:
                    src_end -= dst_end - num_samples
                    dst_end = num_samples

                if ch is not None:
                    output[dst_start:dst_end] = block_data[src_start:src_end, ch]
                else:
                    output[dst_start:dst_end, :] = block_data[src_start:src_end, :]

        return output

    def _read_digital_block_data(self, f) -> np.ndarray:
        """Read digital block data and unpack bits"""
        endian_char = ">" if self.endian == "big" else "<"
        dtype = np.dtype(np.uint16).newbyteorder(endian_char)
        data_uint16 = np.frombuffer(
            f.read(self.SAMPLES_PER_BLOCK * 4), dtype=dtype
        ).reshape(self.SAMPLES_PER_BLOCK, 2)

        data = np.zeros((self.SAMPLES_PER_BLOCK, 32), dtype=bool)
        for sample_idx in range(self.SAMPLES_PER_BLOCK):
            word1 = data_uint16[sample_idx, 0]
            for bit in range(16):
                data[sample_idx, bit] = bool((word1 >> bit) & 1)
            word2 = data_uint16[sample_idx, 1]
            for bit in range(16):
                data[sample_idx, 16 + bit] = bool((word2 >> bit) & 1)

        return data


def downsample_data(
    data: np.ndarray,
    time: np.ndarray,
    target_points: int,
    method: str = "stride",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample time series data for visualization.

    Parameters
    ----------
    data : np.ndarray
        Data array to downsample
    time : np.ndarray
        Time array
    target_points : int
        Target number of points after downsampling
    method : str
        Downsampling method: 'minmax_lttb', 'lttb', 'stride', 'minmax'

    Returns
    -------
    tuple
        (downsampled_data, downsampled_time)
    """
    # Strip NaN entries from both arrays before any downsampling algorithm sees them.
    # Missing blocks produce NaN timestamps; those entries are non-plottable and cause
    # most downsampling libraries to fail. Gaps remain visible as jumps on the time axis.
    valid = ~np.isnan(time) & ~np.isnan(data)
    if not np.all(valid):
        n_nan = int(np.sum(~valid))
        logger.debug(
            f"downsample_data: stripping {n_nan} NaN entries before downsampling"
        )
        data = data[valid]
        time = time[valid]

    if len(data) == 0:
        return data, time

    if len(data) <= target_points:
        return data, time

    logger.info(f"Downsampling {len(data)} points to {target_points} using {method}")

    if method == "minmax_lttb" and HAS_TSDOWNSAMPLE:
        # Use tsdownsample MinMaxLTTB (best for visualization)
        downsampler = MinMaxLTTBDownsampler()
        indices = downsampler.downsample(time, data, n_out=target_points)
        return data[indices], time[indices]

    elif method == "lttb" and HAS_TSDOWNSAMPLE:
        from tsdownsample import LTTBDownsampler

        downsampler = LTTBDownsampler()
        indices = downsampler.downsample(time, data, n_out=target_points)
        return data[indices], time[indices]

    elif method == "minmax":
        # Simple min/max downsampling (preserves peaks)
        bucket_size = len(data) // (target_points // 2)
        n_buckets = len(data) // bucket_size

        result_data = []
        result_time = []

        for i in range(n_buckets):
            start = i * bucket_size
            end = start + bucket_size
            bucket_data = data[start:end]
            bucket_time = time[start:end]

            min_idx = np.argmin(bucket_data)
            max_idx = np.argmax(bucket_data)

            # Add min first, then max (or vice versa based on order)
            if min_idx < max_idx:
                result_data.extend([bucket_data[min_idx], bucket_data[max_idx]])
                result_time.extend([bucket_time[min_idx], bucket_time[max_idx]])
            else:
                result_data.extend([bucket_data[max_idx], bucket_data[min_idx]])
                result_time.extend([bucket_time[max_idx], bucket_time[min_idx]])

        return np.array(result_data), np.array(result_time)

    else:
        # Simple stride-based downsampling
        stride = len(data) // target_points
        indices = np.arange(0, len(data), stride)[:target_points]
        return data[indices], time[indices]


class HybridRun:
    """
    MagnetRun-compatible interface for hybrid magnet data.

    Provides the same interface as MagnetRun while supporting:
    - kHz data from FEPC acquisition
    - RMS data
    - Trigger data

    Performance optimizations:
    - Lazy loading with memory mapping
    - Automatic downsampling for visualization
    - Data caching

    Parameters
    ----------
    housing : str
        Housing name (for MagnetRun compatibility)
    site : str
        Site identifier (e.g., 'Hybrid')
    data : HybridData
        HybridData instance
    start_time : datetime, optional
        Start time of the data

    Examples
    --------
    >>> # Create from directory (similar to MagnetRun.fromtdms)
    >>> hrun = HybridRun.fromdir("/data/hybrid", "2025-01-06")
    >>>
    >>> # Get available keys
    >>> print(hrun.getKeys())
    >>>
    >>> # Get data (similar to MagnetRun)
    >>> data = hrun.getData("kHz/FEPC-LNCMI/I_H1")
    >>>
    >>> # Get downsampled data for plotting
    >>> data_ds, time_ds = hrun.getData(
    ...     "kHz/FEPC-LNCMI/I_H1",
    ...     downsample=10000
    ... )
    """

    def __init__(
        self,
        housing: str = "Hybrid",
        site: str = "",
        data: HybridData | None = None,
        start_time: datetime | None = None,
    ):
        self.Housing = housing
        self.Site = site
        self.HybridData = data
        self.StartTime = start_time

        # Cache for loaded data
        self._cache: dict[str, CacheEntry] = {}
        self._cache_max_size_bytes = 1024 * 1024 * 1024  # 1 GB default
        self._cache_size_bytes = 0

        # Lazy loaders
        self._lazy_loaders: dict[str, LazyKHzLoader] = {}

        # Default load options
        self.default_options = LoadOptions()

    @classmethod
    def fromdir(
        cls,
        base_dir: str,
        date_str: str,
        fepc_system: str | None = None,
        endian: str = "big",
        housing: str = "Hybrid",
        site: str = "",
        defs_file: str | None = None,
    ) -> "HybridRun":
        """
        Create HybridRun from a directory for a given date.

        Similar to MagnetRun.fromtdms() factory method.

        Parameters
        ----------
        base_dir : str
            Base directory containing kHz, rms, trigger subdirectories
        date_str : str
            Date string in YYYY-MM-DD format
        fepc_system : str, optional
            FEPC system name: 'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
        endian : str
            Endianness for binary data
        housing : str
            Housing name (for compatibility)
        site : str
            Site identifier

        Returns
        -------
        HybridRun
            HybridRun instance
        """
        data = HybridData(base_dir, date_str, fepc_system, endian, defs_file=defs_file)

        # Determine start time from first available data
        start_time = None
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            start_time = datetime(d.year, d.month, d.day, 0, 0, 0)
        except ValueError:
            pass

        return cls(
            housing=housing,
            site=site or date_str,
            data=data,
            start_time=start_time,
        )

    def __repr__(self):
        if self.HybridData:
            return (
                f"HybridRun(Housing={self.Housing!r}, Site={self.Site!r}, "
                f"date={self.HybridData.date_str}, "
                f"systems={self.HybridData._info.fepc_systems})"
            )
        return f"HybridRun(Housing={self.Housing!r}, Site={self.Site!r})"

    # -------------------------------------------------------------------------
    # MagnetRun-compatible interface
    # -------------------------------------------------------------------------

    def getInsert(self) -> str:
        """Returns Insert (for MagnetRun compatibility)"""
        if self.HybridData:
            return self.HybridData.FileName
        return ""

    def getSite(self) -> str:
        """Returns Site"""
        return self.Site

    def getHousing(self) -> str:
        """Returns Housing"""
        return self.Housing

    def setSite(self, site: str) -> None:
        """Set Site"""
        self.Site = site

    def getType(self) -> int:
        """Returns Data Type"""
        if self.HybridData is not None:
            return self.HybridData.Type
        raise RuntimeError("HybridRun.getType: no HybridData associated")

    def getMData(self) -> HybridData:
        """Return HybridData object (similar to MagnetRun.getMData())"""
        if self.HybridData is not None:
            return self.HybridData
        raise RuntimeError("No HybridData attached to this HybridRun")

    def getData(
        self,
        key: str | None = None,
        hours: range | list[int] | None = None,
        downsample: int | None = None,
        options: LoadOptions | None = None,
    ) -> dict | tuple[np.ndarray, np.ndarray] | pd.DataFrame:
        """
        Get data for a specific key.

        Parameters
        ----------
        key : str, optional
            Data key in format 'type/system/variable' or 'type/system'
            Examples:
            - 'kHz/FEPC-LNCMI/I_H1' - specific kHz variable
            - 'rms/FEPC-LNCMI/I_H1' - specific RMS variable
            - 'kHz/FEPC-LNCMI' - list of available kHz variables
        hours : range or list of int, optional
            Specific hours to load
        downsample : int, optional
            Target number of points for downsampling (for plotting)
        options : LoadOptions, optional
            Additional loading options

        Returns
        -------
        Data
            - For variable keys: tuple of (data_array, time_array)
            - For group keys: dict of available variables
            - For None: all loaded data
        """
        logger.info(
            f"HybridRun.getData: key={key}, hours={hours},downsample={downsample}, options={options}"
        )

        if self.HybridData is None:
            raise RuntimeError("HybridRun.getData: no HybridData associated")

        opts = options or self.default_options
        if hours is not None:
            opts.hours = hours

        if downsample is not None:
            opts = LoadOptions(
                lazy=opts.lazy,
                cache=opts.cache,
                downsample=downsample,
                downsample_method=opts.downsample_method,
                start_time=opts.start_time,
                end_time=opts.end_time,
                hours=opts.hours,
                apply_calib=opts.apply_calib,
                cnv_dir=opts.cnv_dir,
            )
            logger.debug(f"LoadOptions opts: {opts}")
        logger.debug(f"opts: {opts}")

        if key is None:
            return self.HybridData  # .Data

        # Check cache first
        cache_key = f"{key}:{opts.downsample}:{opts.hours}"
        if opts.cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            logger.debug(f"Cache hit for {cache_key}")
            return entry.data, entry.time

        # Parse key
        parts = key.split("/")
        # print("parts:", parts)
        if len(parts) < 2:
            raise ValueError(
                f"Invalid key format: {key}. Expected 'type/system[/variable]'"
            )

        data_type = parts[0]
        logger.debug(f"data_type: {data_type}")
        system = parts[1]
        variable = parts[2] if len(parts) >= 3 else None
        logger.debug(f"system: {system}")
        logger.debug(f"variable: {variable}")

        # Load data based on type
        if data_type == "kHz":
            if variable is None:
                return self.HybridData.get_khz_variables(system)

            data, time = self.HybridData.read_khz_variable(
                system,
                variable,
                hours=opts.hours,
                apply_calib=opts.apply_calib,
                cnv_dir=opts.cnv_dir,
            )

        elif data_type == "rms":
            if variable is None:
                return self.HybridData.get_rms_variables(system)

            data, time = self.HybridData.read_rms_variable(
                system,
                variable,
                hours=opts.hours,
            )

        elif data_type == "trigger":
            return self.HybridData.list_trigger_files(system)

        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Apply downsampling if requested
        if opts.downsample and len(data) > opts.downsample:
            data, time = downsample_data(
                data, time, opts.downsample, opts.downsample_method
            )

        # Cache result
        if opts.cache:
            self._add_to_cache(cache_key, data, time, opts)

        return data, time

    def getKeys(self) -> list[str]:
        """Return list of data keys (MagnetRun compatible)"""
        if self.HybridData is not None:
            return self.HybridData.getKeys()
        raise RuntimeError("HybridRun.getKeys: no HybridData associated")

    def Units(self, debug: bool = False, json_file: str | None = None) -> None:
        """Populate units — delegates to :meth:`HybridData.Units`."""
        if self.HybridData is not None:
            self.HybridData.Units(debug=debug, json_file=json_file)
        else:
            raise RuntimeError("HybridRun.Units: no HybridData associated")

    def getUnit(self, key: str = "") -> tuple:
        """Return Unit for a key"""
        if self.HybridData is not None:
            return self.HybridData.getUnitKey(key)
        raise RuntimeError("HybridRun.getUnit: no HybridData associated")

    def getStats(self, key: str | None = None) -> dict:
        """
        Return basic statistics for a field.

        Parameters
        ----------
        key : str, optional
            Field key (e.g., 'kHz/FEPC-LNCMI/I_H1')

        Returns
        -------
        dict
            Statistics including min, max, mean, std
        """
        if key is None:
            return {}

        result = self.getData(key)
        if isinstance(result, tuple) and len(result) == 2:
            data, time = result
            data = np.asarray(data)
            time = np.asarray(time)
        else:
            return {}

        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "count": len(data),
            "duration_s": float(time[-1] - time[0]) if len(time) > 1 else 0,
        }

    def saveData(self, filename: str, key: str | None = None) -> None:
        """
        Save data to file.

        Parameters
        ----------
        filename : str
            Output filename (CSV format)
        key : str, optional
            Specific key to save (saves all if None)
        """
        if key:
            data, time = self.getData(key)
            df = pd.DataFrame({"time": time, key: data})
            df.to_csv(filename, index=False)
        else:
            raise NotImplementedError("Saving all data not yet implemented")

    # -------------------------------------------------------------------------
    # Cache management
    # -------------------------------------------------------------------------

    def _add_to_cache(
        self,
        key: str,
        data: np.ndarray,
        time: np.ndarray,
        options: LoadOptions,
    ) -> None:
        """Add data to cache with size management"""
        size_bytes = data.nbytes + time.nbytes

        # Evict old entries if needed
        while (
            self._cache_size_bytes + size_bytes > self._cache_max_size_bytes
            and self._cache
        ):
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].loaded_at)
            evicted = self._cache.pop(oldest_key)
            self._cache_size_bytes -= evicted.size_bytes
            logger.debug(f"Evicted cache entry: {oldest_key}")

        # Add new entry
        self._cache[key] = CacheEntry(
            data=data,
            time=time,
            loaded_at=datetime.now(),
            options=options,
            size_bytes=size_bytes,
        )
        self._cache_size_bytes += size_bytes
        logger.debug(f"Cached {key}: {size_bytes / 1024 / 1024:.1f} MB")

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
        self._cache_size_bytes = 0
        logger.info("Cache cleared")

    def set_cache_size(self, size_mb: int) -> None:
        """Set maximum cache size in MB"""
        self._cache_max_size_bytes = size_mb * 1024 * 1024
        logger.info(f"Cache size set to {size_mb} MB")

    # -------------------------------------------------------------------------
    # Convenience methods for comparison with MagnetRun data
    # -------------------------------------------------------------------------

    def get_time_range(self) -> tuple[datetime, datetime]:
        """Get time range of available data"""
        if self.HybridData is None:
            raise RuntimeError("No HybridData associated")

        # Start of day
        start = datetime.combine(self.HybridData.date, datetime.min.time())
        # End of day
        end = start + timedelta(days=1)

        return start, end

    def get_data_at_time(
        self,
        key: str,
        target_time: datetime,
        window_seconds: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get data around a specific time.

        Useful for comparing with pupitre/tdms data at specific timestamps.

        Parameters
        ----------
        key : str
            Data key
        target_time : datetime
            Target timestamp
        window_seconds : float
            Time window around target (± seconds)

        Returns
        -------
        tuple
            (data_array, time_array) for the window
        """
        data, time = self.getData(key)

        # Convert target time to seconds from start of day
        day_start = datetime.combine(self.HybridData.date, datetime.min.time())  # type: ignore[union-attr]
        target_seconds = (target_time - day_start).total_seconds()

        # Find indices within window
        mask = (time >= target_seconds - window_seconds) & (
            time <= target_seconds + window_seconds
        )

        return data[mask], time[mask]

    def compare_with_magnetrun(
        self,
        mrun: Any,  # MagnetRun
        hybrid_key: str,
        magnetrun_key: str,
        downsample: int = 10000,
    ) -> pd.DataFrame:
        """
        Create comparison DataFrame between HybridRun and MagnetRun data.

        Parameters
        ----------
        mrun : MagnetRun
            MagnetRun instance to compare with
        hybrid_key : str
            Key for hybrid data (e.g., 'kHz/FEPC-LNCMI/I_H1')
        magnetrun_key : str
            Key for MagnetRun data (e.g., 'IH')
        downsample : int
            Number of points for comparison

        Returns
        -------
        pd.DataFrame
            Comparison DataFrame with columns: time_hybrid, hybrid_data, time_mr, mr_data
        """
        # Get hybrid data
        h_data, h_time = self.getData(hybrid_key, downsample=downsample)

        # Get MagnetRun data
        mdata = mrun.getMData()
        mr_df = mdata.getData(magnetrun_key)

        # Create comparison DataFrame
        # Note: Time alignment may be needed
        return pd.DataFrame(
            {
                "time_hybrid": h_time,
                "hybrid": h_data,
                # MagnetRun data would need resampling to match
            }
        )


# Factory function for easy creation
def load_hybrid(
    base_dir: str,
    date_str: str,
    fepc_system: str | None = None,
    **kwargs,
) -> HybridRun:
    """
    Convenience function to load hybrid data.

    Parameters
    ----------
    base_dir : str
        Base directory containing kHz, rms, trigger subdirectories
    date_str : str
        Date string in YYYY-MM-DD format
    fepc_system : str, optional
        FEPC system name
    **kwargs
        Additional arguments passed to HybridRun.fromdir

    Returns
    -------
    HybridRun
        HybridRun instance

    Example
    -------
    >>> hrun = load_hybrid("/data/hybrid", "2025-01-06")
    >>> data, time = hrun.getData("kHz/FEPC-LNCMI/I_H1", downsample=10000)
    """
    return HybridRun.fromdir(base_dir, date_str, fepc_system, **kwargs)
