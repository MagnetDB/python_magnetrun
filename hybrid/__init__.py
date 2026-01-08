"""
Hybrid Data Module

Unified interface for hybrid magnet data from FEPC acquisition systems.

Supports three data types:
- kHz: High-frequency (1 kHz) data from FEPC analog and digital cards
- RMS: Root Mean Square data at lower frequency (typically 10 Hz)
- Trigger: Event-triggered data

Performance features:
- Lazy loading with memory mapping for large binary files
- Automatic downsampling for visualization (uses tsdownsample MinMaxLTTB)
- Data caching to avoid repeated disk reads
- Dedicated outlier detection (separate from plotting)

Two main interfaces:
- HybridData: Low-level data access
- HybridRun: MagnetRun-compatible interface for comparison with pupitre/tdms data

Example usage:
    from hybrid import HybridData, HybridRun

    # Low-level access
    data = HybridData("/data/hybrid", "2025-01-06")
    data.print_summary()
    values, time = data.read_khz_variable("FEPC-LNCMI", "I_H1")

    # MagnetRun-compatible interface (recommended)
    hrun = HybridRun.fromdir("/data/hybrid", "2025-01-06")
    values, time = hrun.getData("kHz/FEPC-LNCMI/I_H1", downsample=10000)

    # Outlier detection (separate from plotting)
    from hybrid.outliers import detect_outliers, OutlierDetector
    result = detect_outliers(values, method='iqr', threshold=1.5)
    clean_values = result.apply_to_data(values, strategy='interpolate')

    # Plot with pre-computed outliers
    from hybrid import plotting
    plotting.plot_khz_variable(
        data, "FEPC-LNCMI", "I_H1",
        outlier_result=result,
        outlier_strategy='highlight',
        downsample=50000
    )

    # Compare with MagnetRun data
    from hybrid.data_protocol import compare_loaders
    result = compare_loaders(hrun, mrun, "kHz/FEPC-LNCMI/I_H1", "IH")

CLI usage:
    python -m hybrid.cli --help
    python -m hybrid.cli --base-dir /data/hybrid --date 2025-01-06
"""

# Direct imports for commonly used items
from .hybrid_data import HybridData, HybridDataInfo
from .hybrid_run import HybridRun, LoadOptions, load_hybrid, downsample_data
from .utils import list_available_dates
from .data_protocol import (
    DataLoader,
    DownsamplingLoader,
    DataSourceType,
    DataInfo,
    load_comparable_data,
    compare_loaders,
)
from .outliers import (
    OutlierDetector,
    OutlierResult,
    OutlierMethod,
    detect_outliers,
    analyze_outliers,
    get_outlier_summary,
    find_outlier_segments,
)

# Sub-modules available via attribute access
# e.g., from hybrid import plotting
# or: import hybrid; hybrid.plotting.plot_khz_variable(...)

__all__ = [
    # Data classes
    "HybridData",
    "HybridDataInfo",
    "HybridRun",
    "LoadOptions",
    # Factory functions
    "load_hybrid",
    # Protocol/interfaces
    "DataLoader",
    "DownsamplingLoader",
    "DataSourceType",
    "DataInfo",
    # Outlier detection
    "OutlierDetector",
    "OutlierResult",
    "OutlierMethod",
    "detect_outliers",
    "analyze_outliers",
    "get_outlier_summary",
    "find_outlier_segments",
    # Utility functions
    "list_available_dates",
    "downsample_data",
    "load_comparable_data",
    "compare_loaders",
]

# Get version from parent package
try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("python_magnetrun")
except PackageNotFoundError:
    __version__ = "0.0.0"
