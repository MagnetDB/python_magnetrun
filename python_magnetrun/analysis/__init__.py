"""
Magnetrun Analysis Module
=========================

Tools for analyzing magnetrun data from TDMS and pupitre files.

This module provides:
- Configuration management for different measurement sites (M8, M9, M10)
- Data loading from TDMS overview, archive, and pupitre files
- Time synchronization between different data sources
- Core data processing logic

Metrics and plotting live in their own sub-namespaces::

    from python_magnetrun.analysis.metrics import calc_mahalanobis, compare_series
    from python_magnetrun.analysis.plotting import plot_data, plot_comparison

Example usage::

    from python_magnetrun.analysis import AnalysisConfig, HousingConfig

    # Get configuration for a specific site
    config = AnalysisConfig.for_housing("M9")

    # Access housing-specific channel mappings
    print(config.housing.reference_gr1_current)  # "IH"
    print(config.housing.reference_gr2_current)  # "IB"

    # Get threshold for a channel
    threshold = config.thresholds.get("Courant_GR1")

Submodules
----------
- config: Configuration dataclasses and constants
- loaders: Data loading and file operations
- synchronization: Time synchronization logic
- metrics: Distance and similarity metrics
- plotting: Visualization functions
- processing: Core data processing logic
- cli: Command-line interface and main entry point
"""

from importlib.metadata import PackageNotFoundError, version

from ..utils.downsampling import (
    DownsampleConfig,
    downsample_arrays,
    downsample_dataframe,
)
from ..utils.timestamps import add_time_columns
from .config import (
    DEFAULT_BINS,
    DEFAULT_DATA_DIR,
    DEFAULT_GROUP,
    DEFAULT_LEVELS,
    DEFAULT_WINDOW_SIZE,
    # Pre-defined configurations
    HOUSING_CONFIGS,
    LAG_THRESHOLD_RATIO,
    SAMPLING_RATE_ARCHIVE,
    SAMPLING_RATE_INCIDENTS,
    # Constants
    SAMPLING_RATE_OVERVIEW,
    AnalysisConfig,
    # Dataclasses
    ChannelMapping,
    HousingConfig,
    ThresholdConfig,
    VoltageChannelMapping,
    # Functions
    get_housing_config,
    get_time_offset,
)
from .loaders import (
    # Constants
    TIMESTAMP_FORMAT,
    # Classes
    FileDiscovery,
    # Dataclasses
    FileMetadata,
    FileSet,
    # Functions
    convert_to_timestamp,
    discover_files,
    load_files_data,
)
from .processing import (
    # Main dataclasses
    OverviewRecord,
    # Configuration
    ProcessingConfig,
    ProcessingResult,
    add_time_column_with_offset,
    # Utilities
    create_overview_dict,
    load_archive_data,
    load_incidents_data,
    # Data loading
    load_overview_data,
    load_pupitre_data,
    print_record_summary,
    process_experiment,
    # Main functions
    process_overview_file,
    summarize_record,
)
from .synchronization import (
    LagResult,
    RegimeMatch,
    # Dataclasses
    SyncResult,
    add_time_column,
    apply_lag_correction,
    check_lag_reliability,
    compute_lag,
    compute_regime_score,
    compute_simple_timeshift,
    find_best_matching_regime,
    get_timestamp_info,
    lag_correlation,
    # Functions
    synchronize_data,
)

# The `cli` submodule is imported lazily below to avoid executing
# the CLI module at package import time (which can trigger
# warnings and circular-import issues when running ``python -m``).

# Names exported from the `cli` submodule that we expose at package level.
_cli_lazy_names = {
    "setup_logging",
    "get_logger",
    "set_log_level",
    "LogConfig",
    "ColoredFormatter",
    "JSONFormatter",
    "ROOT_LOGGER_NAME",
    "ProgressTracker",
    "timed_operation",
    "LogContext",
    "create_argument_parser",
    "parse_arguments",
    "args_to_processing_config",
    "cli_main",
}


def __getattr__(name: str):
    """Lazy-load attributes from the ``cli`` submodule on demand."""
    if name in _cli_lazy_names:
        from importlib import import_module

        _cli = import_module(".cli", __name__)
        # map requested name to actual attribute in cli
        if name == "cli_main":
            return _cli.main
        return getattr(_cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Make the lazy names discoverable in tooling and autocompletion
    return sorted(list(globals().keys()) + list(_cli_lazy_names))


__all__ = [
    # Config constants
    "SAMPLING_RATE_OVERVIEW",
    "SAMPLING_RATE_ARCHIVE",
    "SAMPLING_RATE_INCIDENTS",
    "LAG_THRESHOLD_RATIO",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_BINS",
    "DEFAULT_LEVELS",
    "DEFAULT_DATA_DIR",
    "DEFAULT_GROUP",
    # Config dataclasses
    "ChannelMapping",
    "VoltageChannelMapping",
    "HousingConfig",
    "ThresholdConfig",
    "AnalysisConfig",
    # Pre-defined configurations
    "HOUSING_CONFIGS",
    # Loaders constants
    "TIMESTAMP_FORMAT",
    # Loaders functions
    "convert_to_timestamp",
    "load_files_data",
    "discover_files",
    # Loaders dataclasses
    "FileMetadata",
    "FileSet",
    # Loaders classes
    "FileDiscovery",
    # Synchronization dataclasses
    "SyncResult",
    "LagResult",
    "RegimeMatch",
    # Synchronization functions
    "synchronize_data",
    "apply_lag_correction",
    "compute_lag",
    "lag_correlation",
    "compute_regime_score",
    "find_best_matching_regime",
    "check_lag_reliability",
    "compute_simple_timeshift",
    "add_time_column",
    "add_time_columns",
    "get_timestamp_info",
    # Downsampling (canonical, from utils.downsampling)
    "DownsampleConfig",
    "downsample_arrays",
    "downsample_dataframe",
    # Processing configuration
    "ProcessingConfig",
    # Processing dataclasses
    "OverviewRecord",
    "ProcessingResult",
    # Processing main functions
    "process_overview_file",
    "process_experiment",
    # Processing utilities
    "get_time_offset",
    "add_time_column_with_offset",
    "create_overview_dict",
    "get_housing_config",
    "summarize_record",
    "print_record_summary",
    # Processing data loading
    "load_overview_data",
    "load_archive_data",
    "load_pupitre_data",
    "load_incidents_data",
    # CLI logging setup
    "setup_logging",
    "get_logger",
    "set_log_level",
    "LogConfig",
    "ColoredFormatter",
    "JSONFormatter",
    "ROOT_LOGGER_NAME",
    # CLI progress tracking
    "ProgressTracker",
    "timed_operation",
    "LogContext",
    # CLI argument parsing
    "create_argument_parser",
    "parse_arguments",
    "args_to_processing_config",
    # CLI entry point
    "cli_main",
]

try:
    __version__ = version("python_magnetrun")
except PackageNotFoundError:
    __version__ = "0.0.0"
