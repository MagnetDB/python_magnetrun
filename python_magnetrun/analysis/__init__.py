"""
Magnetrun Analysis Module
=========================

Tools for analyzing magnetrun data from TDMS and pupitre files.

This module provides:
- Configuration management for different measurement sites (M8, M9, M10)
- Data loading from TDMS overview, archive, and pupitre files
- Time synchronization between different data sources
- Distance and similarity metrics (DTW alignment)
- Visualization utilities
- Core data processing logic

Example usage::

    from python_magnetrun.analysis import AnalysisConfig, SiteConfig

    # Get configuration for a specific site
    config = AnalysisConfig.for_site("M9")

    # Access site-specific channel mappings
    print(config.site.reference_gr1_current)  # "IH"
    print(config.site.reference_gr2_current)  # "IB"

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

from .config import (
    # Constants
    SAMPLING_RATE_OVERVIEW,
    SAMPLING_RATE_ARCHIVE,
    SAMPLING_RATE_INCIDENTS,
    LAG_THRESHOLD_RATIO,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_BINS,
    DEFAULT_LEVELS,
    DEFAULT_DATA_DIR,
    DEFAULT_GROUP,
    # Dataclasses
    ChannelMapping,
    VoltageChannelMapping,
    SiteConfig,
    ThresholdConfig,
    AnalysisConfig,
    # Pre-defined configurations
    SITE_CONFIGS,
    # Functions
    get_site_config,
)

from .loaders import (
    # Constants
    TIMESTAMP_FORMAT,
    # Functions
    convert_to_timestamp,
    extract_data,
    find_files,
    select_files,
    load_df,
    load_data,
    merge_data,
    discover_files,
    # Dataclasses
    FileMetadata,
    FileSet,
    # Classes
    FileDiscovery,
)

from .synchronization import (
    # Dataclasses
    SyncResult,
    LagResult,
    RegimeMatch,
    # Functions
    synchronize_data,
    apply_lag_correction,
    compute_lag,
    lag_correlation,
    compute_regime_score,
    find_best_matching_regime,
    check_lag_reliability,
    compute_simple_timeshift,
    add_time_column,
    get_timestamp_info,
)

from .metrics import (
    # Dataclasses
    DistanceResult,
    DTWResult,
    CorrelationResult,
    TLCCResult,
    # Distance functions
    calc_euclidean,
    calc_mae,
    calc_mape,
    calc_correlation,
    compute_all_distances,
    # Cross-correlation functions
    crosscorr,
    compute_tlcc,
    compute_windowed_tlcc,
    compute_rolling_tlcc,
    compute_pearson_correlation,
    # DTW functions
    compute_dtw_distance,
    compute_dtw_distance_fast,
    dtw_with_paa,
    # Plotting
    plot_tlcc,
    plot_dtw_alignment,
    # Convenience
    compare_series,
)

from .plotting import (
    # Dataclasses
    PlotStyle,
    PlotColors,
    # Constants
    DEFAULT_STYLE,
    DEFAULT_COLORS,
    # Downsampling functions
    downsample_for_plot,
    downsample_dataframe,
    downsample_minmax,
    estimate_downsample_percent,
    # Main plotting functions
    plot_data,
    plot_comparison,
    plot_regimes,
    plot_incidents_markers,
    plot_time_series,
    # Utilities
    setup_matplotlib_defaults,
    create_figure_grid,
    save_figure,
)

from .processing import (
    # Configuration
    ProcessingConfig,
    # Main dataclasses
    OverviewRecord,
    ProcessingResult,
    # Main functions
    process_overview_file,
    process_experiment,
    # Utilities
    compute_time_offset,
    add_time_column_with_offset,
    create_overview_dict,
    summarize_record,
    print_record_summary,
    # Data loading
    load_overview_data,
    load_archive_data,
    load_pupitre_data,
    load_incidents_data,
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
            return getattr(_cli, "main")
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
    "SiteConfig",
    "ThresholdConfig",
    "AnalysisConfig",
    # Pre-defined configurations
    "SITE_CONFIGS",
    # Loaders constants
    "TIMESTAMP_FORMAT",
    # Loaders functions
    "convert_to_timestamp",
    "extract_data",
    "find_files",
    "select_files",
    "load_df",
    "load_data",
    "merge_data",
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
    "get_timestamp_info",
    # Metrics dataclasses
    "DistanceResult",
    "DTWResult",
    "CorrelationResult",
    "TLCCResult",
    # Metrics distance functions
    "calc_euclidean",
    "calc_mae",
    "calc_mape",
    "calc_correlation",
    "compute_all_distances",
    # Metrics cross-correlation functions
    "crosscorr",
    "compute_tlcc",
    "compute_windowed_tlcc",
    "compute_rolling_tlcc",
    "compute_pearson_correlation",
    # Metrics DTW functions
    "compute_dtw_distance",
    "compute_dtw_distance_fast",
    "dtw_with_paa",
    # Metrics plotting
    "plot_tlcc",
    "plot_dtw_alignment",
    # Metrics convenience
    "compare_series",
    # Plotting dataclasses
    "PlotStyle",
    "PlotColors",
    # Plotting constants
    "DEFAULT_STYLE",
    "DEFAULT_COLORS",
    # Plotting downsampling functions
    "downsample_for_plot",
    "downsample_dataframe",
    "downsample_minmax",
    "estimate_downsample_percent",
    # Plotting main functions
    "plot_data",
    "plot_comparison",
    "plot_regimes",
    "plot_incidents_markers",
    "plot_time_series",
    # Plotting utilities
    "setup_matplotlib_defaults",
    "create_figure_grid",
    "save_figure",
    # Processing configuration
    "ProcessingConfig",
    # Processing dataclasses
    "OverviewRecord",
    "ProcessingResult",
    # Processing main functions
    "process_overview_file",
    "process_experiment",
    # Processing utilities
    "compute_time_offset",
    "add_time_column_with_offset",
    "create_overview_dict",
    "get_site_config",
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

__version__ = "0.1.0"
