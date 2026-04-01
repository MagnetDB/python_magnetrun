"""
Command-line interface and logging infrastructure for magnetrun analysis.

This module provides:
- Comprehensive logging configuration with console and file handlers
- Colored console output (optional)
- Detailed error logging with file, line, and function information
- Exception logging utilities (log_exception, format_exception_location)
- JSON structured logging (optional)
- Progress tracking utilities
- Timing context managers
- Command-line argument parsing
- Main entry point for the analysis workflow

Usage::

    python -m python_magnetrun.analysis.cli input1.tdms input2.tdms --show --save

    # With verbose logging:
    python -m python_magnetrun.analysis.cli input.tdms --debug --log-file analysis.log

    # With JSON logging:
    python -m python_magnetrun.analysis.cli input.tdms --json-log analysis.json

Example programmatic usage::

    from python_magnetrun.analysis.cli import (
        setup_logging,
        log_exception,
        format_exception_location,
        LogContext,
        timed_operation,
        ProgressTracker,
    )

    # Setup logging
    logger = setup_logging(debug=True, log_file="analysis.log")

    # Exception handling with detailed logging
    try:
        risky_operation()
    except Exception as e:
        log_exception("Operation failed", e, logger)
        # Or get just the location
        location = format_exception_location()
        logger.error(f"Error at {location}: {e}")

    # Use timing context
    with timed_operation("Loading data"):
        df = load_data(files)

    # Track progress
    tracker = ProgressTracker(total=100, description="Processing files")
    for i in range(100):
        # ... do work ...
        tracker.update()
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from natsort import natsorted

from .config import (
    DEFAULT_BINS,
    DEFAULT_DATA_DIR,
    DEFAULT_LEVELS,
    DEFAULT_PIGBROTHER_DATA_DIR,
    DEFAULT_WINDOW_SIZE,
)
from .processing import print_record_summary, process_overview_file

# =============================================================================
# Logging configuration
# =============================================================================

# Root logger for the analysis module
ROOT_LOGGER_NAME = "magnetrun.analysis"

# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Compact format for console
COMPACT_FORMAT = "%(levelname)-8s | %(message)s"

# Detailed format for file logging
DETAILED_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)

# ANSI color codes for console output
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset
}


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds ANSI colors to log levels.

    Only applies colors when output is a terminal.
    For ERROR and CRITICAL levels, includes file, line, and function information.
    """

    def __init__(
        self,
        fmt: str = DEFAULT_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
        use_colors: bool = True,
        detailed_errors: bool = True,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
        self.detailed_errors = detailed_errors
        # Detailed format for errors and critical messages
        self.detailed_fmt = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(filename)s:%(lineno)d:%(funcName)s | %(message)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        # Use detailed format for ERROR and CRITICAL levels if enabled
        if self.detailed_errors and record.levelno >= logging.ERROR:
            # Create a temporary formatter with detailed format
            detailed_formatter = logging.Formatter(
                fmt=self.detailed_fmt, datefmt=DEFAULT_DATE_FORMAT
            )
            # Format with detailed info
            if self.use_colors and record.levelname in COLORS:
                original_levelname = record.levelname
                record.levelname = (
                    f"{COLORS[record.levelname]}{record.levelname}{COLORS['RESET']}"
                )
                result = detailed_formatter.format(record)
                record.levelname = original_levelname
                return result
            return detailed_formatter.format(record)

        # Regular formatting for other levels
        if self.use_colors and record.levelname in COLORS:
            # Store original levelname
            original_levelname = record.levelname
            record.levelname = (
                f"{COLORS[record.levelname]}{record.levelname}{COLORS['RESET']}"
            )
            result = super().format(record)
            # Restore original
            record.levelname = original_levelname
            return result
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs log records as JSON lines.

    Useful for structured logging and log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


@dataclass
class LogConfig:
    """
    Configuration for logging setup.

    Attributes
    ----------
    level : int
        Log level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
    console : bool
        Enable console logging
    console_format : str
        Format string for console output
    use_colors : bool
        Enable colored console output
    log_file : Path or None
        Path to log file (None to disable file logging)
    file_format : str
        Format string for file output
    json_file : Path or None
        Path to JSON log file (None to disable)
    propagate : bool
        Propagate to parent loggers
    """

    level: int = logging.INFO
    console: bool = True
    console_format: str = COMPACT_FORMAT
    use_colors: bool = True
    log_file: Path | None = None
    file_format: str = DETAILED_FORMAT
    json_file: Path | None = None
    propagate: bool = False


def setup_logging(
    debug: bool = False,
    log_file: str | Path | None = None,
    json_file: str | Path | None = None,
    use_colors: bool = True,
    quiet: bool = False,
    config: LogConfig | None = None,
) -> logging.Logger:
    """
    Configure logging for the analysis module.

    Parameters
    ----------
    debug : bool, optional
        If True, set log level to DEBUG; otherwise INFO
    log_file : str or Path, optional
        Path to log file (enables file logging)
    json_file : str or Path, optional
        Path to JSON log file (enables structured logging)
    use_colors : bool, optional
        Enable colored console output (default: True)
    quiet : bool, optional
        If True, only show warnings and errors (default: False)
    config : LogConfig, optional
        Full logging configuration (overrides other parameters)

    Returns
    -------
    logging.Logger
        Configured root logger for the analysis module

    Examples
    --------
    >>> # Basic setup
    >>> logger = setup_logging(debug=True)
    >>> logger.info("Analysis started")

    >>> # With file logging
    >>> logger = setup_logging(log_file="analysis.log", json_file="analysis.json")
    """
    # Build config from parameters if not provided
    if config is None:
        if quiet:
            level = logging.WARNING
        elif debug:
            level = logging.DEBUG
        else:
            level = logging.INFO

        config = LogConfig(
            level=level,
            console=True,
            use_colors=use_colors,
            log_file=Path(log_file) if log_file else None,
            json_file=Path(json_file) if json_file else None,
        )

    # Get the root logger for the analysis module
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(config.level)
    logger.propagate = config.propagate

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    if config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)

        if config.use_colors:
            formatter = ColoredFormatter(
                fmt=config.console_format,
                datefmt=DEFAULT_DATE_FORMAT,
                use_colors=True,
            )
        else:
            formatter = logging.Formatter(
                fmt=config.console_format,
                datefmt=DEFAULT_DATE_FORMAT,
            )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (text)
    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(config.log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_handler.setFormatter(
            logging.Formatter(
                fmt=config.file_format,
                datefmt=DEFAULT_DATE_FORMAT,
            )
        )
        logger.addHandler(file_handler)

        logger.debug("Logging to file: %s", config.log_file)

    # JSON file handler
    if config.json_file:
        config.json_file.parent.mkdir(parents=True, exist_ok=True)

        json_handler = logging.FileHandler(config.json_file, mode="a")
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

        logger.debug("JSON logging to file: %s", config.json_file)

    return logger


def get_logger(name: str = "") -> logging.Logger:
    """
    Get a logger for a specific module.

    Parameters
    ----------
    name : str, optional
        Module name (appended to root logger name)

    Returns
    -------
    logging.Logger
        Logger instance

    Examples
    --------
    >>> logger = get_logger("processing")
    >>> # Returns logger named "magnetrun.analysis.processing"
    """
    if name:
        return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    return logging.getLogger(ROOT_LOGGER_NAME)


def set_log_level(level: int | str) -> None:
    """
    Set the log level for all analysis loggers.

    Parameters
    ----------
    level : int or str
        Log level (e.g., logging.DEBUG, "DEBUG", "INFO")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)


# =============================================================================
# Exception logging utilities
# =============================================================================


def log_exception(
    logger: logging.Logger,
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
    import traceback

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
    import traceback

    exc_type, exc_value, exc_tb = sys.exc_info()

    if exc_tb is not None:
        tb = traceback.extract_tb(exc_tb)
        if tb:
            frame = tb[-1]
            filename = Path(frame.filename).name
            return f"{filename}:{frame.lineno}:{frame.name}"

    return "unknown:?:?"


# =============================================================================
# Progress tracking
# =============================================================================


@dataclass
class ProgressTracker:
    """
    Simple progress tracker with logging output.

    Attributes
    ----------
    total : int
        Total number of items to process
    description : str
        Description of the operation
    log_interval : int
        How often to log progress (every N items)

    Examples
    --------
    >>> tracker = ProgressTracker(total=100, description="Processing files")
    >>> for item in items:
    ...     process(item)
    ...     tracker.update()
    >>> tracker.finish()
    """

    total: int
    description: str = "Processing"
    log_interval: int = 10

    current: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.time, init=False)
    _logger: logging.Logger = field(default=None, init=False)

    def __post_init__(self):
        self._logger = get_logger()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def percent(self) -> float:
        """Completion percentage."""
        return (self.current / self.total) * 100 if self.total > 0 else 0

    @property
    def rate(self) -> float:
        """Items per second."""
        elapsed = self.elapsed
        return self.current / elapsed if elapsed > 0 else 0

    @property
    def eta(self) -> float:
        """Estimated time remaining in seconds."""
        rate = self.rate
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else float("inf")

    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n

        if self.current % self.log_interval == 0 or self.current == self.total:
            self._logger.info(
                "%s: %d/%d (%.1f%%) - %.1f/s - ETA: %.1fs",
                self.description,
                self.current,
                self.total,
                self.percent,
                self.rate,
                self.eta,
            )

    def finish(self) -> None:
        """Mark progress as complete and log summary."""
        self._logger.info(
            "%s: Complete - %d items in %.2fs (%.1f/s)",
            self.description,
            self.total,
            self.elapsed,
            self.rate,
        )


@contextmanager
def timed_operation(
    description: str,
    logger: logging.Logger | None = None,
    log_start: bool = True,
) -> Generator[dict, None, None]:
    """
    Context manager for timing operations.

    Parameters
    ----------
    description : str
        Description of the operation
    logger : logging.Logger, optional
        Logger to use (defaults to module logger)
    log_start : bool, optional
        Whether to log at start (default: True)

    Yields
    ------
    dict
        Dictionary that will contain 'elapsed' after completion

    Examples
    --------
    >>> with timed_operation("Loading data") as timing:
    ...     df = pd.read_csv("data.csv")
    >>> print(f"Took {timing['elapsed']:.2f}s")
    """
    log = logger or get_logger()

    if log_start:
        log.info("%s...", description)

    result = {}
    start_time = time.time()

    try:
        yield result
    finally:
        elapsed = time.time() - start_time
        result["elapsed"] = elapsed
        log.info("%s completed in %.2fs", description, elapsed)


class LogContext:
    """
    Context manager for adding extra context to log records.

    Examples
    --------
    >>> with LogContext(file="data.tdms", site="M9"):
    ...     logger.info("Processing file")  # Will include context in JSON logs
    """

    def __init__(self, **context: Any):
        self.context = context
        self.old_factory = None

    def __enter__(self) -> LogContext:
        self.old_factory = logging.getLogRecordFactory()
        context = self.context
        old_factory = self.old_factory

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.extra_data = context
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
        return False


# =============================================================================
# Argument parsing
# =============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the analysis CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Analyze magnetrun data from TDMS and pupitre files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s M9_Overview_*.tdms --show
      Process all M9 overview files and display plots

  %(prog)s input.tdms --save --debug --log-file analysis.log
      Process with debug logging to file

  %(prog)s input.tdms --synchronize --lag --downsample 10
      Synchronize data, compute lag, plot 10%% of points
        """,
    )

    # Required arguments
    parser.add_argument(
        "input_file",
        nargs="+",
        type=Path,
        help="Input TDMS overview files to process",
    )

    # Data directories
    dir_group = parser.add_argument_group("Data directories")
    dir_group.add_argument(
        "--pupitre-datadir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        metavar="DIR",
        help="Directory containing pupitre data files",
    )
    dir_group.add_argument(
        "--pigbrother-datadir",
        type=Path,
        default=Path(DEFAULT_PIGBROTHER_DATA_DIR),
        metavar="DIR",
        help="Directory containing pigbrother data files",
    )
    dir_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Directory for output files",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing options")
    proc_group.add_argument(
        "--tkey",
        type=str,
        choices=["t", "timestamp"],
        default="t",
        help="Time column to use for plotting",
    )
    proc_group.add_argument(
        "--synchronize",
        action="store_true",
        help="Synchronize pupitre clock with overview",
    )
    proc_group.add_argument(
        "--lag",
        action="store_true",
        help="Compute lag correlation between sources",
    )
    proc_group.add_argument(
        "--distance",
        action="store_true",
        help="Compute distance/DTW metrics between series",
    )
    proc_group.add_argument(
        "--flow-params",
        action="store_true",
        help="Compute flow parameters",
    )
    proc_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files but don't load/process data",
    )

    # Analysis parameters
    param_group = parser.add_argument_group("Analysis parameters")
    param_group.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        metavar="N",
        help="Number of bins for histograms",
    )
    param_group.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        metavar="N",
        help="Rolling window size for smoothing",
    )
    param_group.add_argument(
        "--levels",
        type=int,
        default=DEFAULT_LEVELS,
        metavar="N",
        help="Number of levels for piecewise fitting",
    )
    param_group.add_argument(
        "--downsample",
        type=float,
        default=100.0,
        metavar="PERCENT",
        help="Percentage of data points to plot (1-100)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (requires X11)",
    )
    output_group.add_argument(
        "--save",
        action="store_true",
        help="Save plots to PNG files",
    )

    # Logging options
    log_group = parser.add_argument_group("Logging options")
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    log_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show warnings and errors",
    )
    log_group.add_argument(
        "--log-file",
        type=Path,
        metavar="FILE",
        help="Write logs to file",
    )
    log_group.add_argument(
        "--json-log",
        type=Path,
        metavar="FILE",
        help="Write structured JSON logs to file",
    )
    log_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored console output",
    )

    return parser


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    args : list of str, optional
        Arguments to parse (defaults to sys.argv[1:])

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = create_argument_parser()
    return parser.parse_args(args)


def args_to_processing_config(args: argparse.Namespace):
    """
    Convert parsed arguments to ProcessingConfig.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    ProcessingConfig
        Configuration for processing
    """
    from .processing import ProcessingConfig

    return ProcessingConfig(
        pupitre_datadir=str(args.pupitre_datadir),
        pigbrother_datadir=str(args.pigbrother_datadir),
        tkey=args.tkey,
        synchronize=args.synchronize,
        compute_lag=args.lag,
        compute_distance=args.distance,
        compute_flow_params=getattr(args, "flow_params", False),
        levels=args.levels,
        dry_run=args.dry_run,
        debug=args.debug,
        show=args.show,
        save=args.save,
        downsample_percent=args.downsample,
    )


# =============================================================================
# Main entry point
# =============================================================================


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the analysis CLI.

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments (for testing)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors)
    """
    parsed_args = parse_arguments(args)

    # Setup logging
    logger = setup_logging(
        debug=parsed_args.debug,
        log_file=parsed_args.log_file,
        json_file=parsed_args.json_log,
        use_colors=not parsed_args.no_color,
        quiet=parsed_args.quiet,
    )

    logger.info("Starting magnetrun analysis")
    logger.debug(f"Arguments: {parsed_args}")
    logger.debug(f"input_file: {parsed_args.input_file}")  # noqa: F823

    try:
        from .config import get_site_config
        from .metrics import (
            calc_correlation,
            calc_euclidean,
            calc_mape,
            compute_dtw_distance,
        )
        from .plotting import (
            estimate_downsample_percent,
            plot_data,
        )

        # Convert args to processing config
        config = args_to_processing_config(parsed_args)

        # TODO if '*' in args.input_file
        #
        # Sort input files naturally
        input_files = natsorted(parsed_args.input_file)
        logger.debug(f"input_files: {input_files}")
        logger.info("Processing %d input files", len(input_files))

        # Create output directory
        if parsed_args.save:
            parsed_args.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each file
        tracker = ProgressTracker(
            total=len(input_files),
            description="Processing files",
            log_interval=1,
        )

        results = []
        for input_file in input_files:
            with timed_operation(f"Processing {Path(input_file).name}", logger):
                try:
                    record = process_overview_file(input_file, config)
                    print_record_summary(record)

                    results.append(record)
                    logger.info(
                        f"Processed {record.filename}: site={record.site}, duration={record.duration:.1f}s, has_pupitre={record.has_data('pupitre')}"
                    )

                    # Skip further processing if dry run
                    if config.dry_run:
                        continue

                    # Get site config for channel mappings
                    site_config = get_site_config(record.site)

                    # Build channel dictionaries for plotting
                    channels_dict = {
                        "Référence_GR1": "Courant_GR1",
                        "Référence_GR2": "Courant_GR2",
                    }
                    pupitre_dict = {
                        record.site: {
                            "Référence_GR1": site_config.reference_gr1_current,
                            "Référence_GR2": site_config.reference_gr2_current,
                        }
                    }

                    # Get DataFrames
                    df_overview = record.get_overview()
                    print(f"df_overview columns: {df_overview.columns.tolist()}")
                    df_archive = record.get_archive()
                    print(f"df_archive columns: {df_archive.columns.tolist()}")
                    df_pupitre = record.get_pupitre()
                    print(f"df_pupitre columns: {df_pupitre.columns.tolist()}")
                    df_incidents = record.get_incidents()
                    for key, _dfs in df_incidents.items():
                        print(f"df_incidents[{key}]:")
                        for _df in _dfs:
                            print(f"columns: {_df.columns.tolist()}")
                    logger.info("get database done")
                    logger.info(f"df_archive: {df_archive.head()}")

                    # Determine keys to analyze
                    keys = [
                        "Référence_GR1",
                        "Référence_GR2",
                    ]

                    # Process each key
                    for key in keys:
                        if key not in df_overview.columns:
                            logger.warning("Key %s not found in overview data", key)
                            continue

                        pupitre_key = pupitre_dict[record.site].get(key)

                        # === PLOTTING ===
                        if parsed_args.show or parsed_args.save:
                            with timed_operation(f"Plotting {key}", logger):
                                # Estimate downsampling if not specified
                                downsample_pct = parsed_args.downsample
                                if downsample_pct == 100.0 and len(df_overview) > 10000:
                                    downsample_pct = estimate_downsample_percent(
                                        len(df_overview), target_points=10000
                                    )
                                    logger.info(
                                        "Auto-downsampling to %.1f%% for plotting",
                                        downsample_pct,
                                    )

                                # Determine output path
                                output_path = None
                                if parsed_args.save:
                                    output_path = (
                                        parsed_args.output_dir
                                        / f"{record.filename}_{key.replace('Courant_', '')}.png"
                                    )

                                # Sync message
                                msg = "(nosync)"
                                if (
                                    config.synchronize
                                    and "timeshift_seconds" in record.sync_info
                                ):
                                    shift = record.sync_info["timeshift_seconds"]
                                    msg = f"(sync: {shift:.2f}s)"

                                # Create plot
                                plot_data(
                                    df_overview=df_overview,
                                    df_archive=df_archive,
                                    df_pupitre=df_pupitre,
                                    df_incidents=df_incidents,
                                    channels_dict=channels_dict,
                                    pupitre_dict=pupitre_dict,
                                    site=record.site,
                                    tkey=parsed_args.tkey,
                                    key=key,
                                    title=record.filename,
                                    msg=msg,
                                    show=parsed_args.show,
                                    save=parsed_args.save,
                                    output_path=(
                                        str(output_path) if output_path else None
                                    ),
                                    downsample_percent=downsample_pct,
                                )

                                if output_path:
                                    logger.info("Saved plot to %s", output_path)

                        # === DISTANCE METRICS ===
                        if (
                            parsed_args.distance
                            and pupitre_key
                            and record.has_data("pupitre")
                        ):
                            if pupitre_key in df_pupitre.columns:
                                with timed_operation(
                                    f"Computing metrics for {key}", logger
                                ):
                                    # Get aligned time series
                                    # Use overview as reference
                                    series1 = df_overview[key].values

                                    # Resample pupitre to match overview length
                                    import numpy as np

                                    pupitre_values = df_pupitre[pupitre_key].values
                                    if len(pupitre_values) != len(series1):
                                        # Simple resampling by interpolation
                                        x_orig = np.linspace(0, 1, len(pupitre_values))
                                        x_new = np.linspace(0, 1, len(series1))
                                        series2 = np.interp(
                                            x_new, x_orig, pupitre_values
                                        )
                                    else:
                                        series2 = pupitre_values

                                    # Compute metrics
                                    euclidean = calc_euclidean(series1, series2)
                                    mape = calc_mape(series1, series2)
                                    correlation = calc_correlation(series1, series2)

                                    logger.info(
                                        "Metrics for %s vs %s: "
                                        "Euclidean=%.4f, MAPE=%.2f%%, Correlation=%.4f",
                                        key,
                                        pupitre_key,
                                        euclidean.value,
                                        mape.value,
                                        correlation.value,
                                    )

                                    # Store in record
                                    record.metrics[key] = {
                                        "euclidean": euclidean.value,
                                        "mape": mape.value,
                                        "correlation": correlation.value,
                                    }

                                    # DTW (can be slow for large datasets)
                                    if len(series1) <= 5000:
                                        dtw_result = compute_dtw_distance(
                                            series1, series2
                                        )
                                        # print(dtw_result.distance)           # The DTW distance
                                        # print(dtw_result.path)               # The warping path
                                        # print(dtw_result.normalized_distance) # Normalized by length
                                        # print(dtw_result.similarity_score)    # Distance per path step
                                        logger.info(
                                            "DTW distance for %s: %.4f",
                                            key,
                                            dtw_result.similarity_score,
                                        )
                                        record.metrics[key][
                                            "dtw"
                                        ] = dtw_result.similarity_score
                                    else:
                                        logger.info(
                                            "Skipping DTW for %s (dataset too large: %d points)",
                                            key,
                                            len(series1),
                                        )
                            else:
                                logger.warning(
                                    "Pupitre key %s not found for distance metrics",
                                    pupitre_key,
                                )

                except (OSError, ValueError, KeyError, RuntimeError) as e:
                    logger.error("Failed to process %s: %s", input_file, e)
                    if parsed_args.debug:
                        logger.exception("Full traceback:")

            tracker.update()

        tracker.finish()

        # === FINAL SUMMARY ===
        successful = len(results)
        failed = len(input_files) - successful

        logger.info("Analysis complete: %d successful, %d failed", successful, failed)

        # Print metrics summary if computed
        if parsed_args.distance and results:
            logger.info("=== Metrics Summary ===")
            for record in results:
                if record.metrics:
                    logger.info("File: %s", record.filename)
                    for key, metrics in record.metrics.items():
                        logger.info(
                            "  %s: Euclidean=%.4f, MAPE=%.2f%%, Corr=%.4f%s",
                            key,
                            metrics.get("euclidean", 0),
                            metrics.get("mape", 0),
                            metrics.get("correlation", 0),
                            f", DTW={metrics['dtw']:.4f}" if "dtw" in metrics else "",
                        )

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except (ImportError, OSError, RuntimeError) as e:
        logger.exception("Analysis failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
