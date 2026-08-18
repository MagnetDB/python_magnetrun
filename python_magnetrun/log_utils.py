"""Shared logging setup for python_magnetrun entry points.

Library modules should only do:
    logger = logging.getLogger(__name__)

Entry points (if __name__ == "__main__" blocks and CLI main() functions)
call setup_logging() once to configure the root handler.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Root logger name — all python_magnetrun.* loggers are children of this
ROOT_LOGGER_NAME = "python_magnetrun"

# Standard format: timestamp + module name + level + message
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Compact format for console (no timestamp)
COMPACT_FORMAT = "%(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"

# Detailed format for file logging
DETAILED_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)

# Lightweight format for scripts that don't need timestamps
SIMPLE_FORMAT = "%(levelname)s: %(message)s"

# Bare format: message only, useful for user-facing CLI output
BARE_FORMAT = "%(message)s"

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
    """Formatter that adds ANSI colors to log levels.

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
        self.detailed_fmt = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(filename)s:%(lineno)d:%(funcName)s | %(message)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        if self.detailed_errors and record.levelno >= logging.ERROR:
            detailed_formatter = logging.Formatter(
                fmt=self.detailed_fmt, datefmt=DEFAULT_DATE_FORMAT
            )
            if self.use_colors and record.levelname in COLORS:
                original_levelname = record.levelname
                record.levelname = (
                    f"{COLORS[record.levelname]}{record.levelname}{COLORS['RESET']}"
                )
                result = detailed_formatter.format(record)
                record.levelname = original_levelname
                return result
            return detailed_formatter.format(record)

        if self.use_colors and record.levelname in COLORS:
            original_levelname = record.levelname
            record.levelname = (
                f"{COLORS[record.levelname]}{record.levelname}{COLORS['RESET']}"
            )
            result = super().format(record)
            record.levelname = original_levelname
            return result
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON lines.

    Useful for structured logging and log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        return json.dumps(log_data)


@dataclass
class LogConfig:
    """Configuration for logging setup.

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
    level: int | str = logging.INFO,
    fmt: str = COMPACT_FORMAT,
    log_file: str | Path | None = None,
    *,
    debug: bool = False,
    json_file: str | Path | None = None,
    use_colors: bool = True,
    quiet: bool = False,
    config: LogConfig | None = None,
) -> logging.Logger:
    """Configure logging for python_magnetrun entry points.

    Backward-compatible with the original setup_logging(level, fmt, log_file) API.
    Extended with debug, json_file, use_colors, quiet, and config parameters.

    Calls logging.captureWarnings(True) so warnings.warn() messages are routed
    through the logging system.

    Parameters
    ----------
    level : int or str, optional
        Logging level (e.g. logging.DEBUG, "DEBUG", logging.INFO).
    fmt : str, optional
        Format string for console log records. Defaults to COMPACT_FORMAT.
    log_file : str or Path, optional
        Path to log file (enables file logging).
    debug : bool, optional
        If True, set log level to DEBUG (overrides level).
    json_file : str or Path, optional
        Path to JSON structured log file (enables structured logging).
    use_colors : bool, optional
        Enable colored console output (default: True).
    quiet : bool, optional
        If True, only show warnings and errors (overrides level).
    config : LogConfig, optional
        Full logging configuration (overrides all other parameters).

    Returns
    -------
    logging.Logger
        Configured root logger for the package.
    """
    logging.captureWarnings(True)

    if config is None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        # debug/quiet override level
        if quiet:
            level = logging.WARNING
        elif debug:
            level = logging.DEBUG

        config = LogConfig(
            level=level,
            console=True,
            console_format=fmt,
            use_colors=use_colors,
            log_file=Path(log_file) if log_file else None,
            json_file=Path(json_file) if json_file else None,
        )

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(config.level)
    logger.propagate = config.propagate
    logger.handlers.clear()

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

    if config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(fmt=config.file_format, datefmt=DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(file_handler)
        logger.debug(f"Logging to file: {config.log_file}")

    if config.json_file:
        config.json_file.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(config.json_file, mode="a")
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)
        logger.debug(f"JSON logging to file: {config.json_file}")

    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Get a logger for a specific module.

    Parameters
    ----------
    name : str, optional
        Module name (appended to root logger name).

    Returns
    -------
    logging.Logger
        Logger named "{ROOT_LOGGER_NAME}.{name}", or ROOT_LOGGER_NAME if empty.
    """
    if name:
        return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    return logging.getLogger(ROOT_LOGGER_NAME)


def set_log_level(level: int | str) -> None:
    """Set the log level for all python_magnetrun loggers.

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


def log_exception(
    logger: logging.Logger,
    message: str,
    exception: Exception,
    logger_instance: logging.Logger | None = None,
    use_print: bool = False,
    include_traceback: bool = True,
) -> None:
    """Log exception with traceback information.

    Parameters
    ----------
    logger : logging.Logger
        Fallback logger to use if logger_instance is None.
    message : str
        Custom error message to display.
    exception : Exception
        The exception that was caught.
    logger_instance : logging.Logger, optional
        Logger instance to use. If None, uses logger or print.
    use_print : bool
        If True and logger_instance is None, uses print instead of logger.
    include_traceback : bool
        If True, includes full traceback. Otherwise just file, line, and function.
    """
    import traceback

    exc_type, exc_value, exc_tb = sys.exc_info()

    if include_traceback:
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        error_msg = f"{message}: {exception}\n{''.join(tb_lines)}"
    else:
        if exc_tb is not None:
            tb = traceback.extract_tb(exc_tb)
            if tb:
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

    if logger_instance:
        logger_instance.error(error_msg)
    elif use_print:
        print(error_msg)
    else:
        logger.error(error_msg)


def format_exception_location(exception: Exception | None = None) -> str:
    """Get a concise string with file:line:function where exception occurred.

    Parameters
    ----------
    exception : Exception, optional
        The exception (not used, kept for API consistency).

    Returns
    -------
    str
        Formatted string like "file.py:123:function_name".
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


@dataclass
class ProgressTracker:
    """Simple progress tracker with logging output.

    Attributes
    ----------
    total : int
        Total number of items to process.
    description : str
        Description of the operation.
    log_interval : int
        How often to log progress (every N items).
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
                f"{self.description}: {self.current}/{self.total} ({self.percent:.1f}%) - {self.rate:.1f}/s - ETA: {self.eta:.1f}s"
            )

    def finish(self) -> None:
        """Mark progress as complete and log summary."""
        self._logger.info(
            f"{self.description}: Complete - {self.total} items in {self.elapsed:.2f}s ({self.rate:.1f}/s)"
        )


@contextmanager
def timed_operation(
    description: str,
    logger: logging.Logger | None = None,
    log_start: bool = True,
) -> Generator[dict, None, None]:
    """Context manager for timing operations.

    Parameters
    ----------
    description : str
        Description of the operation.
    logger : logging.Logger, optional
        Logger to use (defaults to package root logger).
    log_start : bool, optional
        Whether to log at start (default: True).

    Yields
    ------
    dict
        Dictionary that will contain 'elapsed' after completion.
    """
    log = logger or get_logger()
    if log_start:
        log.info(f"{description}...")
    result = {}
    start_time = time.time()
    try:
        yield result
    finally:
        elapsed = time.time() - start_time
        result["elapsed"] = elapsed
        log.info(f"{description} completed in {elapsed:.2f}s")


class LogContext:
    """Context manager for adding extra context to log records.

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
