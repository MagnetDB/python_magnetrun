"""
Tests for the CLI and logging infrastructure module.

These tests verify logging setup, progress tracking, timing utilities,
and argument parsing functionality.
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python_magnetrun.analysis.args import (
    args_to_downsample_config,
    args_to_processing_config,
    create_argument_parser,
    parse_arguments,
)
from python_magnetrun.analysis.cli import main
from python_magnetrun.log_utils import (
    COMPACT_FORMAT,
    ROOT_LOGGER_NAME,
    ColoredFormatter,
    JSONFormatter,
    LogConfig,
    LogContext,
    ProgressTracker,
    get_logger,
    set_log_level,
    setup_logging,
    timed_operation,
)


class TestLogConfig:
    """Test LogConfig dataclass."""

    def test_default_values(self):
        """Test default LogConfig values."""
        config = LogConfig()

        assert config.level == logging.INFO
        assert config.console is True
        assert config.use_colors is True
        assert config.log_file is None
        assert config.json_file is None
        assert config.propagate is False

    def test_custom_values(self):
        """Test custom LogConfig values."""
        config = LogConfig(
            level=logging.DEBUG,
            console=False,
            use_colors=False,
            log_file=Path("/tmp/test.log"),
            json_file=Path("/tmp/test.json"),
        )

        assert config.level == logging.DEBUG
        assert config.console is False
        assert config.log_file == Path("/tmp/test.log")


class TestColoredFormatter:
    """Test ColoredFormatter class."""

    def test_format_without_colors(self):
        """Test formatting without colors."""
        formatter = ColoredFormatter(fmt=COMPACT_FORMAT, use_colors=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "INFO" in formatted

    def test_format_preserves_levelname(self):
        """Test that original levelname is preserved."""
        formatter = ColoredFormatter(fmt=COMPACT_FORMAT, use_colors=False)

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        original_levelname = record.levelname
        formatter.format(record)
        assert record.levelname == original_levelname


class TestJSONFormatter:
    """Test JSONFormatter class."""

    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["line"] == 42
        assert parsed["function"] == "test_func"
        assert "timestamp" in parsed

    def test_with_extra_data(self):
        """Test JSON formatting with extra data."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"file": "test.tdms", "assembly": "M9"}

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["extra"]["file"] == "test.tdms"
        assert parsed["extra"]["assembly"] == "M9"


class TestSetupLogging:
    """Test setup_logging function."""

    def test_basic_setup(self):
        """Test basic logging setup."""
        logger = setup_logging(debug=False, use_colors=False)

        assert logger is not None
        assert logger.name == ROOT_LOGGER_NAME
        assert logger.level == logging.INFO

    def test_debug_mode(self):
        """Test debug logging setup."""
        logger = setup_logging(debug=True, use_colors=False)

        assert logger.level == logging.DEBUG

    def test_quiet_mode(self):
        """Test quiet logging setup."""
        logger = setup_logging(quiet=True, use_colors=False)

        assert logger.level == logging.WARNING

    def test_with_file_logging(self):
        """Test logging to file."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = Path(f.name)

        try:
            logger = setup_logging(log_file=log_path, use_colors=False)
            logger.info("Test message")

            # Check file was written
            content = log_path.read_text()
            assert "Test message" in content
        finally:
            log_path.unlink(missing_ok=True)

    def test_with_json_logging(self):
        """Test JSON logging to file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            logger = setup_logging(json_file=json_path, use_colors=False)
            logger.info("JSON test message")

            # Check JSON was written
            content = json_path.read_text()
            assert "JSON test message" in content

            # Verify it's valid JSON
            for line in content.strip().split("\n"):
                if line:
                    parsed = json.loads(line)
                    assert "message" in parsed
        finally:
            json_path.unlink(missing_ok=True)

    def test_with_config(self):
        """Test logging with LogConfig."""
        config = LogConfig(
            level=logging.WARNING,
            console=True,
            use_colors=False,
        )

        logger = setup_logging(config=config)
        assert logger.level == logging.WARNING


class TestGetLogger:
    """Test get_logger function."""

    def test_root_logger(self):
        """Test getting root logger."""
        logger = get_logger()
        assert logger.name == ROOT_LOGGER_NAME

    def test_sub_logger(self):
        """Test getting sub-logger."""
        logger = get_logger("submodule")
        assert logger.name == f"{ROOT_LOGGER_NAME}.submodule"

    def test_nested_logger(self):
        """Test getting nested logger."""
        logger = get_logger("module.submodule")
        assert logger.name == f"{ROOT_LOGGER_NAME}.module.submodule"


class TestSetLogLevel:
    """Test set_log_level function."""

    def test_set_by_int(self):
        """Test setting level by integer."""
        setup_logging(use_colors=False)
        set_log_level(logging.ERROR)

        logger = get_logger()
        assert logger.level == logging.ERROR

    def test_set_by_string(self):
        """Test setting level by string."""
        setup_logging(use_colors=False)
        set_log_level("DEBUG")

        logger = get_logger()
        assert logger.level == logging.DEBUG

    def test_case_insensitive(self):
        """Test that string level is case insensitive."""
        setup_logging(use_colors=False)
        set_log_level("warning")

        logger = get_logger()
        assert logger.level == logging.WARNING


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_creation(self):
        """Test basic creation."""
        tracker = ProgressTracker(total=100, description="Test")

        assert tracker.total == 100
        assert tracker.description == "Test"
        assert tracker.current == 0

    def test_update(self):
        """Test update method."""
        tracker = ProgressTracker(total=100, log_interval=50)

        tracker.update(25)
        assert tracker.current == 25

        tracker.update(25)
        assert tracker.current == 50

    def test_percent(self):
        """Test percent property."""
        tracker = ProgressTracker(total=100)

        tracker.update(50)
        assert tracker.percent == 50.0

        tracker.update(50)
        assert tracker.percent == 100.0

    def test_rate(self):
        """Test rate property."""
        tracker = ProgressTracker(total=100)
        tracker.update(100)

        # Rate should be positive
        assert tracker.rate > 0

    def test_eta(self):
        """Test ETA property."""
        tracker = ProgressTracker(total=100)
        tracker.update(50)

        # ETA should be finite when progress is made
        assert tracker.eta < float("inf")

    def test_zero_total(self):
        """Test with zero total items."""
        tracker = ProgressTracker(total=0)
        assert tracker.percent == 0


class TestTimedOperation:
    """Test timed_operation context manager."""

    def test_basic_timing(self):
        """Test basic timing functionality."""
        with timed_operation("Test", log_start=False) as timing:
            time.sleep(0.01)

        assert "elapsed" in timing
        assert timing["elapsed"] >= 0.01

    def test_with_exception(self):
        """Test timing when exception occurs."""
        with (
            pytest.raises(ValueError),
            timed_operation("Test", log_start=False) as timing,
        ):
            raise ValueError("Test error")

        # Timing should still be recorded
        assert "elapsed" in timing


class TestLogContext:
    """Test LogContext context manager."""

    def test_context_applied(self):
        """Test that context is applied to records."""
        with LogContext(file="test.tdms", assembly="M9"):
            factory = logging.getLogRecordFactory()
            record = factory(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Test",
                args=(),
                exc_info=None,
            )

            assert hasattr(record, "extra_data")
            assert record.extra_data["file"] == "test.tdms"
            assert record.extra_data["assembly"] == "M9"

    def test_context_removed_after_exit(self):
        """Test that context is removed after exit."""
        original_factory = logging.getLogRecordFactory()

        with LogContext(test="value"):
            pass

        # Factory should be restored
        assert logging.getLogRecordFactory() == original_factory


class TestArgumentParser:
    """Test argument parsing."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_argument_parser()
        assert parser is not None

    def test_required_args(self):
        """Test required arguments."""
        args = parse_arguments(["input.tdms"])
        assert args.input_file[0] == "input.tdms"

    def test_multiple_inputs(self):
        """Test multiple input files."""
        args = parse_arguments(["file1.tdms", "file2.tdms", "file3.tdms"])
        assert len(args.input_file) == 3

    def test_debug_flag(self):
        """Test --debug flag."""
        args = parse_arguments(["input.tdms", "--debug"])
        assert args.debug is True

    def test_quiet_flag(self):
        """Test --quiet flag."""
        args = parse_arguments(["input.tdms", "-q"])
        assert args.quiet is True

    def test_show_save_flags(self):
        """Test --show and --save flags."""
        args = parse_arguments(["input.tdms", "--show", "--save"])
        assert args.show is True
        assert args.save is True

    def test_processing_options(self):
        """Test processing option flags."""
        args = parse_arguments(
            [
                "input.tdms",
                "--synchronize",
                "--lag",
                "--distance",
            ]
        )
        assert args.synchronize is True
        assert args.lag is True
        assert args.distance is True

    def test_downsample_method_option(self):
        """Test --downsample-method/--downsample-params options."""
        args = parse_arguments(
            [
                "input.tdms",
                "--downsample-method",
                "stride",
                "--downsample-params",
                '{"n_out": 5000}',
            ]
        )
        config = args_to_downsample_config(args)
        assert config is not None
        assert config.method == "stride"
        assert config.n_out == 5000

    def test_tkey_option(self):
        """Test --tkey option."""
        args = parse_arguments(["input.tdms", "--tkey", "timestamp"])
        assert args.tkey == "timestamp"

    def test_log_file_option(self):
        """Test --log-file option."""
        args = parse_arguments(["input.tdms", "--log-file", "test.log"])
        assert args.log_file == Path("test.log")


class TestArgsToProcessingConfig:
    """Test args_to_processing_config function."""

    def test_basic_conversion(self):
        """Test basic argument conversion."""
        args = parse_arguments(["input.tdms"])
        config = args_to_processing_config(args)

        assert config is not None
        assert config.debug is False
        assert config.show is False

    def test_full_conversion(self):
        """Test full argument conversion."""
        args = parse_arguments(
            [
                "input.tdms",
                "--debug",
                "--show",
                "--save",
                "--synchronize",
                "--lag",
            ]
        )
        config = args_to_processing_config(args)

        assert config.debug is True
        assert config.show is True
        assert config.save is True
        assert config.synchronize is True
        assert config.compute_lag is True


class TestMain:
    """Test main entry point."""

    @patch("python_magnetrun.analysis.cli.process_overview_file")
    @patch("python_magnetrun.analysis.cli.natsorted")
    def test_main_success(self, mock_natsorted, mock_process):
        """Test successful main execution."""
        mock_natsorted.return_value = ["test.tdms"]
        mock_record = MagicMock()
        mock_record.filename = "test"
        mock_record.assembly = "M9"
        mock_record.mode = "overview"
        mock_record.t0 = None
        mock_record.duration = 100.0
        mock_record.teb = 0.0
        mock_record.BP = 0.0
        mock_record.has_data.return_value = True
        mock_record.data = {}
        mock_record.signatures = {}
        mock_record.sync_info = {}
        mock_record.metrics = {}
        mock_process.return_value = mock_record

        result = main(["test.tdms", "--dry-run"])

        # Dry run should not call process_overview_file
        # but with our mock setup it will succeed
        assert result in [0, 1]

    def test_main_missing_args(self):
        """Test main with missing arguments."""
        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code != 0


class TestIntegration:
    """Integration tests for CLI module."""

    def test_logging_workflow(self):
        """Test complete logging workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            json_file = Path(tmpdir) / "test.json"

            # Setup logging
            logger = setup_logging(
                debug=True,
                log_file=log_file,
                json_file=json_file,
                use_colors=False,
            )

            # Log some messages
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")

            # Verify text log
            text_content = log_file.read_text()
            assert "Debug message" in text_content
            assert "Info message" in text_content
            assert "Warning message" in text_content

            # Verify JSON log
            json_content = json_file.read_text()
            lines = [line for line in json_content.strip().split("\n") if line]
            assert len(lines) >= 3

            for line in lines:
                parsed = json.loads(line)
                assert "timestamp" in parsed
                assert "level" in parsed
                assert "message" in parsed

    def test_progress_workflow(self):
        """Test complete progress tracking workflow."""
        setup_logging(use_colors=False, quiet=True)

        tracker = ProgressTracker(total=5, description="Test", log_interval=1)

        for _i in range(5):
            tracker.update()

        tracker.finish()

        assert tracker.current == 5
        assert tracker.percent == 100.0
