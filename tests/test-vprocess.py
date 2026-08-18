"""
VProcess Reader - Test/Demo Script
===================================

Test script with mock data generator for VProcess reader.

Usage:
    python test-vprocess.py                   # Run all tests
    python test-vprocess.py --create-mock     # Create a mock vprocess file
    python test-vprocess.py --test-file FILE  # Test with specific file
"""

import argparse
import logging
import math
import struct
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from python_magnetrun.log_utils import BARE_FORMAT, setup_logging

# Setup logger
logger = logging.getLogger(__name__)


# Keep standalone test helpers resilient to expected runtime failures
# while avoiding blind exception handling.
_EXPECTED_TEST_EXCEPTIONS = (
    AssertionError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
)


def create_mock_vprocess_file(
    filepath: str = "test_data.vprocess",
    n_samples: int = 3600,
    n_analog: int = 10,
    n_digital: int = 2,
) -> str:
    """
    Create a mock vprocess file for testing.

    Parameters
    ----------
    filepath : str
        Output file path
    n_samples : int
        Number of samples to generate
    n_analog : int
        Number of analog variables
    n_digital : int
        Number of digital variables

    Returns
    -------
    str
        Path to created file
    """
    logger.info(f"Creating mock vprocess file: {filepath}")
    logger.info(f"  Samples: {n_samples}")
    logger.info(f"  Analog vars: {n_analog}")
    logger.info(f"  Digital vars: {n_digital}")

    # Generate variable names (alphabetically sorted)
    analog_vars = [f"TEST_ANA_{i:03d}" for i in range(n_analog)]
    digital_vars = [f"TEST_DIG_{i:03d}" for i in range(n_digital)]
    all_vars = sorted(analog_vars + digital_vars)

    # Calculate sample width
    sample_width = 8 + (n_analog * 4) + (n_digital * 1)

    # Create header
    header_lines = []
    header_lines.append("# vprocess data file - v3.0 (Mock for Testing)")
    header_lines.append("# processed on TEST-SYSTEM [127.0.0.1] (test)")
    header_lines.append("# header [encoding:UTF-8 - line-ending:unix]")

    # Variables line
    var_specs = []
    for _i, var in enumerate(all_vars):
        if var in analog_vars:
            spec = f"{var} [type:float32|unit:U|min:0.00|max:100.00|df:%.2f]"
        else:
            spec = f"{var} [type:dig]"
        var_specs.append(spec)

    variables_line = "# variables = " + "; ".join(var_specs)
    header_lines.append(variables_line)

    # Time window
    start_time = datetime.now(UTC)
    end_time = start_time + timedelta(seconds=n_samples - 1)
    windows_line = (
        f"# windows = [UTC] {start_time.strftime('%d/%m/%Y-%H:%M:%S.%f')[:-3]} "
        f"-> {end_time.strftime('%d/%m/%Y-%H:%M:%S.%f')[:-3]}"
    )
    header_lines.append(windows_line)

    # Frequency
    header_lines.append("# frequency = 1.000 Hz")
    header_lines.append("# timestamp = absolute")

    # Data helper (offset will be calculated)
    data_helper_line = (
        f"# data-helper [offset:PLACEHOLDER - time:8(B) - width:{sample_width}(B)]"
    )
    header_lines.append(data_helper_line)

    # Write file
    with open(filepath, "wb") as f:
        # Write header
        header_text = "\n".join(header_lines) + "\n"
        header_bytes = header_text.encode("utf-8")

        # Update offset in header
        # PLACEHOLDER is 11 chars; pad hex to same length so the replacement
        # doesn't shift the offset value we just measured.
        offset = len(header_bytes)
        header_text = header_text.replace("PLACEHOLDER", f"0x{offset:09x}")
        header_bytes = header_text.encode("utf-8")

        f.write(header_bytes)

        # Write binary data
        timestamp = start_time.timestamp()

        for i in range(n_samples):
            f.write(struct.pack(">d", timestamp + i))

            for j, _avar in enumerate(analog_vars):
                value = 50 + 30 * math.sin(2 * math.pi * (i / n_samples) * (j + 1))
                f.write(struct.pack(">f", value))

            for j in range(n_digital):
                value = (i + j) % 2
                f.write(struct.pack("B", value))

    logger.info(f"Mock file created: {filepath}")
    logger.info(f"  File size: {Path(filepath).stat().st_size} bytes")
    logger.info(f"  Offset: 0x{offset:x}")
    return filepath


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def filepath(tmp_path):
    """Create a small mock vprocess file for each test."""
    return create_mock_vprocess_file(
        filepath=str(tmp_path / "test_data.vprocess"),
        n_samples=100,
        n_analog=5,
        n_digital=1,
    )


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


def test_basic_reading(filepath: str) -> None:
    """Test basic file reading."""
    from python_magnetrun.hybrid.vprocess.vprocess_reader import VProcessFileReader

    reader = VProcessFileReader(filepath)
    df = reader.read()

    assert df is not None
    assert len(df) > 0, "DataFrame should have rows"
    assert len(df.columns) > 0, "DataFrame should have columns"


def test_header_parsing(filepath: str) -> None:
    """Test header parsing."""
    from python_magnetrun.hybrid.vprocess.vprocess_reader import VProcessFileReader

    reader = VProcessFileReader(filepath)
    reader.parse_header()

    metadata = reader.get_metadata()
    assert isinstance(metadata, dict), "metadata should be a dict"
    assert len(metadata) > 0, "metadata should not be empty"

    var_info = reader.get_variable_info()
    assert var_info is not None, "variable info should not be None"
    assert len(var_info) > 0, "should have at least one variable"


def test_variable_selection(filepath: str) -> None:
    """Test selecting specific variables."""
    from python_magnetrun.hybrid.vprocess.vprocess_reader import read_vprocess_file

    df = read_vprocess_file(filepath)
    assert len(df.columns) >= 3, "need at least 3 columns to test selection"

    selected_vars = list(df.columns[:3])
    df_selected = df[selected_vars]

    assert df_selected.shape[1] == 3
    assert df_selected.shape[0] == df.shape[0]


def test_time_filtering(filepath: str) -> None:
    """Test time-based filtering."""
    from python_magnetrun.hybrid.vprocess.vprocess_reader import read_vprocess_file

    df = read_vprocess_file(filepath)
    assert len(df) > 10, "need enough samples to filter"

    n_filter = min(10, len(df))
    df_filtered = df.iloc[:n_filter]

    assert df_filtered.shape[0] == n_filter
    assert df_filtered.shape[1] == df.shape[1]


def test_validation_script(filepath: str) -> None:
    """Test validation script."""
    from python_magnetrun.hybrid.vprocess.validate import validate_vprocess_file

    results = validate_vprocess_file(filepath, check_data=True, verbose=False)

    assert results["valid"], f"Validation failed: {results.get('errors')}"


def test_export_formats(filepath: str, tmp_path) -> None:
    """Test exporting to different formats."""
    from python_magnetrun.hybrid.vprocess.vprocess_reader import read_vprocess_file

    df = read_vprocess_file(filepath)
    df = df.iloc[:20]

    csv_file = tmp_path / "test_export.csv"
    df.to_csv(str(csv_file))
    assert csv_file.exists()

    try:
        parquet_file = tmp_path / "test_export.parquet"
        df.to_parquet(str(parquet_file))
        assert parquet_file.exists()
    except ImportError:
        pass  # pyarrow not installed, skip


# ---------------------------------------------------------------------------
# Standalone helpers (used by main() / run_all_tests)
# ---------------------------------------------------------------------------


def _test_basic_reading(filepath: str) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Basic Reading")
    logger.info("=" * 70)
    try:
        from python_magnetrun.hybrid.vprocess.vprocess_reader import VProcessFileReader

        reader = VProcessFileReader(filepath)
        df = reader.read()
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Time range: {df.index[0]} to {df.index[-1]}")
        logger.info("\nFirst 3 rows:")
        logger.info(str(df.head(3)))
        logger.info("\nTest passed")
        return True
    except _EXPECTED_TEST_EXCEPTIONS as e:
        logger.error(f"\nTest failed: {str(e)}")
        return False


def _test_header_parsing(filepath: str) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Header Parsing")
    logger.info("=" * 70)
    try:
        from python_magnetrun.hybrid.vprocess.vprocess_reader import VProcessFileReader

        reader = VProcessFileReader(filepath)
        reader.parse_header()
        metadata = reader.get_metadata()
        logger.info("\nMetadata:")
        for key, value in metadata.items():
            logger.info(f"  {key}: {value}")
        var_info = reader.get_variable_info()
        logger.info(f"\nVariables ({len(var_info)}):")
        logger.info(str(var_info))
        logger.info("\nTest passed")
        return True
    except _EXPECTED_TEST_EXCEPTIONS as e:
        logger.error(f"\nTest failed: {str(e)}")
        return False


def _test_variable_selection(filepath: str) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Variable Selection")
    logger.info("=" * 70)
    try:
        from python_magnetrun.hybrid.vprocess.vprocess_reader import read_vprocess_file

        df = read_vprocess_file(filepath)
        selected_vars = list(df.columns[:3])
        df_selected = df[selected_vars]
        logger.info(f"\nSelected variables: {selected_vars}")
        logger.info(f"  Shape: {df_selected.shape}")
        logger.info("\nStatistics:")
        logger.info(str(df_selected.describe()))
        logger.info("\nTest passed")
        return True
    except _EXPECTED_TEST_EXCEPTIONS as e:
        logger.error(f"\nTest failed: {str(e)}")
        return False


def _test_time_filtering(filepath: str) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Time Filtering")
    logger.info("=" * 70)
    try:
        from python_magnetrun.hybrid.vprocess.vprocess_reader import read_vprocess_file

        df = read_vprocess_file(filepath)
        df_filtered = df.iloc[:100]
        logger.info(f"\nFiltered data shape: {df_filtered.shape}")
        logger.info(f"  Original: {df.shape[0]} samples")
        logger.info(f"  Filtered: {df_filtered.shape[0]} samples")
        logger.info(f"  Time range: {df_filtered.index[0]} to {df_filtered.index[-1]}")
        logger.info("\nTest passed")
        return True
    except _EXPECTED_TEST_EXCEPTIONS as e:
        logger.error(f"\nTest failed: {str(e)}")
        return False


def _test_validation_script(filepath: str) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Validation Script")
    logger.info("=" * 70)
    try:
        from python_magnetrun.hybrid.vprocess.validate import validate_vprocess_file

        results = validate_vprocess_file(filepath, check_data=True, verbose=False)
        if results["valid"]:
            logger.info("Validation passed")
        else:
            logger.error(f"Validation failed: {results['errors']}")
            return False
        logger.info("\nTest passed")
        return True
    except _EXPECTED_TEST_EXCEPTIONS as e:
        logger.error(f"\nTest failed: {str(e)}")
        return False


def _test_export_formats(filepath: str) -> bool:
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: Export Formats")
    logger.info("=" * 70)
    try:
        from python_magnetrun.hybrid.vprocess.vprocess_reader import read_vprocess_file

        df = read_vprocess_file(filepath)
        df = df.iloc[:100]

        csv_file = "test_export.csv"
        df.to_csv(csv_file)
        logger.info(f"CSV export successful: {csv_file}")
        Path(csv_file).unlink()

        try:
            parquet_file = "test_export.parquet"
            df.to_parquet(parquet_file)
            logger.info(f"Parquet export successful: {parquet_file}")
            Path(parquet_file).unlink()
        except ImportError:
            logger.info("  Parquet export skipped (pyarrow not installed)")

        logger.info("\nTest passed")
        return True
    except _EXPECTED_TEST_EXCEPTIONS as e:
        logger.error(f"\nTest failed: {str(e)}")
        return False


def run_all_tests(filepath: str) -> bool:
    """Run all tests."""
    logger.info("\n" + "*" * 70)
    logger.info("VPROCESS READER - TEST SUITE")
    logger.info("*" * 70)

    tests = [
        ("Basic Reading", _test_basic_reading),
        ("Header Parsing", _test_header_parsing),
        ("Variable Selection", _test_variable_selection),
        ("Time Filtering", _test_time_filtering),
        ("Validation Script", _test_validation_script),
        ("Export Formats", _test_export_formats),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func(filepath):
                passed += 1
            else:
                failed += 1
        except _EXPECTED_TEST_EXCEPTIONS as e:
            logger.error(f"\nTest '{name}' crashed: {str(e)}")
            failed += 1

    logger.info("\n" + "*" * 70)
    logger.info("TEST SUMMARY")
    logger.info("*" * 70)
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        logger.info("\nAll tests passed!")
        return True
    else:
        logger.error(f"\n{failed} test(s) failed")
        return False


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(description="Test VProcess reader functionality")

    parser.add_argument(
        "--create-mock",
        action="store_true",
        help="Create a mock vprocess file for testing",
    )
    parser.add_argument(
        "--output", "-o", default="mock_data.vprocess", help="Output file for mock data"
    )
    parser.add_argument("--samples", type=int, default=3600, help="Number of samples")
    parser.add_argument(
        "--analog", type=int, default=10, help="Number of analog variables"
    )
    parser.add_argument(
        "--digital", type=int, default=2, help="Number of digital variables"
    )
    parser.add_argument("--test-file", help="Test with specific file instead of mock")

    args = parser.parse_args()

    # Configure logging
    setup_logging(fmt=BARE_FORMAT)

    if args.create_mock:
        create_mock_vprocess_file(
            filepath=args.output,
            n_samples=args.samples,
            n_analog=args.analog,
            n_digital=args.digital,
        )
        return 0

    if args.test_file:
        test_file = args.test_file
    else:
        test_file = create_mock_vprocess_file(
            filepath="test_data.vprocess",
            n_samples=100,
            n_analog=5,
            n_digital=1,
        )

    success = run_all_tests(test_file)

    if not args.test_file:
        try:
            Path(test_file).unlink()
            logger.info(f"\nCleaned up test file: {test_file}")
        except OSError:
            pass

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
