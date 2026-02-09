"""
VProcess Reader - Test/Demo Script
===================================

Test script with mock data generator for VProcess reader.

Usage:
    python test.py                   # Run all tests
    python test.py --create-mock     # Create a mock vprocess file
    python test.py --test-file FILE  # Test with specific file
"""

import argparse
import struct
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
import logging
import math

# Setup logger
logger = logging.getLogger(__name__)


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
    for i, var in enumerate(all_vars):
        if var in analog_vars:
            spec = f"{var} [type:float32|unit:U|min:0.00|max:100.00|df:%.2f]"
        else:
            spec = f"{var} [type:dig]"
        var_specs.append(spec)

    variables_line = "# variables = " + "; ".join(var_specs)
    header_lines.append(variables_line)

    # Time window
    start_time = datetime.now(timezone.utc)
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
        offset = len(header_bytes)
        header_text = header_text.replace("PLACEHOLDER", f"0x{offset:x}")
        header_bytes = header_text.encode("utf-8")

        f.write(header_bytes)

        # Write binary data
        timestamp = start_time.timestamp()

        for i in range(n_samples):
            # Timestamp (little-endian double)
            f.write(struct.pack("<d", timestamp + i))

            # Analog variables (sine waves with different frequencies)
            for j, var in enumerate(analog_vars):
                value = 50 + 30 * math.sin(2 * math.pi * (i / n_samples) * (j + 1))
                f.write(struct.pack("<f", value))

            # Digital variables (alternating pattern)
            for j in range(n_digital):
                value = (i + j) % 2
                f.write(struct.pack("B", value))

    logger.info(f"✓ Mock file created: {filepath}")
    logger.info(f"  File size: {Path(filepath).stat().st_size} bytes")
    logger.info(f"  Offset: 0x{offset:x}")
    return filepath


def test_basic_reading(filepath: str) -> bool:
    """Test basic file reading."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Basic Reading")
    logger.info("=" * 70)

    try:
        from vprocess_reader import VProcessFileReader

        reader = VProcessFileReader(filepath)
        logger.info(f"✓ Reader initialized")

        # Read data
        df = reader.read()
        logger.info(f"✓ Data read successfully")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Time range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"\nFirst 3 rows:")
        logger.info(str(df.head(3)))

        logger.info(f"\n✓ Test passed")
        return True

    except Exception as e:
        logger.error(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_header_parsing(filepath: str) -> bool:
    """Test header parsing."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Header Parsing")
    logger.info("=" * 70)

    try:
        from vprocess_reader import VProcessFileReader

        reader = VProcessFileReader(filepath)
        reader.parse_header()

        # Check metadata
        metadata = reader.get_metadata()
        logger.info(f"\n✓ Header parsed successfully")
        logger.info(f"\nMetadata:")
        for key, value in metadata.items():
            logger.info(f"  {key}: {value}")

        # Check variables
        var_info = reader.get_variable_info()
        logger.info(f"\n✓ Variable info extracted")
        logger.info(f"\nVariables ({len(var_info)}):")
        logger.info(str(var_info))

        logger.info(f"\n✓ Test passed")
        return True

    except Exception as e:
        logger.error(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_variable_selection(filepath: str) -> bool:
    """Test selecting specific variables."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Variable Selection")
    logger.info("=" * 70)

    try:
        from vprocess_reader import read_vprocess_file

        df = read_vprocess_file(filepath)

        # Select first 3 variables
        selected_vars = list(df.columns[:3])
        df_selected = df[selected_vars]

        logger.info(f"\n✓ Selected variables: {selected_vars}")
        logger.info(f"  Shape: {df_selected.shape}")
        logger.info(f"\nStatistics:")
        logger.info(str(df_selected.describe()))

        logger.info(f"\n✓ Test passed")
        return True

    except Exception as e:
        logger.error(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_time_filtering(filepath: str) -> bool:
    """Test time-based filtering."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Time Filtering")
    logger.info("=" * 70)

    try:
        from vprocess_reader import read_vprocess_file
        import pandas as pd

        df = read_vprocess_file(filepath)

        # Filter first 100 samples
        df_filtered = df.iloc[:100]

        logger.info(f"\n✓ Filtered data shape: {df_filtered.shape}")
        logger.info(f"  Original: {df.shape[0]} samples")
        logger.info(f"  Filtered: {df_filtered.shape[0]} samples")
        logger.info(f"  Time range: {df_filtered.index[0]} to {df_filtered.index[-1]}")

        logger.info(f"\n✓ Test passed")
        return True

    except Exception as e:
        logger.error(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_validation_script(filepath: str) -> bool:
    """Test validation script."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Validation Script")
    logger.info("=" * 70)

    try:
        from validate import validate_vprocess_file

        logger.info("\n✓ Running validation...")
        results = validate_vprocess_file(filepath, check_data=True, verbose=False)

        if results["valid"]:
            logger.info(f"✓ Validation passed")
        else:
            logger.error(f"✗ Validation failed")
            logger.error(f"  Errors: {results['errors']}")
            return False

        logger.info(f"\n✓ Test passed")
        return True

    except Exception as e:
        logger.error(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_export_formats(filepath: str) -> bool:
    """Test exporting to different formats."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: Export Formats")
    logger.info("=" * 70)

    try:
        from vprocess_reader import read_vprocess_file

        df = read_vprocess_file(filepath)

        # Limit to first 100 samples for faster testing
        df = df.iloc[:100]

        # Test CSV export
        csv_file = "test_export.csv"
        df.to_csv(csv_file)
        logger.info(f"✓ CSV export successful: {csv_file}")
        Path(csv_file).unlink()

        # Test Parquet (if available)
        try:
            parquet_file = "test_export.parquet"
            df.to_parquet(parquet_file)
            logger.info(f"✓ Parquet export successful: {parquet_file}")
            Path(parquet_file).unlink()
        except ImportError:
            logger.info(f"  Parquet export skipped (pyarrow not installed)")

        logger.info(f"\n✓ Test passed")
        return True

    except Exception as e:
        logger.error(f"\n✗ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests(filepath: str) -> bool:
    """Run all tests."""
    logger.info("\n" + "*" * 70)
    logger.info("VPROCESS READER - TEST SUITE")
    logger.info("*" * 70)

    tests = [
        ("Basic Reading", test_basic_reading),
        ("Header Parsing", test_header_parsing),
        ("Variable Selection", test_variable_selection),
        ("Time Filtering", test_time_filtering),
        ("Validation Script", test_validation_script),
        ("Export Formats", test_export_formats),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func(filepath):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"\n✗ Test '{name}' crashed: {str(e)}")
            failed += 1

    logger.info("\n" + "*" * 70)
    logger.info("TEST SUMMARY")
    logger.info("*" * 70)
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        logger.info("\n✓ All tests passed!")
        return True
    else:
        logger.error(f"\n✗ {failed} test(s) failed")
        return False


def main():
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
    parser.add_argument("--analog", type=int, default=10, help="Number of analog variables")
    parser.add_argument("--digital", type=int, default=2, help="Number of digital variables")
    parser.add_argument("--test-file", help="Test with specific file instead of mock")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    if args.create_mock:
        # Just create mock file
        create_mock_vprocess_file(
            filepath=args.output,
            n_samples=args.samples,
            n_analog=args.analog,
            n_digital=args.digital,
        )
        return 0

    # Determine test file
    if args.test_file:
        test_file = args.test_file
    else:
        # Create mock file for testing
        test_file = create_mock_vprocess_file(
            filepath="test_data.vprocess",
            n_samples=100,  # Smaller for faster tests
            n_analog=5,
            n_digital=1,
        )

    # Run tests
    success = run_all_tests(test_file)

    # Cleanup mock file if created
    if not args.test_file:
        try:
            Path(test_file).unlink()
            logger.info(f"\nCleaned up test file: {test_file}")
        except:
            pass

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
