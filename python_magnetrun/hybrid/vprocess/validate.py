"""
VProcess File Validation and Inspection Utility
================================================

Utility for validating VProcess files and inspecting their contents.

Usage:
    python -m python_magnetrun.hybrid.vprocess.validate <filepath>
    python -m python_magnetrun.hybrid.vprocess.validate <filepath> --check-data
    python -m python_magnetrun.hybrid.vprocess.validate <filepath> --verbose
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .vprocess_reader import VProcessFileReader

# Setup logger
logger = logging.getLogger(__name__)


def validate_vprocess_file(
    filepath: str, check_data: bool = False, verbose: bool = True
) -> dict[str, Any]:
    """
    Validate a VProcess file structure and content.

    Parameters
    ----------
    filepath : str
        Path to VProcess file
    check_data : bool
        Whether to perform detailed data validation
    verbose : bool
        Whether to print detailed output

    Returns
    -------
    dict
        Validation results with 'valid', 'errors', 'warnings', and 'info' keys
    """
    results: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }

    try:
        # Check file exists
        file_path = Path(filepath)
        if not file_path.exists():
            results["valid"] = False
            results["errors"].append(f"File not found: {filepath}")
            return results

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        results["info"]["file_size_mb"] = file_size_mb

        if verbose:
            print("=" * 70)
            print(f"VPROCESS FILE VALIDATION: {file_path.name}")
            print("=" * 70)
            print("\nFile Information:")
            print(f"  Path: {file_path}")
            print(f"  Size: {file_size_mb:.2f} MB")

        # Initialize reader
        reader = VProcessFileReader(filepath)

        # Parse header
        try:
            reader.parse_header()
        except (OSError, ValueError, RuntimeError) as e:
            results["valid"] = False
            results["errors"].append(f"Header parsing failed: {str(e)}")
            return results

        # Get metadata
        metadata = reader.get_metadata()
        results["info"]["metadata"] = metadata

        if verbose:
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

        # Validate header information
        if not metadata.get("data_offset"):
            results["errors"].append("Data offset not found in header")
            results["valid"] = False

        if not metadata.get("sample_width"):
            results["errors"].append("Sample width not found in header")
            results["valid"] = False

        if not metadata.get("frequency"):
            results["warnings"].append("Frequency not found in header")

        # Variable information
        if not reader.variables:
            results["errors"].append("No variables found in header")
            results["valid"] = False
        else:
            results["info"]["num_variables"] = len(reader.variables)
            results["info"]["num_analog"] = sum(
                1 for v in reader.variables if v.is_analog
            )
            results["info"]["num_digital"] = sum(
                1 for v in reader.variables if not v.is_analog
            )

            if verbose:
                print("\nVariables:")
                print(f"  Total: {len(reader.variables)}")
                print(f"  Analog: {results['info']['num_analog']}")
                print(f"  Digital: {results['info']['num_digital']}")

        # Check sample width calculation
        if metadata.get("sample_width"):
            expected_width = (
                8
                + (results["info"].get("num_analog", 0) * 4)
                + (results["info"].get("num_digital", 0) * 1)
            )
            actual_width = metadata["sample_width"]

            if actual_width != expected_width:
                results["warnings"].append(
                    f"Sample width mismatch: header={actual_width}, calculated={expected_width}"
                )

        # Time window information
        if metadata.get("start_time") and metadata.get("end_time"):
            start = metadata["start_time"]
            end = metadata["end_time"]
            duration = (end - start).total_seconds()

            results["info"]["start_time"] = start
            results["info"]["end_time"] = end
            results["info"]["duration_seconds"] = duration

            if verbose:
                print("\nTime Window:")
                print(f"  Start: {start}")
                print(f"  End: {end}")
                print(
                    f"  Duration: {duration:.1f} seconds ({duration / 3600:.2f} hours)"
                )

                if metadata.get("frequency"):
                    expected_samples = int(duration * metadata["frequency"])
                    print(f"  Expected samples: {expected_samples}")

        # Variable details
        if verbose and reader.variables:
            print("\nVariables (first 10):")
            var_info = reader.get_variable_info()
            print(var_info.head(10).to_string(index=False))
            if len(reader.variables) > 10:
                print(f"  ... and {len(reader.variables) - 10} more")

        # Data validation
        if check_data:
            if verbose:
                print("\n" + "-" * 70)
                print("DATA VALIDATION")
                print("-" * 70)

            try:
                # Read data
                if verbose:
                    print("\nReading data...")

                df = reader.read()
                results["info"]["data_samples"] = len(df)
                results["info"]["data_shape"] = df.shape

                if verbose:
                    print(f"✓ Successfully read {len(df)} samples")
                    print(f"  Shape: {df.shape}")
                    print(
                        f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                    )

                # Check for NaN values
                nan_counts = df.isna().sum()
                if nan_counts.any():
                    if verbose:
                        print("\n⚠ NaN values detected:")
                    for var, count in nan_counts[nan_counts > 0].items():
                        warning = f"NaN values in {var}: {count}"
                        results["warnings"].append(warning)
                        if verbose:
                            print(f"    {var}: {count} NaN values")

                # Check timestamp ordering
                if len(df) > 1:
                    time_diffs = df.index.to_series().diff()
                    negative_diffs = (time_diffs < pd.Timedelta(0)).sum()
                    if negative_diffs > 0:
                        warning = (
                            f"Non-monotonic timestamps: {negative_diffs} backward jumps"
                        )
                        results["warnings"].append(warning)
                        if verbose:
                            print(f"\n⚠ {warning}")

                # Check sampling frequency consistency
                if len(df) > 10 and metadata.get("frequency"):
                    actual_freq = len(df) / (df.index[-1] - df.index[0]).total_seconds()
                    expected_freq = metadata["frequency"]
                    freq_diff = abs(actual_freq - expected_freq)

                    if freq_diff > expected_freq * 0.1:  # More than 10% difference
                        warning = (
                            f"Sampling frequency mismatch: expected {expected_freq:.2f} Hz, "
                            f"actual {actual_freq:.2f} Hz"
                        )
                        results["warnings"].append(warning)
                        if verbose:
                            print(f"\n⚠ {warning}")

                # Show sample data
                if verbose:
                    print("\nSample Data (first 5 rows, first 5 columns):")
                    print(df.iloc[:5, :5].to_string())

                    # Basic statistics
                    print("\nData Statistics (first 5 analog variables):")
                    analog_vars = [v.name for v in reader.variables if v.is_analog][:5]
                    if analog_vars:
                        stats = df[analog_vars].describe()
                        print(stats.to_string())

            except (OSError, ValueError, RuntimeError) as e:
                results["errors"].append(f"Data reading error: {str(e)}")
                results["valid"] = False
                if verbose:
                    print(f"\n✗ Data reading failed: {str(e)}")

        # Summary
        if verbose:
            print("\n" + "=" * 70)
            print("VALIDATION SUMMARY")
            print("=" * 70)

            if results["valid"]:
                print("✓ File is valid")
            else:
                print("✗ File validation FAILED")

            if results["errors"]:
                print(f"\nErrors ({len(results['errors'])}):")
                for error in results["errors"]:
                    print(f"  ✗ {error}")

            if results["warnings"]:
                print(f"\nWarnings ({len(results['warnings'])}):")
                for warning in results["warnings"]:
                    print(f"  ⚠ {warning}")

            print()

    except (OSError, ValueError, RuntimeError) as e:
        results["valid"] = False
        results["errors"].append(f"Unexpected error: {str(e)}")
        if verbose:
            print(f"\n✗ Validation failed with error: {str(e)}")
            import traceback

            traceback.print_exc()

    return results
