"""
RMS File Reader Validation Script
==================================

This script validates RMS files and checks for common issues.
"""

import sys
from pathlib import Path

from rms_reader import RMSFileReader


def validate_rms_file(filepath: str) -> bool:
    """
    Validate an RMS file and report any issues.

    Returns True if file is valid, False otherwise.
    """
    print(f"Validating: {filepath}")
    print("=" * 70)

    try:
        reader = RMSFileReader(filepath)

        # Test 1: Header parsing
        print("\n[1/5] Testing header parsing...")
        reader.parse_header()

        if not reader.header_lines:
            print("  ❌ FAILED: No header lines found")
            return False

        print(f"  ✅ PASSED: Found {len(reader.header_lines)} header lines")

        # Test 2: Metadata extraction
        print("\n[2/5] Testing metadata extraction...")
        required_metadata = [
            "frequency",
            "data_offset",
            "sample_width",
            "timestamp_bytes",
        ]
        missing = [key for key in required_metadata if key not in reader.metadata]

        if missing:
            print(f"  ❌ FAILED: Missing metadata: {missing}")
            return False

        print("  ✅ PASSED: All required metadata present")
        print(f"      - Frequency: {reader.metadata['frequency']} Hz")
        print(
            f"      - Data offset: 0x{reader.metadata['data_offset']:x} ({reader.metadata['data_offset']} bytes)"
        )
        print(f"      - Sample width: {reader.metadata['sample_width']} bytes")
        print(f"      - Timestamp size: {reader.metadata['timestamp_bytes']} bytes")

        # Test 3: Variable definitions
        print("\n[3/5] Testing variable definitions...")
        if not reader.variables:
            print("  ❌ FAILED: No variables defined")
            return False

        analog_vars = sum(1 for v in reader.variables if v.is_analog)
        digital_vars = len(reader.variables) - analog_vars

        print(f"  ✅ PASSED: Found {len(reader.variables)} variables")
        print(f"      - Analog (float32): {analog_vars}")
        print(f"      - Digital (bit): {digital_vars}")

        # Check expected sample width
        expected_width = reader.metadata["timestamp_bytes"]
        expected_width += analog_vars * 4  # 4 bytes per analog
        expected_width += digital_vars * 1  # 1 byte per digital

        print(f"      - Expected width: {expected_width} bytes")
        print(f"      - Actual width: {reader.metadata['sample_width']} bytes")

        if expected_width != reader.metadata["sample_width"]:
            diff = reader.metadata["sample_width"] - expected_width
            print(f"      ⚠️  WARNING: Width mismatch ({diff} bytes difference)")
            print("          This is expected for FEPC-AUX-LNCMI (7 unnamed digital variables)")

        # Test 4: File size check
        print("\n[4/5] Testing file integrity...")
        file_size = Path(filepath).stat().st_size
        data_offset = reader.metadata["data_offset"]
        sample_width = reader.metadata["sample_width"]

        available_data = file_size - data_offset
        expected_samples = available_data // sample_width
        remainder = available_data % sample_width

        print(f"  File size: {file_size:,} bytes")
        print(f"  Header size: {data_offset:,} bytes")
        print(f"  Data size: {available_data:,} bytes")
        print(f"  Expected samples: {expected_samples:,}")

        if remainder != 0:
            print(f"  ⚠️  WARNING: {remainder} extra bytes (incomplete sample)")
        else:
            print("  ✅ PASSED: File size is consistent")

        # Test 5: Data reading
        print("\n[5/5] Testing data reading...")
        df = reader.read_binary_data()

        if df is None or len(df) == 0:
            print("  ❌ FAILED: No data read")
            return False

        print(f"  ✅ PASSED: Successfully read {len(df):,} samples")
        print(f"      - Time range: {df.index[0]} to {df.index[-1]}")

        duration = (df.index[-1] - df.index[0]).total_seconds()
        print(f"      - Duration: {duration:.2f} seconds")

        actual_freq = (len(df) - 1) / duration if duration > 0 else 0
        print(f"      - Actual sampling rate: {actual_freq:.3f} Hz")
        print(f"      - Expected sampling rate: {reader.metadata['frequency']} Hz")

        freq_diff = abs(actual_freq - reader.metadata["frequency"])
        if freq_diff > 0.1:
            print(f"      ⚠️  WARNING: Sampling rate mismatch ({freq_diff:.3f} Hz difference)")

        # Check for data quality issues
        print("\n[6/6] Data quality checks...")

        # Check for NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print("  ⚠️  WARNING: Found NaN values:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"      - {col}: {count} NaN values")
        else:
            print("  ✅ No NaN values found")

        # Check for constant values (might indicate sensor issues)
        constant_vars = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_vars.append(col)

        if constant_vars:
            print(f"  ⚠️  WARNING: {len(constant_vars)} variables have constant values:")
            for var in constant_vars[:5]:  # Show first 5
                print(f"      - {var} = {df[var].iloc[0]}")
            if len(constant_vars) > 5:
                print(f"      ... and {len(constant_vars) - 5} more")
        else:
            print("  ✅ All variables show variation")

        print("\n" + "=" * 70)
        print("✅ VALIDATION SUCCESSFUL")
        print("=" * 70)
        return True

    except (OSError, ValueError, RuntimeError) as e:
        print("\n❌ VALIDATION FAILED with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def compare_rms_files(filepath1: str, filepath2: str) -> bool:
    """Compare two RMS files to check if they have the same structure."""
    print("\nComparing RMS files:")
    print(f"  File 1: {filepath1}")
    print(f"  File 2: {filepath2}")
    print("=" * 70)

    try:
        reader1 = RMSFileReader(filepath1)
        reader2 = RMSFileReader(filepath2)

        reader1.parse_header()
        reader2.parse_header()

        # Compare metadata
        print("\nMetadata comparison:")
        for key in ["frequency", "sample_width", "timestamp_bytes"]:
            val1 = reader1.metadata.get(key, "N/A")
            val2 = reader2.metadata.get(key, "N/A")
            match = "✅" if val1 == val2 else "❌"
            print(f"  {match} {key}: {val1} vs {val2}")

        # Compare variables
        print("\nVariable comparison:")
        vars1 = set(v.name for v in reader1.variables)
        vars2 = set(v.name for v in reader2.variables)

        only_in_1 = vars1 - vars2
        only_in_2 = vars2 - vars1
        common = vars1 & vars2

        print(f"  Common variables: {len(common)}")

        if only_in_1:
            print(f"  Only in file 1: {len(only_in_1)}")
            for var in list(only_in_1)[:5]:
                print(f"    - {var}")

        if only_in_2:
            print(f"  Only in file 2: {len(only_in_2)}")
            for var in list(only_in_2)[:5]:
                print(f"    - {var}")

        if not only_in_1 and not only_in_2:
            print("  ✅ Files have identical variable sets")

        return True

    except (OSError, ValueError, RuntimeError) as e:
        print("\n❌ Comparison failed with error:")
        print(f"   {type(e).__name__}: {e}")
        return False


def inspect_rms_file(filepath: str) -> bool:
    """Provide detailed inspection of an RMS file."""
    print(f"\nInspecting: {filepath}")
    print("=" * 70)

    try:
        reader = RMSFileReader(filepath)
        reader.parse_header()

        print("\n📋 HEADER LINES:")
        print("-" * 70)
        for i, line in enumerate(reader.header_lines, 1):
            print(f"{i}. {line}")

        print("\n📊 METADATA:")
        print("-" * 70)
        for key, value in sorted(reader.metadata.items()):
            print(f"  {key}: {value}")

        print("\n📈 VARIABLES:")
        print("-" * 70)
        var_info = reader.get_variable_info()

        # Group by type
        analog = var_info[var_info["type"] == "float32"]
        digital = var_info[var_info["type"] == "bit"]

        print(f"\nAnalog variables ({len(analog)}):")
        if len(analog) > 0:
            print(analog.to_string(index=False, max_rows=10))

        print(f"\nDigital variables ({len(digital)}):")
        if len(digital) > 0:
            print(digital[["name", "type"]].to_string(index=False, max_rows=10))

        # Try to read a small sample
        print("\n📦 DATA SAMPLE:")
        print("-" * 70)
        df = reader.read()
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())

        return True

    except (OSError, ValueError, RuntimeError) as e:
        print("\n❌ Inspection failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
RMS File Validation Script

Usage:
  python validate_rms.py validate <file>       - Validate an RMS file
  python validate_rms.py inspect <file>        - Detailed inspection
  python validate_rms.py compare <file1> <file2> - Compare two files
  
Examples:
  python validate_rms.py validate data.rms
  python validate_rms.py inspect data.rms
  python validate_rms.py compare data1.rms data2.rms
        """)
        sys.exit(1)

    command = sys.argv[1]

    if command == "validate":
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            sys.exit(1)
        filepath = sys.argv[2]
        success = validate_rms_file(filepath)
        sys.exit(0 if success else 1)

    elif command == "inspect":
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            sys.exit(1)
        filepath = sys.argv[2]
        success = inspect_rms_file(filepath)
        sys.exit(0 if success else 1)

    elif command == "compare":
        if len(sys.argv) < 4:
            print("Error: Please provide two file paths")
            sys.exit(1)
        filepath1 = sys.argv[2]
        filepath2 = sys.argv[3]
        success = compare_rms_files(filepath1, filepath2)
        sys.exit(0 if success else 1)

    else:
        print(f"Unknown command: {command}")
        print("Use 'validate', 'inspect', or 'compare'")
        sys.exit(1)
