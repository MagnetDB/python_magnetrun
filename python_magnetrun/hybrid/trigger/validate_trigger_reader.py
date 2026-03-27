"""
Validation script for trigger reader implementation

This script performs comprehensive validation:
1. File structure validation
2. Binary format validation
3. Configuration parsing validation
4. Data reading validation
5. Calibration validation
"""

import logging
import sys
from pathlib import Path

import numpy as np
from trigger_reader import (
    list_trigger_files,
    load_trigger_config,
    parse_trigger_directory,
    read_trigger_data,
    read_trigger_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_directory_structure(trigger_dir: Path) -> bool:
    """Validate trigger directory structure"""
    print("\n" + "=" * 70)
    print("VALIDATING DIRECTORY STRUCTURE")
    print("=" * 70)

    errors = []

    # Check main directory
    if not trigger_dir.exists():
        errors.append(f"Trigger directory does not exist: {trigger_dir}")
        return False

    print(f"✓ Trigger directory exists: {trigger_dir.name}")

    # Check for FEPC subdirectories
    systems = []
    for system in ["FEPC-LNCMI", "FEPC-AUX-LNCMI"]:
        system_dir = trigger_dir / system
        if system_dir.exists():
            print(f"✓ Found system directory: {system}")
            systems.append(system)

            # Check for required files
            cfg_files = list(system_dir.glob("HOST_*_DATA.CFG"))
            if cfg_files:
                print(f"  ✓ Found CFG file: {cfg_files[0].name}")
            else:
                errors.append(f"  ✗ No CFG file found in {system}")

            props_file = system_dir / "EventInfo.properties"
            if props_file.exists():
                print("  ✓ Found EventInfo.properties")
            else:
                print("  ⚠ EventInfo.properties not found (optional)")

            # Check for binary files
            bin_files = list(system_dir.glob("host_*.bin"))
            if bin_files:
                print(f"  ✓ Found {len(bin_files)} binary file(s)")
            else:
                errors.append(f"  ✗ No binary files found in {system}")

    if not systems:
        errors.append("No FEPC system directories found")

    if errors:
        print("\n✗ VALIDATION FAILED:")
        for error in errors:
            print(f"  {error}")
        return False

    print("\n✓ Directory structure validation passed")
    return True


def validate_binary_files(trigger_dir: Path, system: str) -> bool:
    """Validate binary file structure"""
    print("\n" + "=" * 70)
    print(f"VALIDATING BINARY FILES - {system}")
    print("=" * 70)

    errors = []
    warnings = []

    files = list_trigger_files(trigger_dir, system)

    if not files:
        errors.append(f"No binary files found for {system}")
        print(f"✗ {errors[-1]}")
        return False

    print(f"Found {len(files)} binary file(s)")

    for tf in files:
        print(f"\nFile: {tf.filepath.name}")
        print(f"  Slot: {tf.slot}")
        print(f"  Card type: {tf.card_type}")
        print(f"  File size: {tf.file_size:,} bytes")
        print(f"  Expected size: {tf.expected_size:,} bytes")

        # Validate file size
        if tf.file_size == tf.expected_size:
            print("  ✓ File size matches expected")
        else:
            warning = (
                f"  ⚠ File size mismatch (expected {tf.expected_size:,}, got {tf.file_size:,})"
            )
            warnings.append(warning)
            print(warning)

        # Try to read header
        try:
            from trigger_reader import read_trigger_file_header

            timestamp = read_trigger_file_header(tf.filepath, tf.card_type)
            print("  ✓ Header read successfully")
            print(f"    Timestamp: {timestamp}")
        except (OSError, ValueError, RuntimeError) as e:
            error = f"  ✗ Failed to read header: {e}"
            errors.append(error)
            print(error)

        # Try to read first 100 samples
        try:
            data, timestamp = read_trigger_file(tf.filepath, tf.card_type, num_samples=100)
            print("  ✓ Data read successfully")
            print(f"    Shape: {data.shape}")
            print(f"    Data type: {data.dtype}")
        except (OSError, ValueError, RuntimeError) as e:
            error = f"  ✗ Failed to read data: {e}"
            errors.append(error)
            print(error)

    if warnings:
        print("\n⚠ Warnings:")
        for warning in warnings:
            print(f"  {warning}")

    if errors:
        print("\n✗ VALIDATION FAILED:")
        for error in errors:
            print(f"  {error}")
        return False

    print("\n✓ Binary file validation passed")
    return True


def validate_configuration(trigger_dir: Path, system: str) -> bool:
    """Validate configuration parsing"""
    print("\n" + "=" * 70)
    print(f"VALIDATING CONFIGURATION - {system}")
    print("=" * 70)

    errors = []

    # Load config
    try:
        config = load_trigger_config(trigger_dir, system)
        if config is None:
            errors.append("Config loading returned None")
            print(f"✗ {errors[-1]}")
            return False

        print("✓ Configuration loaded successfully")
        print(f"  FEPC name: {config.fepc_name}")
        print(f"  Number of cards: {config.num_cards}")

    except (OSError, ValueError, RuntimeError) as e:
        errors.append(f"Failed to load config: {e}")
        print(f"✗ {errors[-1]}")
        return False

    # Validate cards
    if config.num_cards == 0:
        errors.append("No cards found in configuration")
    else:
        print(f"\nValidating {config.num_cards} card(s):")

        for card in config.cards:
            print(f"\n  Slot {card.slot}:")
            print(f"    Type: {card.card_type}")
            print(f"    Channels: {card.num_channels}")
            print(f"    Variables: {len(card.variable_names)}")

            # Validate variable count matches channel count
            if len(card.variable_names) != card.num_channels:
                error = f"    ✗ Variable count mismatch: {len(card.variable_names)} != {card.num_channels}"
                errors.append(error)
                print(error)
            else:
                print("    ✓ Variable count matches channel count")

            # Check for calibrations
            if card.calibrations:
                print("    ✓ Calibration data available")
                # Validate calibration count
                if len(card.calibrations) != card.num_channels:
                    warning = f"    ⚠ Calibration count mismatch: {len(card.calibrations)} != {card.num_channels}"
                    print(warning)
            else:
                print("    ⚠ No calibration data")

    if errors:
        print("\n✗ VALIDATION FAILED:")
        for error in errors:
            print(f"  {error}")
        return False

    print("\n✓ Configuration validation passed")
    return True


def validate_data_reading(trigger_dir: Path, system: str) -> bool:
    """Validate data reading functionality"""
    print("\n" + "=" * 70)
    print(f"VALIDATING DATA READING - {system}")
    print("=" * 70)

    errors = []

    # Load config to get variable names
    try:
        config = load_trigger_config(trigger_dir, system)
        if config is None:
            errors.append("Could not load config")
            return False

        # Get first analog card and variable
        analog_cards = [c for c in config.cards if c.card_type == "ANA"]
        if not analog_cards:
            print("⚠ No analog cards found, skipping data reading validation")
            return True

        card = analog_cards[0]
        variable = card.variable_names[0]

        print(f"Testing with variable: {variable} (slot {card.slot})")

    except (OSError, ValueError, RuntimeError) as e:
        errors.append(f"Failed to get test variable: {e}")
        print(f"✗ {errors[-1]}")
        return False

    # Test reading by variable name
    print("\nTest 1: Read by variable name")
    try:
        data, timestamp, config = read_trigger_data(
            trigger_dir,
            system,
            variable_name=variable,
            num_samples=1000,  # Read only 1000 samples for speed
        )

        print("  ✓ Data read successfully")
        print(f"    Shape: {data.shape}")
        print("    Expected shape: (1000,)")
        print(f"    Data type: {data.dtype}")
        print(f"    Timestamp: {timestamp}")

        if data.shape != (1000,):
            errors.append(f"    ✗ Unexpected data shape: {data.shape}")

        if data.dtype not in [np.uint16, np.float32, np.float64]:
            errors.append(f"    ✗ Unexpected data type: {data.dtype}")

    except (OSError, ValueError, RuntimeError) as e:
        errors.append(f"  ✗ Failed to read by variable name: {e}")
        print(errors[-1])

    # Test reading by slot
    print("\nTest 2: Read by slot number")
    try:
        data, timestamp, config = read_trigger_data(
            trigger_dir, system, slot=card.slot, num_samples=1000
        )

        print("  ✓ Data read successfully")
        print(f"    Shape: {data.shape}")
        print(f"    Expected shape: (1000, {card.num_channels})")

        if data.shape != (1000, card.num_channels):
            errors.append(f"    ✗ Unexpected data shape: {data.shape}")

    except (OSError, ValueError, RuntimeError) as e:
        errors.append(f"  ✗ Failed to read by slot: {e}")
        print(errors[-1])

    # Test reading full dataset
    print("\nTest 3: Read full dataset (700,000 samples)")
    try:
        data, timestamp, config = read_trigger_data(trigger_dir, system, variable_name=variable)

        print("  ✓ Full dataset read successfully")
        print(f"    Shape: {data.shape}")
        print("    Expected samples: 700,000")

        if len(data) != 700000:
            warning = f"    ⚠ Unexpected sample count: {len(data)}"
            print(warning)

    except (OSError, ValueError, RuntimeError) as e:
        errors.append(f"  ✗ Failed to read full dataset: {e}")
        print(errors[-1])

    if errors:
        print("\n✗ VALIDATION FAILED:")
        for error in errors:
            print(f"  {error}")
        return False

    print("\n✓ Data reading validation passed")
    return True


def validate_trigger_metadata(trigger_dir: Path) -> bool:
    """Validate trigger metadata parsing"""
    print("\n" + "=" * 70)
    print("VALIDATING TRIGGER METADATA")
    print("=" * 70)

    errors = []

    try:
        trigger_info = parse_trigger_directory(trigger_dir)

        print("✓ Metadata parsed successfully")
        print(f"  Trigger name: {trigger_info.trigger_name}")
        print(f"  Timestamp: {trigger_info.timestamp}")
        print(f"  Sample index: {trigger_info.sample_idx}")
        print(f"  RT Block ID: {trigger_info.rtblock_id}")
        print(f"  RT Block Phase: {trigger_info.rtblock_phase}")
        print(f"  PRE samples: {trigger_info.pre_samples}")
        print(f"  POST samples: {trigger_info.post_samples}")
        print(f"  Total samples: {trigger_info.total_samples}")

        # Validate sample counts
        if trigger_info.total_samples != trigger_info.pre_samples + trigger_info.post_samples:
            errors.append("Total samples != PRE + POST")

        if trigger_info.total_samples != 700000:
            warning = f"⚠ Non-standard total samples: {trigger_info.total_samples}"
            print(f"  {warning}")

    except (OSError, ValueError, RuntimeError) as e:
        errors.append(f"Failed to parse metadata: {e}")
        print(f"✗ {errors[-1]}")
        return False

    if errors:
        print("\n✗ VALIDATION FAILED:")
        for error in errors:
            print(f"  {error}")
        return False

    print("\n✓ Metadata validation passed")
    return True


def run_full_validation(trigger_dir: Path) -> bool:
    """Run complete validation suite"""
    print("\n" + "=" * 70)
    print("TRIGGER READER VALIDATION SUITE")
    print("=" * 70)
    print(f"\nTrigger directory: {trigger_dir}")

    results = {}

    # 1. Directory structure
    results["structure"] = validate_directory_structure(trigger_dir)

    if not results["structure"]:
        print("\n✗ Directory structure validation failed. Cannot continue.")
        return False

    # 2. Trigger metadata
    results["metadata"] = validate_trigger_metadata(trigger_dir)

    # 3. Validate each system
    for system in ["FEPC-LNCMI", "FEPC-AUX-LNCMI"]:
        system_dir = trigger_dir / system
        if system_dir.exists():
            results[f"binary_{system}"] = validate_binary_files(trigger_dir, system)
            results[f"config_{system}"] = validate_configuration(trigger_dir, system)
            results[f"data_{system}"] = validate_data_reading(trigger_dir, system)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")

    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test}")

    if passed == total:
        print("\n✓ ALL VALIDATIONS PASSED")
        return True
    else:
        print("\n✗ SOME VALIDATIONS FAILED")
        return False


def main() -> int:
    """Main validation entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate trigger reader implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("trigger_dir", type=Path, help="Path to trigger directory")

    args = parser.parse_args()

    if not args.trigger_dir.exists():
        print(f"Error: Trigger directory not found: {args.trigger_dir}")
        return 1

    success = run_full_validation(args.trigger_dir)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
