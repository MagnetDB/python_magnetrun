"""
Quick Test Script - Verify CFG File Parsing

This script provides a simple way to test if the CFG parser works with your file.
"""

import sys
from pathlib import Path

import pytest


@pytest.fixture
def cfg_path():
    """Provide path to a sample CFG file, skipping if not available."""
    p = Path("data/kHz/2025-11-05/FEPC-LNCMI/HOST_1_DATA.CFG")
    if not p.exists():
        pytest.skip(f"CFG test data not available: {p}")
    return str(p)


def test_cfg_parsing(cfg_path: str) -> None:
    """Test parsing a CFG file."""
    from python_magnetrun.hybrid.kHz.fepc_reader import parse_cfg_file

    config = parse_cfg_file(cfg_path)

    assert config.fepc_name, "fepc_name should not be empty"
    assert config.num_cards > 0, "should have at least one card"
    assert len(config.cards) == config.num_cards

    for card in config.cards:
        assert card.slot is not None
        assert card.card_type is not None
        assert card.num_channels >= 0


def show_raw_file(cfg_path: str, num_lines: int = 20) -> None:
    """Display first few lines of CFG file"""
    print("\n" + "=" * 70)
    print(f"RAW FILE CONTENT (first {num_lines} lines)")
    print("=" * 70)

    try:
        with open(cfg_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if i > num_lines:
                    print("...")
                    break
                print(f"{i:3d}: {line.rstrip()}")
    except OSError as e:
        print(f"Error reading file: {e}")


def _run_cfg_parsing(cfg_path: str) -> bool:
    """Run CFG parsing and return True on success (for standalone use)."""
    print("=" * 70)
    print("FEPC CFG File Parser - Quick Test")
    print("=" * 70)
    print(f"\nTesting file: {cfg_path}")

    if not Path(cfg_path).exists():
        print(f"ERROR: File not found: {cfg_path}")
        return False

    print("File found")

    try:
        from python_magnetrun.hybrid.kHz.fepc_reader import parse_cfg_file

        print("\nParsing configuration...")
        config = parse_cfg_file(cfg_path)

        print("Parsing successful!")
        print("\n" + "-" * 70)
        print("CONFIGURATION SUMMARY")
        print("-" * 70)
        print(f"FEPC Name:    {config.fepc_name}")
        print(f"Total Cards:  {config.num_cards}")
        print(f"Analog Slots: {config.get_analog_slots()}")
        print(f"Digital Slots: {config.get_digital_slots()}")

        print("\n" + "-" * 70)
        print("SLOT DETAILS")
        print("-" * 70)

        for card in config.cards:
            print(f"\nSlot {card.slot}: {card.card_type} card")
            print(f"  - Channels: {card.num_channels}")
            print(f"  - Variables: {len(card.variable_names)}")
            if card.variable_names:
                print(f"  - First variable: {card.variable_names[0]}")
                print(f"  - Last variable: {card.variable_names[-1]}")

        print("\n" + "=" * 70)
        print("TEST PASSED - Configuration parsed successfully!")
        print("=" * 70)

        return True

    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nERROR during parsing: {e}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    else:
        cfg_file = "HOST_1_DATA.CFG"
        print(f"No file specified, using default: {cfg_file}")
        print("Usage: python test-cfg-parser.py <path_to_cfg_file>\n")

    if Path(cfg_file).exists():
        show_raw_file(cfg_file, num_lines=15)

    success = _run_cfg_parsing(cfg_file)

    if success:
        print("\nYour CFG file is compatible with this reader!")
        print("\nNext steps:")
        print("  1. Use cfg_analyzer.py for detailed analysis")
        print("  2. Use example_fepc_usage.py to read data files")
        print("  3. Check README.md for full documentation")
    else:
        print("\nThere was an issue parsing your CFG file.")
        print("\nTroubleshooting:")
        print("  1. Check the raw file content above")
        print("  2. Verify the file format matches the expected structure")
        print("  3. Check for encoding issues or special characters")
        print("  4. You may need to adjust the parse_cfg_file() function")
